#!/usr/bin/env python
# coding: utf-8

"""
This script processes the events of MAGIC calibrated data (*_Y_*.root) with the
MARS-like image cleaning and computes the DL1 parameters (i.e., Hillas, timing
and leakage parameters). It saves only the events that all the DL1 parameters
are successfully reconstructed. The telescope IDs are reset to the following
ones for the combined analysis with LST-1, whose telescope ID is 1:
MAGIC-I: tel_id = 2,  MAGIC-II: tel_id = 3

When an input is real data, the script searches for all the subrun files belonging
to the same observation ID and stored in the same directory as an input subrun file.
Then it reads drive reports from the files and uses the information to reconstruct
the telescope pointing direction. Thus, it is best to store all the files in the
same directory.

If the "--process-run" argument is given, it not only reads drive reports but also
processes all the events of the subrun files at once.

Usage:
$ python magic_calib_to_dl1.py
--input-file ./data/calibrated/20201216_M1_05093711.001_Y_CrabNebula-W0.40+035.root
--output-dir ./data/dl1
--config-file ./config.yaml
(--process-run)
"""

import argparse
import logging
import re
import time
import warnings
from pathlib import Path

import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import angular_separation
from ctapipe.core import Container, Field
from ctapipe.image import (
    hillas_parameters,
    leakage_parameters,
    number_of_islands,
    timing_parameters,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import HDF5TableWriter
from ctapipe_io_magic import MAGICEventSource
from magicctapipe.image import MAGICClean
from magicctapipe.utils import calculate_disp, calculate_impact

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Ignore RuntimeWarnings appeared in the image cleaning:
warnings.simplefilter("ignore", category=RuntimeWarning)

SEC2NSEC = 1e9

pedestal_types = np.array(
    [
        "fundamental",
        "from_extractor",
        "from_extractor_rndm",
    ]
)

__all__ = [
    "EventInfoContainer",
    "SimEventInfoContainer",
    "magic_calib_to_dl1",
]


class EventInfoContainer(Container):
    """Container to store event information"""

    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    pointing_alt = Field(-1, "Telescope pointing altitude", u.rad)
    pointing_az = Field(-1, "Telescope pointing azimuth", u.rad)
    time_sec = Field(-1, "Event trigger time second", u.s)
    time_nanosec = Field(-1, "Event trigger time nanosecond", u.ns)
    time_diff = Field(-1, "Event trigger time difference from the previous event", u.s)
    n_pixels = Field(-1, "Number of pixels of a cleaned image")
    n_islands = Field(-1, "Number of islands of a cleaned image")


class SimEventInfoContainer(Container):
    """Container to store simulated event information"""

    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    pointing_alt = Field(-1, "Telescope pointing altitude", u.rad)
    pointing_az = Field(-1, "Telescope pointing azimuth", u.rad)
    true_energy = Field(-1, "Simulated event true energy", u.TeV)
    true_alt = Field(-1, "Simulated event true altitude", u.deg)
    true_az = Field(-1, "Simulated event true azimuth", u.deg)
    true_disp = Field(-1, "Simulated event true disp", u.deg)
    true_core_x = Field(-1, "Simulated event true core x", u.m)
    true_core_y = Field(-1, "Simulated event true core y", u.m)
    true_impact = Field(-1, "Simulated event true impact", u.m)
    off_axis = Field(-1, "Simulated event off-axis angle", u.deg)
    n_pixels = Field(-1, "Number of pixels of a cleaned image")
    n_islands = Field(-1, "Number of islands of a cleaned image")


def magic_calib_to_dl1(
    input_file,
    output_dir,
    config,
    process_run=False,
):
    """
    Processes MAGIC calibrated events and computes the DL1 parameters.

    Parameters
    ----------
    input_file: str
        Path to an input MAGIC calibrated data file
    output_dir: str
        Path to a directory where to save an output DL1 data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    process_run: bool
        If True, it processes the events of all the subrun files at once
    """

    logger.info(f"\nInput file:\n{input_file}")
    logger.info(f"\nProcess run: {process_run}")

    # Load the input file:
    event_source = MAGICEventSource(input_file, process_run=process_run)

    obs_id = event_source.obs_ids[0]
    tel_id = event_source.telescope
    is_simulation = event_source.is_simulation

    logger.info(f"\nObservation ID: {obs_id}")
    logger.info(f"Telescope ID: {tel_id}")
    logger.info(f"Is simulation: {is_simulation}")

    if process_run:
        logger.info("\nProcess the following data:")
        for root_file in event_source.file_list:
            logger.info(root_file)

    if not is_simulation:
        time_diffs = event_source.event_time_diffs

    subarray = event_source.subarray
    camera_geom = subarray.tel[tel_id].camera.geometry
    tel_position = subarray.positions[tel_id]

    # Configure the MAGIC image cleaning:
    config_clean = config["MAGIC"]["magic_clean"]

    logger.info("\nMAGIC image cleaning:")

    if is_simulation and (config_clean["find_hotpixels"] is not False):
        logger.warning(
            "Hot pixels do not exist in a simulation. "
            "Setting the 'find_hotpixels' option to False..."
        )
        config_clean.update({"find_hotpixels": False})

    for key, value in config_clean.items():
        logger.info(f"\t{key}: {value}")

    find_hotpixels = config_clean["find_hotpixels"]

    if find_hotpixels:
        pedestal_type = config_clean.pop("pedestal_type")
        i_ped_type = np.where(pedestal_types == pedestal_type)[0][0]

    magic_clean = MAGICClean(camera_geom, config_clean)

    # Prepare for saving data to an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if is_simulation:
        regex = r"GA_M\d_(\S+)_\d_\d+_Y_*"
        file_name = Path(input_file).name

        parser = re.findall(regex, file_name)[0]
        output_file = f"{output_dir}/dl1_M{tel_id}_GA_{parser}.Run{obs_id}.h5"

    else:
        if process_run:
            output_file = f"{output_dir}/dl1_M{tel_id}.Run{obs_id:08}.h5"
        else:
            subrun_id = event_source.metadata["subrun_number"][0]
            output_file = f"{output_dir}/dl1_M{tel_id}.Run{obs_id:08}.{subrun_id:03}.h5"

    # Start processing the events:
    logger.info("\nProcessing the events...")

    with HDF5TableWriter(output_file, group_name="events", mode="w") as writer:

        for event in event_source:

            if event.count % 100 == 0:
                logger.info(f"{event.count} events")

            # Apply the image cleaning:
            image = event.dl1.tel[tel_id].image
            peak_time = event.dl1.tel[tel_id].peak_time

            if find_hotpixels:
                pixel_status = event.mon.tel[tel_id].pixel_status
                dead_pixels = pixel_status.hardware_failing_pixels[0]
                badrms_pixels = pixel_status.pedestal_failing_pixels[i_ped_type]
                unsuitable_mask = np.logical_or(dead_pixels, badrms_pixels)
            else:
                unsuitable_mask = None

            signal_pixels, image, peak_time = magic_clean.clean_image(
                image,
                peak_time,
                unsuitable_mask,
            )

            image_cleaned = image.copy()
            image_cleaned[~signal_pixels] = 0

            peak_time_cleaned = peak_time.copy()
            peak_time_cleaned[~signal_pixels] = 0

            n_pixels = np.count_nonzero(signal_pixels)
            n_islands, _ = number_of_islands(camera_geom, signal_pixels)

            if n_pixels == 0:
                logger.warning(
                    f"--> {event.count} event (event ID: {event.index.event_id}): "
                    "Could not survive the image cleaning."
                )
                continue

            # Try to parametrize the image:
            try:
                hillas_params = hillas_parameters(
                    camera_geom,
                    image_cleaned,
                )

                timing_params = timing_parameters(
                    camera_geom,
                    image_cleaned,
                    peak_time_cleaned,
                    hillas_params,
                    signal_pixels,
                )

                leakage_params = leakage_parameters(
                    camera_geom,
                    image_cleaned,
                    signal_pixels,
                )

            except Exception:
                logger.warning(
                    f"--> {event.count} event (event ID: {event.index.event_id}): "
                    "Image parametrization failed."
                )
                continue

            if is_simulation:

                # Calculate the off-axis angle:
                off_axis = angular_separation(
                    lon1=event.pointing.tel[tel_id].azimuth,
                    lat1=event.pointing.tel[tel_id].altitude,
                    lon2=event.simulation.shower.az,
                    lat2=event.simulation.shower.alt,
                )

                # Calculate the DISP parameter:
                true_disp = calculate_disp(
                    pointing_alt=event.pointing.tel[tel_id].altitude,
                    pointing_az=event.pointing.tel[tel_id].azimuth,
                    shower_alt=event.simulation.shower.alt,
                    shower_az=event.simulation.shower.az,
                    cog_x=hillas_params.x,
                    cog_y=hillas_params.y,
                    camera_frame=camera_geom.frame,
                )

                # Calculate the impact parameter:
                true_impact = calculate_impact(
                    core_x=event.simulation.shower.core_x,
                    core_y=event.simulation.shower.core_y,
                    az=event.simulation.shower.az,
                    alt=event.simulation.shower.alt,
                    tel_pos_x=tel_position[0],
                    tel_pos_y=tel_position[1],
                    tel_pos_z=tel_position[2],
                )

                # Set the event information to the container:
                event_info = SimEventInfoContainer(
                    obs_id=event.index.obs_id,
                    event_id=event.index.event_id,
                    pointing_alt=event.pointing.tel[tel_id].altitude,
                    pointing_az=event.pointing.tel[tel_id].azimuth,
                    true_energy=event.simulation.shower.energy,
                    true_alt=event.simulation.shower.alt,
                    true_az=event.simulation.shower.az,
                    true_disp=true_disp,
                    true_core_x=event.simulation.shower.core_x,
                    true_core_y=event.simulation.shower.core_y,
                    true_impact=true_impact,
                    off_axis=off_axis,
                    n_pixels=n_pixels,
                    n_islands=n_islands,
                )

            else:
                timestamp = event.trigger.tel[tel_id].time.to_value(
                    format="unix",
                    subfmt="long",
                )

                # To keep the precision of a timestamp for the event coincidence
                # with LST-1, here we set the integral and fractional parts
                # separately as "time_sec" and "time_nanosec":
                fractional, integral = np.modf(timestamp)

                time_sec = u.Quantity(int(np.round(integral)), u.s)
                time_nanosec = u.Quantity(int(np.round(fractional * SEC2NSEC)), u.ns)

                time_diff = time_diffs[event.count]

                # Set the event information to the container:
                event_info = EventInfoContainer(
                    obs_id=event.index.obs_id,
                    event_id=event.index.event_id,
                    pointing_alt=event.pointing.tel[tel_id].altitude,
                    pointing_az=event.pointing.tel[tel_id].azimuth,
                    time_sec=time_sec,
                    time_nanosec=time_nanosec,
                    time_diff=time_diff,
                    n_pixels=n_pixels,
                    n_islands=n_islands,
                )

            # Reset the telescope IDs:
            if tel_id == 1:
                event_info.tel_id = 2  # MAGIC-I

            elif tel_id == 2:
                event_info.tel_id = 3  # MAGIC-II

            # Save the parameters to the output file:
            writer.write(
                "parameters",
                (event_info, hillas_params, timing_params, leakage_params),
            )

        n_events_processed = event.count + 1
        logger.info(f"\nIn total {n_events_processed} events are processed.")

    # Reset the telescope IDs of the subarray descriptions:
    tel_positions = {
        2: subarray.positions[1],  # MAGIC-I
        3: subarray.positions[2],  # MAGIC-II
    }

    tel_descriptions = {
        2: event_source.subarray.tel[1],  # MAGIC-I
        3: event_source.subarray.tel[2],  # MAGIC-II
    }

    subarray_magic = SubarrayDescription("MAGIC-Array", tel_positions, tel_descriptions)
    subarray_magic.to_hdf(output_file)

    if is_simulation:
        # Save the simulation configuration:
        with HDF5TableWriter(output_file, group_name="simulation", mode="a") as writer:
            writer.write("config", event_source.simulation_config[obs_id])

    logger.info("\nOutput file:")
    logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        "-i",
        dest="input_file",
        type=str,
        required=True,
        help="Path to an input MAGIC calibrated data file.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL1 data file.",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config.yaml",
        help="Path to a yaml configuration file.",
    )

    parser.add_argument(
        "--process-run",
        dest="process_run",
        action="store_true",
        help="Process the events of all the subrun files at once.",
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Process the input data:
    magic_calib_to_dl1(args.input_file, args.output_dir, config, args.process_run)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
