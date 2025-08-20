#!/usr/bin/env python
# coding: utf-8

"""
This script processes the events of MAGIC calibrated data (*_Y_*.root)
with the MARS-like image cleaning and computes the DL1 parameters, i.e.,
Hillas, timing and leakage parameters. It saves only the events that all
the DL1 parameters are successfully reconstructed.

When the input is real data, it searches for all the subrun files with
the same observation ID and stored in the same directory as the input
subrun file. Then, it reads their drive reports and uses the information
to reconstruct the telescope pointing direction. Since the accuracy of
the reconstruction improves, it is recommended to store all the subrun
files in the same directory.

If the `--process-run` argument is given, it not only reads the drive
reports but also processes all the events of the subrun files at once.

If the `--magic-only` argument is given, the processing is unchanged,
but the subarray will contain only the MAGIC telescopes. This is needed
when performing a MAGIC-only analysis using the pipeline. In all other
cases, LST is included in the subarray.

Please note that it is also possible to process SUM trigger data with
this script, but since the MaTaJu cleaning is not yet implemented in
this pipeline, it applies the standard cleaning instead.

Usage:
$ python magic_calib_to_dl1.py
--input-file calib/20201216_M1_05093711.001_Y_CrabNebula-W0.40+035.root
(--output-dir dl1)
(--config-file config.yaml)
(--magic-only)
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
from ctapipe.containers import DL1CameraContainer
from ctapipe.image import (
    concentration_parameters,
    hillas_parameters,
    leakage_parameters,
    number_of_islands,
    timing_parameters,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import HDF5TableWriter
from ctapipe_io_lst import LSTEventSource
from ctapipe_io_magic import MAGICEventSource

from magicctapipe.image import MAGICClean
from magicctapipe.io import (
    RealEventInfoContainer,
    SimEventInfoContainer,
    check_input_list,
    format_object,
)
from magicctapipe.utils import calculate_disp, calculate_impact

__all__ = ["magic_calib_to_dl1"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Ignore runtime warnings appeared during the image cleaning
warnings.simplefilter("ignore", category=RuntimeWarning)

# The pedestal types to find bad RMS pixels
PEDESTAL_TYPES = ["fundamental", "from_extractor", "from_extractor_rndm"]
TEL_COMBINATIONS = {
    "LST1_M1": [1, 2],  # combo_type = 0
    "LST1_M1_M2": [1, 2, 3],  # combo_type = 1
    "LST1_M2": [1, 3],  # combo_type = 2
    "M1_M2": [2, 3],  # combo_type = 3
}  # TODO: REMOVE WHEN SWITCHING TO THE NEW RFs IMPLEMENTTATION (1 RF PER TELESCOPE)


def magic_calib_to_dl1(
    input_file, output_dir, config, max_events, magic_only=False, process_run=False
):
    """
    Processes the events of MAGIC calibrated data and computes the DL1 parameters.

    Parameters
    ----------
    input_file : str
        Path to an input MAGIC calibrated data file
    output_dir : str
        Path to a directory where to save an output DL1 data file
    config : dict
        Configuration for the LST-1 + MAGIC analysis
    max_events : int
        Maximum number of events to process
    magic_only : bool, optional
        If `True`, it will store subarray information only for the MAGIC
        telescopes. This is needed if the pipeline will be used for
        MAGIC-only analysis.
    process_run : bool, optional
        If `True`, it processes the events of all the subrun files
        found in the same directory of the input subrun file at once
        (applicable only to real data)
    """

    # Load the input file
    logger.info(f"\nInput file: {input_file}")

    event_source = MAGICEventSource(
        input_file, process_run=process_run, max_events=max_events
    )

    is_simulation = event_source.is_simulation
    logger.info(f"\nIs simulation: {is_simulation}")

    obs_id = event_source.obs_ids[0]
    tel_id = event_source.telescope
    logger.info(f"\nObservation ID: {obs_id}")
    logger.info(f"Telescope ID: {tel_id}")

    is_stereo_trigger = event_source.is_stereo
    is_sum_trigger = event_source.is_sumt

    logger.info(f"\nIs stereo trigger: {is_stereo_trigger}")
    logger.info(f"Is SUM trigger: {is_sum_trigger}")

    if is_sum_trigger:
        logger.warning(
            "\nWARNING: The MaTaJu cleaning is not yet implemented. "
            "Will apply the standard image cleaning instead."
        )

    if not is_simulation:
        logger.info("\nThe following files are found to read drive reports:")
        for subrun_file in event_source.file_list_drive:
            logger.info(subrun_file)

        logger.info(f"\nProcess run: {process_run}")
        time_diffs = event_source.event_time_diffs

    subarray = event_source.subarray

    camera_geom = subarray.tel[tel_id].camera.geometry
    tel_position = subarray.positions[tel_id]

    # Configure the MAGIC image cleaning
    config_clean = config["MAGIC"]["magic_clean"]

    logger.info("\nMAGIC image cleaning:")
    logger.info(format_object(config_clean))

    magic_clean = MAGICClean(camera_geom, config_clean)

    if config_clean["find_hotpixels"]:
        # Get the index of the pedestal type
        i_ped_type = PEDESTAL_TYPES.index(config_clean["pedestal_type"])

    # Prepare for saving data to an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if is_simulation:
        regex = r"GA_M\d_(\S+)_\d_\d+_Y_*"
        input_file_name = Path(input_file).name

        zenith_range = re.findall(regex, input_file_name)[0]
        output_file = f"{output_dir}/dl1_M{tel_id}_GA_{zenith_range}.Run{obs_id}.h5"

    else:
        if process_run:
            output_file = f"{output_dir}/dl1_M{tel_id}.Run{obs_id:08}.h5"
        else:
            subrun_id = event_source.metadata["subrun_number"][0]
            output_file = f"{output_dir}/dl1_M{tel_id}.Run{obs_id:08}.{subrun_id:03}.h5"

    assigned_tel_ids = config[
        "mc_tel_ids"
    ]  # This variable becomes the dictionary {'LST-1': 1, 'MAGIC-I': 2, 'MAGIC-II': 3} or similar

    subarray_lst = LSTEventSource.create_subarray()

    # Reset the telescope IDs of the subarray description
    if not magic_only:
        tel_positions_magic_lst = {
            assigned_tel_ids["LST-1"]: [-8.09, 77.13, 0.78] * u.m,  # LST-1
            assigned_tel_ids["MAGIC-I"]: [39.3, -62.55, -0.97] * u.m,  # MAGIC-I
            assigned_tel_ids["MAGIC-II"]: [-31.21, -14.57, 0.2] * u.m,  # MAGIC-II
        }

        tel_descriptions_magic_lst = {
            # dummy telescope description for LST-1, same as MAGIC-I
            assigned_tel_ids["LST-1"]: subarray_lst.tel[1],  # LST-1
            assigned_tel_ids["MAGIC-I"]: subarray.tel[1],  # MAGIC-I
            assigned_tel_ids["MAGIC-II"]: subarray.tel[2],  # MAGIC-II
        }

        subarray_magic = SubarrayDescription(
            "MAGIC-LST-Array", tel_positions_magic_lst, tel_descriptions_magic_lst
        )
    else:
        tel_positions_magic = {
            assigned_tel_ids["MAGIC-I"]: subarray.positions[1],  # MAGIC-I
            assigned_tel_ids["MAGIC-II"]: subarray.positions[2],  # MAGIC-II
        }

        tel_descriptions_magic = {
            assigned_tel_ids["MAGIC-I"]: subarray.tel[1],  # MAGIC-I
            assigned_tel_ids["MAGIC-II"]: subarray.tel[2],  # MAGIC-II
        }

        subarray_magic = SubarrayDescription(
            "MAGIC-Array", tel_positions_magic, tel_descriptions_magic
        )

    save_images = config.get("save_images", False)
    if save_images:
        dl1cont = DL1CameraContainer(prefix="")
    # Loop over every shower event
    logger.info("\nProcessing the events...")

    with HDF5TableWriter(
        output_file, group_name="events", mode="w", add_prefix=True
    ) as writer:
        for event in event_source:
            if event.count % 100 == 0:
                logger.info(f"{event.count} events")

            if config_clean["find_hotpixels"]:
                # Find dead and bad RMS pixels
                pixel_status = event.mon.tel[tel_id].pixel_status
                dead_pixels = pixel_status.hardware_failing_pixels[0]
                badrms_pixels = pixel_status.pedestal_failing_pixels[i_ped_type]
                unsuitable_mask = np.logical_or(dead_pixels, badrms_pixels)

            else:
                unsuitable_mask = None

            # Apply the image cleaning
            signal_pixels, image, peak_time = magic_clean.clean_image(
                event_image=event.dl1.tel[tel_id].image,
                event_pulse_time=event.dl1.tel[tel_id].peak_time,
                unsuitable_mask=unsuitable_mask,
            )

            if not any(signal_pixels):
                logger.info(
                    f"--> {event.count} event (event ID: {event.index.event_id}) "
                    "could not survive the image cleaning. Skipping..."
                )
                continue

            n_pixels = np.count_nonzero(signal_pixels)
            n_islands, _ = number_of_islands(camera_geom, signal_pixels)

            camera_geom_masked = camera_geom[signal_pixels]
            image_masked = image[signal_pixels]
            peak_time_masked = peak_time[signal_pixels]

            if any(image_masked < 0):
                logger.info(
                    f"--> {event.count} event (event ID: {event.index.event_id}) "
                    "cannot be parametrized due to the pixels with negative charges. "
                    "Skipping..."
                )
                continue

            # Parametrize the image
            hillas_params = hillas_parameters(camera_geom_masked, image_masked)

            timing_params = timing_parameters(
                camera_geom_masked, image_masked, peak_time_masked, hillas_params
            )

            if np.isnan(timing_params.slope):
                logger.info(
                    f"--> {event.count} event (event ID: {event.index.event_id}) "
                    "failed to extract finite timing parameters. Skipping..."
                )
                continue

            leakage_params = leakage_parameters(camera_geom, image, signal_pixels)
            conc_params = concentration_parameters(camera_geom, image, hillas_params)

            if is_simulation:
                # Calculate additional parameters
                true_disp = calculate_disp(
                    pointing_alt=event.pointing.tel[tel_id].altitude,
                    pointing_az=event.pointing.tel[tel_id].azimuth,
                    shower_alt=event.simulation.shower.alt,
                    shower_az=event.simulation.shower.az,
                    cog_x=hillas_params.x,
                    cog_y=hillas_params.y,
                    camera_frame=camera_geom.frame,
                )

                true_impact = calculate_impact(
                    shower_alt=event.simulation.shower.alt,
                    shower_az=event.simulation.shower.az,
                    core_x=event.simulation.shower.core_x,
                    core_y=event.simulation.shower.core_y,
                    tel_pos_x=tel_position[0],
                    tel_pos_y=tel_position[1],
                    tel_pos_z=tel_position[2],
                )

                off_axis = angular_separation(
                    lon1=event.pointing.tel[tel_id].azimuth,
                    lat1=event.pointing.tel[tel_id].altitude,
                    lon2=event.simulation.shower.az,
                    lat2=event.simulation.shower.alt,
                )

                # Set the simulated event information to the container
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
                # With the UNIX format and the "long" type, we can get a
                # timestamp without being affected by the rounding issue
                timestamp = event.trigger.tel[tel_id].time.to_value(
                    format="unix", subfmt="long"
                )

                # To keep the precision for the coincidence with LST-1,
                # we save the integral and fractional parts separately
                fractional, integral = np.modf(timestamp)

                time_sec = u.Quantity(integral, unit="s", dtype=int)

                time_nanosec = u.Quantity(fractional, unit="s").to("ns")
                time_nanosec = u.Quantity(time_nanosec.round(), dtype=int)

                # Set the real event information to the container
                event_info = RealEventInfoContainer(
                    obs_id=event.index.obs_id,
                    event_id=event.index.event_id,
                    pointing_alt=event.pointing.tel[tel_id].altitude,
                    pointing_az=event.pointing.tel[tel_id].azimuth,
                    time_sec=time_sec,
                    time_nanosec=time_nanosec,
                    time_diff=time_diffs[event.count],
                    n_pixels=n_pixels,
                    n_islands=n_islands,
                )

            tel_ids_new_assignments = {
                1: assigned_tel_ids["MAGIC-I"],
                2: assigned_tel_ids["MAGIC-II"],
                3: assigned_tel_ids["LST-1"],
            }

            # Reset the telescope IDs
            event_info.tel_id = tel_ids_new_assignments[tel_id]

            # encode tels_with_trigger as an int value
            # that can be decoded later as a binary
            # tels_with_trigger = sum_{tel_id} 2**tel_id
            # where tel_id is only for those triggered
            tels_with_trigger_binary_int = np.sum(
                2
                ** np.array(
                    [
                        tel_ids_new_assignments[tel_idx]
                        for tel_idx in event.trigger.tels_with_trigger
                    ]
                )
            )

            event_info.tels_with_trigger = tels_with_trigger_binary_int

            # Save the parameters to the output file
            # Setting all the prefixes except of concentration to empty string
            event_info.prefix = ""
            hillas_params.prefix = ""
            timing_params.prefix = ""
            leakage_params.prefix = ""
            writer.write(
                "parameters",
                (event_info, hillas_params, timing_params, leakage_params, conc_params),
            )

            if save_images:
                dl1cont.image = image
                dl1cont.peak_time = peak_time
                dl1cont.image_mask = signal_pixels
                dl1cont.is_valid = True
                writer.write(table_name="dl1/image", containers=[event_info, dl1cont])

        n_events_processed = event.count + 1
        logger.info(f"\nIn total {n_events_processed} events are processed.")

    # Save the subarray description
    subarray_magic.to_hdf(output_file)

    if is_simulation:
        # Save the simulation configuration
        with HDF5TableWriter(output_file, group_name="simulation", mode="a") as writer:
            writer.write("config", event_source.simulation_config[obs_id])

    logger.info(f"\nOutput file: {output_file}")


def main():
    """Main function."""
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        "-i",
        dest="input_file",
        type=str,
        required=True,
        help="Path to an input MAGIC calibrated data file",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL1 data file",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config.yaml",
        help="Path to a configuration file",
    )

    parser.add_argument(
        "--max-evt",
        "-m",
        dest="max_events",
        type=int,
        default=None,
        help="Max. number of processed showers",
    )

    parser.add_argument(
        "--magic-only",
        dest="magic_only",
        action="store_true",
        help="Process file(s) for MAGIC-only analysis",
    )

    parser.add_argument(
        "--process-run",
        dest="process_run",
        action="store_true",
        help="Process the events of all the subrun files at once",
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Checking if the input telescope list is properly organized:
    check_input_list(config)

    # Process the input data
    magic_calib_to_dl1(
        args.input_file,
        args.output_dir,
        config,
        args.max_events,
        args.magic_only,
        args.process_run,
    )
    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
