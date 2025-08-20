#!/usr/bin/env python
# coding: utf-8

"""
This script processes LST and MAGIC events of simtel MC DL0 data
(*.simtel.gz) and computes the DL1 parameters, i.e., Hillas, timing and
leakage parameters. It saves only the events where all the DL1 parameters
are successfully reconstructed.

Since it cannot identify the telescopes from the input file, please assign
the correct telescope ID to each telescope in the configuration file.

The telescope coordinates will be converted to those
relative to the center of the LST and MAGIC positions (including the
altitude) for the convenience of the geometrical stereo reconstruction.

Usage:
$ python lst1_magic_mc_dl0_to_dl1.py
--input-file dl0/gamma_40deg_90deg_run1.simtel.gz
(--output-dir dl1)
(--config-file config_step1.yaml)
"""

import argparse  # Parser for command-line options, arguments etc
import logging  # Used to manage the log file
import re
import time
from pathlib import Path

import numpy as np
import yaml
from astropy.coordinates import Angle, angular_separation
from ctapipe.calib import CameraCalibrator
from ctapipe.image import (
    concentration_parameters,
    hillas_parameters,
    leakage_parameters,
    number_of_islands,
    timing_parameters,
)
from ctapipe.instrument import FocalLengthKind
from ctapipe.io import EventSource, HDF5TableWriter
from traitlets.config import Config

from magicctapipe.image import MAGICClean
from magicctapipe.image.calib import calibrate
from magicctapipe.io import SimEventInfoContainer, check_input_list, format_object
from magicctapipe.utils import calculate_disp, calculate_impact

__all__ = ["mc_dl0_to_dl1"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The CORSIKA particle types #CORSIKA simulates Cherenkov light
PARTICLE_TYPES = {1: "gamma", 3: "electron", 14: "proton", 402: "helium"}


def mc_dl0_to_dl1(input_file, output_dir, config, focal_length):
    """
    Processes LST-1 and MAGIC events of simtel MC DL0 data and computes
    the DL1 parameters.

    Parameters
    ----------
    input_file : str
        Path to an input simtel MC DL0 data file
    output_dir : str
        Path to a directory where to save an output DL1 data file
    config : dict
        Configuration for the LST-1 + MAGIC analysis
    focal_length : str
        Focal length choice, effective or equivalent
    """

    assigned_tel_ids = config[
        "mc_tel_ids"
    ]  # This variable becomes the dictionary {'LST-1': 1, 'MAGIC-I': 2, 'MAGIC-II': 3}

    logger.info(
        "\nAssigned telescope IDs:"
    )  # Here we are just adding infos to the log file
    logger.info(format_object(assigned_tel_ids))

    # Load the input file
    logger.info(f"\nInput file: {input_file}")

    if focal_length == "effective":
        focal_length = FocalLengthKind.EFFECTIVE
    elif focal_length == "equivalent":
        focal_length = FocalLengthKind.EQUIVALENT
    else:
        raise ValueError(
            f"Accepted choices for focal_length are 'effective' or 'equivalent'.\n"
            f"Chosen value: {focal_length}."
        )

    event_source = EventSource(
        input_file,
        allowed_tels=list(
            filter(lambda check_id: check_id > 0, assigned_tel_ids.values())
        ),  # Here we load the events for all telescopes with ID > 0.
        focal_length_choice=focal_length,
    )

    obs_id = event_source.obs_ids[0]
    subarray = event_source.subarray

    tel_descriptions = subarray.tel
    tel_positions = subarray.positions

    logger.info("\nSubarray description:")
    logger.info(format_object(tel_descriptions))

    camera_geoms = {}
    for tel_id, telescope in tel_descriptions.items():
        camera_geoms[tel_id] = telescope.camera.geometry

    # Configure the LST event processors
    config_lst = config["LST"]
    config_lst["mc_tel_ids"] = assigned_tel_ids

    logger.info("\nLST image extractor:")
    logger.info(format_object(config_lst["image_extractor"]))

    extractor_type_lst = config_lst["image_extractor"].pop("type")
    config_extractor_lst = {extractor_type_lst: config_lst["image_extractor"]}

    calibrator_lst = CameraCalibrator(
        image_extractor_type=extractor_type_lst,
        config=Config(config_extractor_lst),
        subarray=subarray,
    )

    logger.info("\nLST NSB modifier:")
    logger.info(format_object(config_lst["increase_nsb"]))

    logger.info("\nLST PSF modifier:")
    logger.info(format_object(config_lst["increase_psf"]))

    logger.info("\nLST tailcuts cleaning:")
    logger.info(format_object(config_lst["tailcuts_clean"]))

    logger.info("\nLST time delta cleaning:")
    logger.info(format_object(config_lst["time_delta_cleaning"]))

    logger.info("\nLST dynamic cleaning:")
    logger.info(format_object(config_lst["dynamic_cleaning"]))

    use_only_main_island = config_lst["use_only_main_island"]
    logger.info(f"\nLST use only main island: {use_only_main_island}")

    # Configure the MAGIC event processors
    config_magic = config["MAGIC"]
    config_magic["mc_tel_ids"] = assigned_tel_ids

    logger.info("\nMAGIC image extractor:")
    logger.info(format_object(config_magic["image_extractor"]))

    for imagic in [1, 2]:
        if f"increase_nsb_m{imagic}" in config_magic:
            logger.info("\nMAGIC-" + imagic * "I" + " NSB modifier:")
            logger.info(format_object(config_magic[f"increase_nsb_m{imagic}"]))

    extractor_type_magic = config_magic["image_extractor"].pop("type")
    config_extractor_magic = {extractor_type_magic: config_magic["image_extractor"]}

    calibrator_magic = CameraCalibrator(
        image_extractor_type=extractor_type_magic,
        config=Config(config_extractor_magic),
        subarray=subarray,
    )

    logger.info("\nMAGIC charge correction:")
    logger.info(format_object(config_magic["charge_correction"]))

    if config_magic["magic_clean"]["find_hotpixels"]:
        logger.warning(
            "\nWARNING: Hot pixels do not exist in a simulation. "
            "Setting the `find_hotpixels` option to False..."
        )
        config_magic["magic_clean"].update({"find_hotpixels": False})

    logger.info("\nMAGIC image cleaning:")
    logger.info(format_object(config_magic["magic_clean"]))

    # Prepare for saving data to an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    sim_config = event_source.simulation_config[obs_id]
    corsika_inputcard = event_source.file_.corsika_inputcards[0].decode()

    regex = r".*\nPRMPAR\s+(\d+)\s+.*"

    particle_id = int(re.findall(regex, corsika_inputcard)[0])
    particle_type = PARTICLE_TYPES.get(particle_id, "unknown")

    zenith = 90 - sim_config["max_alt"].to_value("deg")
    azimuth = Angle(sim_config["max_az"]).wrap_at("360 deg").degree
    logger.info(np.asarray(list(assigned_tel_ids.values())))
    LSTs_IDs = np.asarray(list(assigned_tel_ids.values())[0:4])
    LSTs_in_use = (
        np.where(LSTs_IDs > 0)[0] + 1
    )  # Here we select which LSTs are/is in use

    if len(LSTs_in_use) == 0:
        LSTs_in_use = "".join(str(k) for k in LSTs_in_use)
    elif len(LSTs_in_use) > 0:
        LSTs_in_use = "LST" + "_LST".join(str(k) for k in LSTs_in_use)

    MAGICs_IDs = np.asarray(list(assigned_tel_ids.values())[4:6])
    MAGICs_in_use = (
        np.where(MAGICs_IDs > 0)[0] + 1
    )  # Here we select which MAGICs are/is in use

    if len(MAGICs_in_use) == 0:
        MAGICs_in_use = "".join(str(k) for k in MAGICs_in_use)
    elif len(MAGICs_in_use) > 0:
        MAGICs_in_use = "MAGIC" + "_MAGIC".join(str(k) for k in MAGICs_in_use)
    magic_clean = {}
    for k in MAGICs_IDs:
        if k > 0:
            magic_clean[k] = MAGICClean(camera_geoms[k], config_magic["magic_clean"])

    output_file = (
        f"{output_dir}/dl1_{particle_type}_zd_{zenith.round(3)}deg_"
        f"az_{azimuth.round(3)}deg_{LSTs_in_use}_{MAGICs_in_use}_run{obs_id}.h5"
    )  # The files are saved with the names of all telescopes involved

    # Loop over every shower event
    logger.info("\nProcessing the events...")

    with HDF5TableWriter(
        output_file, group_name="events", mode="w", add_prefix=True
    ) as writer:
        for event in event_source:
            if event.count % 100 == 0:
                logger.info(f"{event.count} events")

            tels_with_trigger = event.trigger.tels_with_trigger

            # Check if the event triggers both M1 and M2 or not
            magic_stereo = (set(MAGICs_IDs).issubset(set(tels_with_trigger))) and (
                MAGICs_in_use == "MAGIC1_MAGIC2"
            )  # If both have trigger, then magic_stereo = True

            for tel_id in tels_with_trigger:
                if (
                    tel_id in LSTs_IDs
                ):  # If the ID is in the LST list, we call calibrate on the LST()
                    # Calibrate the LST-1 event
                    signal_pixels, image, peak_time = calibrate(
                        event=event,
                        tel_id=tel_id,
                        obs_id=obs_id,
                        config=config_lst,
                        camera_geoms=camera_geoms,
                        calibrator=calibrator_lst,
                        is_lst=True,
                    )
                elif tel_id in MAGICs_IDs:
                    # Calibrate the MAGIC event
                    signal_pixels, image, peak_time = calibrate(
                        event=event,
                        tel_id=tel_id,
                        obs_id=obs_id,
                        config=config_magic,
                        magic_clean=magic_clean,
                        calibrator=calibrator_magic,
                        is_lst=False,
                    )
                else:
                    logger.info(
                        f"--> Telescope ID {tel_id} not in LST list or MAGIC list. Please check if the IDs are OK in the configuration file"
                    )
                if not any(
                    signal_pixels
                ):  # So: if there is no event, we skip it and go back to the loop in the next event
                    logger.info(
                        f"--> {event.count} event (event ID: {event.index.event_id}, "
                        f"telescope {tel_id}) could not survive the image cleaning. "
                        "Skipping..."
                    )
                    continue

                n_pixels = np.count_nonzero(signal_pixels)
                n_islands, _ = number_of_islands(camera_geoms[tel_id], signal_pixels)

                camera_geom_masked = camera_geoms[tel_id][signal_pixels]
                image_masked = image[signal_pixels]
                peak_time_masked = peak_time[signal_pixels]

                if any(image_masked < 0):
                    logger.info(
                        f"--> {event.count} event (event ID: {event.index.event_id}, "
                        f"telescope {tel_id}) cannot be parametrized due to the pixels "
                        "with negative charges. Skipping..."
                    )
                    continue

                # Parametrize the image
                hillas_params = hillas_parameters(camera_geom_masked, image_masked)

                #
                if any(np.isnan(value) for value in hillas_params.values()):
                    logger.info(
                        f"--> {event.count} event (event ID: {event.index.event_id}, "
                        f"telescope {tel_id}): non-valid Hillas parameters. Skipping..."
                    )
                    continue

                timing_params = timing_parameters(
                    camera_geom_masked, image_masked, peak_time_masked, hillas_params
                )

                if np.isnan(timing_params.slope):
                    logger.info(
                        f"--> {event.count} event (event ID: {event.index.event_id}, "
                        f"telescope {tel_id}) failed to extract finite timing "
                        "parameters. Skipping..."
                    )
                    continue

                leakage_params = leakage_parameters(
                    camera_geoms[tel_id], image, signal_pixels
                )
                conc_params = concentration_parameters(
                    camera_geoms[tel_id], image, hillas_params
                )

                # Calculate additional parameters
                true_disp = calculate_disp(
                    pointing_alt=event.pointing.tel[tel_id].altitude,
                    pointing_az=event.pointing.tel[tel_id].azimuth,
                    shower_alt=event.simulation.shower.alt,
                    shower_az=event.simulation.shower.az,
                    cog_x=hillas_params.x,
                    cog_y=hillas_params.y,
                    camera_frame=camera_geoms[tel_id].frame,
                )

                true_impact = calculate_impact(
                    shower_alt=event.simulation.shower.alt,
                    shower_az=event.simulation.shower.az,
                    core_x=event.simulation.shower.core_x,
                    core_y=event.simulation.shower.core_y,
                    tel_pos_x=tel_positions[tel_id][0],
                    tel_pos_y=tel_positions[tel_id][1],
                    tel_pos_z=tel_positions[tel_id][2],
                )

                off_axis = angular_separation(
                    lon1=event.pointing.tel[tel_id].azimuth,
                    lat1=event.pointing.tel[tel_id].altitude,
                    lon2=event.simulation.shower.az,
                    lat2=event.simulation.shower.alt,
                )

                tels_with_trigger = np.intersect1d(
                    tels_with_trigger,
                    np.concatenate(
                        (LSTs_IDs[LSTs_IDs != 0], MAGICs_IDs[MAGICs_IDs != 0])
                    ),
                    assume_unique=True,
                ).tolist()

                # encode tels_with_trigger as an int value
                # that can be decoded later as a binary
                # tels_with_trigger = sum_{tel_id} 2**tel_id
                # where tel_id is only for those triggered
                tels_with_trigger_binary_int = np.array(
                    [2 ** (tel_id) for tel_id in tels_with_trigger]
                ).sum()

                # Set the event information
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
                    magic_stereo=magic_stereo,
                    tels_with_trigger=tels_with_trigger_binary_int,
                )

                # Reset the telescope IDs
                event_info.tel_id = tel_id

                # Save the parameters to the output file
                # Setting all the prefixes except of concentration to empty string
                event_info.prefix = ""
                hillas_params.prefix = ""
                timing_params.prefix = ""
                leakage_params.prefix = ""
                writer.write(
                    "parameters",
                    (
                        event_info,
                        hillas_params,
                        timing_params,
                        leakage_params,
                        conc_params,
                    ),
                )

        n_events_processed = event.count + 1
        logger.info(f"\nIn total {n_events_processed} events are processed.")

    # Save the subarray description
    subarray.to_hdf(output_file)

    # Save the simulation configuration
    with HDF5TableWriter(output_file, group_name="simulation", mode="a") as writer:
        writer.write("config", sim_config)

    logger.info(f"\nOutput file: {output_file}")


def main():
    """Main function."""

    start_time = time.time()

    parser = argparse.ArgumentParser()

    # Here we are simply collecting the parameters from the command line, as input file, output directory, and configuration file
    parser.add_argument(
        "--input-file",
        "-i",
        dest="input_file",
        type=str,
        required=True,
        help="Path to an input simtel MC DL0 data file",
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
        "--focal_length_choice",
        "-f",
        dest="focal_length_choice",
        type=str,
        choices=["equivalent", "effective"],
        default="effective",
        help='Choice of focal length, either "effective" or "equivalent". The default (and standard) value is "effective"',
    )

    args = parser.parse_args()  # Here we select all 3 parameters collected above

    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(
            f
        )  # Here we collect the inputs from the configuration file

    # Checking if the input telescope list is properly organized:
    check_input_list(config)

    config["mc_tel_ids"] = dict(
        sorted(config["mc_tel_ids"].items())
    )  # Sorting needed to correctly name the output file

    # Process the input data
    mc_dl0_to_dl1(args.input_file, args.output_dir, config, args.focal_length_choice)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
