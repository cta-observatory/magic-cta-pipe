#!/usr/bin/env python
# coding: utf-8

"""
This script processes LST-1 and MAGIC events of simtel MC DL0 data (*.simtel.gz).

Usage:
$ python lst1_magic_mc_dl0_to_dl1.py
--input-file ./data/gamma_off0.4deg/dl0/gamma_40deg_90deg_run1___cta-prod5-lapalma_LST-1_MAGIC_desert-2158m_mono_off0.4.simtel.gz
--config-file ./config.yaml
"""

import re
import time
import yaml
import logging
import argparse
import numpy as np
from pathlib import Path
from astropy.table import Table
from traitlets.config import Config
from ctapipe.io import EventSource
from ctapipe.calib import CameraCalibrator
from ctapipe.image import (
    ImageExtractor,
    tailcuts_clean,
    apply_time_delta_cleaning,
    number_of_islands,
)
from lstchain.image.cleaning import apply_dynamic_cleaning
from lstchain.image.muon import create_muon_table
from magicctapipe.image import MAGICClean
from magicctapipe.image.muons import perform_muon_analysis

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [
    "mc_dl0_to_muons",
]


def mc_dl0_to_muons(input_file, output_dir, config, plots_path):
    """
    Processes LST-1 and MAGIC events of simtel MC DL0 data
    and computes the DL1 parameters.

    Parameters
    ----------
    input_file: str
        Path to an input simtel MC DL0 data file
    output_dir: str
        Path to a directory where to save an output file
    config: dict
        Configuration for the analysis
    plots_path: string
        Path to output plots. No plots are created if None is provided
    """

    # Load the input file:
    logger.info("\nLoading the input file:")
    logger.info(input_file)

    event_source = EventSource(
        input_file,
    )

    subarray = event_source.subarray

    camera_geoms = {}
    tel_positions = subarray.positions

    logger.info("\nSubarray configuration:")
    for tel_id in subarray.tel:
        logger.info(
            f"Telescope {tel_id}: {subarray.tel[tel_id]}, position = {tel_positions[tel_id]}"
        )
        camera_geoms[tel_id] = subarray.tel[tel_id].camera.geometry

    # Configure the LST event processors:
    config_lst = config["LST"]

    logger.info("\nLST image extractor:")
    logger.info(config_lst["image_extractor"])

    extractor_type_lst = config_lst["image_extractor"].pop("type")
    config_extractor_lst = Config({extractor_type_lst: config_lst["image_extractor"]})

    calibrator_lst = CameraCalibrator(
        image_extractor_type=extractor_type_lst,
        config=config_extractor_lst,
        subarray=subarray,
    )

    logger.info("\nLST image cleaning:")
    for cleaning in ["tailcuts_clean", "time_delta_cleaning", "dynamic_cleaning"]:
        logger.info(f"{cleaning}: {config_lst[cleaning]}")

    use_time_delta_cleaning = config_lst["time_delta_cleaning"].pop("use")
    use_dynamic_cleaning = config_lst["dynamic_cleaning"].pop("use")

    use_only_main_island = config_lst["use_only_main_island"]
    logger.info(f"use_only_main_island: {use_only_main_island}")

    # Configure the MAGIC event processors:
    config_magic = config["MAGIC"]

    logger.info("\nMAGIC image extractor:")
    logger.info(config_magic["image_extractor"])

    extractor_type_magic = config_magic["image_extractor"].pop("type")
    config_extractor_magic = Config(
        {extractor_type_magic: config_magic["image_extractor"]}
    )

    logger.info("\nMAGIC charge correction:")
    logger.info(config_magic["charge_correction"])

    use_charge_correction = config_magic["charge_correction"].pop("use")

    logger.info("\nMAGIC image cleaning:")
    if config_magic["magic_clean"]["find_hotpixels"] is not False:
        logger.warning(
            'Hot pixels do not exist in a simulation. Setting the "find_hotpixels" option to False...'
        )
        config_magic["magic_clean"].update({"find_hotpixels": False})

    logger.info(config_magic["magic_clean"])
    i = None
    for j in subarray.tel:
        if i is None or (
            "MAGIC" in subarray.tel[j].name
            or "UNKNOWN-232M2" == subarray.tel[j].name
            or "UNKNOWN-235M2" == subarray.tel[j].name
        ):
            i = j
    magic_clean = MAGICClean(camera_geoms[i], config_magic["magic_clean"])

    # Configure the muon analysis:
    muon_parameters = create_muon_table()
    muon_parameters["telescope_name"] = []
    r1_dl1_calibrator_for_muon_rings = {}

    extractor_muon_name_lst = "GlobalPeakWindowSum"
    extractor_lst_muons = ImageExtractor.from_name(
        extractor_muon_name_lst, subarray=subarray, config=config_extractor_lst
    )
    r1_dl1_calibrator_for_muon_rings["LST"] = CameraCalibrator(
        subarray, image_extractor=extractor_lst_muons
    )
    # Use the standard MAGIC calibration and charextraction to be comparable with MAGIC data
    extractor_magic_muons = ImageExtractor.from_name(
        extractor_type_magic, config=config_extractor_magic, subarray=subarray
    )

    r1_dl1_calibrator_for_muon_rings_magic = CameraCalibrator(
        subarray, image_extractor=extractor_magic_muons
    )
    r1_dl1_calibrator_for_muon_rings["MAGIC"] = r1_dl1_calibrator_for_muon_rings_magic
    muon_config = {"LST": {}, "MAGIC": {}}
    if "muon_ring" in config_lst:
        muon_config["LST"] = config_lst["muon_ring"]
    if "muon_ring" in config_magic:
        muon_config["MAGIC"] = config_magic["muon_ring"]
    # Prepare for saving data to an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    regex_off = r"(\S+)_run(\d+)_.*_off(\S+)\.simtel.gz"
    regex = r"(\S+)_run(\d+)[_\.].*simtel.gz"

    file_name = Path(input_file).resolve().name

    if re.fullmatch(regex_off, file_name):
        parser = re.findall(regex_off, file_name)[0]
        output_file = f"{output_dir}/muons_{parser[0]}_off{parser[2]}deg_LST-1_MAGIC_run{parser[1]}.fits"

    elif re.fullmatch(regex, file_name):
        parser = re.findall(regex, file_name)[0]
        output_file = f"{output_dir}/muons_{parser[0]}_LST-1_MAGIC_run{parser[1]}.fits"

    else:
        output_file = f'{output_dir}/muons_{file_name.replace(".simtel","").replace(".gz","")}.fits'

    # Start processing the events:
    logger.info("\nProcessing the events...")

    for event in event_source:
        if event.count % 100 == 0:
            logger.info(f"{event.count} events")

        tels_with_trigger = event.trigger.tels_with_trigger

        for tel_id in tels_with_trigger:
            if "LST" == subarray.tel[tel_id].name:
                name = "LST"
                # Calibrate the event:
                calibrator_lst._calibrate_dl0(event, tel_id)
                calibrator_lst._calibrate_dl1(event, tel_id)

                image = event.dl1.tel[tel_id].image
                peak_time = event.dl1.tel[tel_id].peak_time

                # Apply the image cleaning:
                signal_pixels = tailcuts_clean(
                    camera_geoms[tel_id], image, **config_lst["tailcuts_clean"]
                )

                if use_time_delta_cleaning:
                    signal_pixels = apply_time_delta_cleaning(
                        camera_geoms[tel_id],
                        signal_pixels,
                        peak_time,
                        **config_lst["time_delta_cleaning"],
                    )

                if use_dynamic_cleaning:
                    signal_pixels = apply_dynamic_cleaning(
                        image, signal_pixels, **config_lst["dynamic_cleaning"]
                    )

                if use_only_main_island:
                    _, island_labels = number_of_islands(
                        camera_geoms[tel_id], signal_pixels
                    )
                    n_pixels_on_island = np.bincount(island_labels.astype(np.int64))
                    n_pixels_on_island[
                        0
                    ] = 0  # first island is no-island and should not be considered
                    max_island_label = np.argmax(n_pixels_on_island)
                    signal_pixels[island_labels != max_island_label] = False

            elif (
                "MAGIC" in subarray.tel[tel_id].name
                or "UNKNOWN-232M2" == subarray.tel[tel_id].name
                or "UNKNOWN-235M2" == subarray.tel[tel_id].name
            ):
                name = "MAGIC"
                # Calibrate the event:
                r1_dl1_calibrator_for_muon_rings_magic._calibrate_dl0(event, tel_id)
                r1_dl1_calibrator_for_muon_rings_magic._calibrate_dl1(event, tel_id)

                if use_charge_correction:
                    # Scale the charges of the DL1 image by the correction factor:
                    event.dl1.tel[tel_id].image *= config_magic["charge_correction"][
                        "factor"
                    ]

                # Apply the image cleaning:
                signal_pixels, image, peak_time = magic_clean.clean_image(
                    event.dl1.tel[tel_id].image, event.dl1.tel[tel_id].peak_time
                )
            else:
                logger.exception(
                    "Telescope name not recognised "
                    + subarray.tel[tel_id].name
                    + "\nConsider modifying the script to handle it if this is expected"
                )

            image_cleaned = image.copy()
            image_cleaned[~signal_pixels] = 0

            peak_time_cleaned = peak_time.copy()
            peak_time_cleaned[~signal_pixels] = 0

            n_pixels = np.count_nonzero(signal_pixels)

            if n_pixels == 0:
                continue

            perform_muon_analysis(
                muon_parameters,
                event=event,
                telescope_id=tel_id,
                telescope_name=subarray.tel[tel_id].name,
                image=image,
                subarray=subarray,
                r1_dl1_calibrator_for_muon_rings=r1_dl1_calibrator_for_muon_rings[name],
                good_ring_config=muon_config[name],
                data_type="mc",
                plot_rings=(plots_path is not None),
                plots_path=plots_path,
            )

    table = Table(muon_parameters)
    table.write(output_file, format="fits", overwrite=True)
    logger.info(f"\nOutput muons file: {output_file}")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        "-i",
        dest="input_file",
        type=str,
        required=True,
        help="Path to an input simtel MC DL0 data file.",
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
        "--plots_path",
        "-pp",
        dest="plots_path",
        type=str,
        default=None,
        help="If provided, muon rings will be plotted at this destination",
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Process the input data:
    mc_dl0_to_muons(args.input_file, args.output_dir, config, args.plots_path)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
