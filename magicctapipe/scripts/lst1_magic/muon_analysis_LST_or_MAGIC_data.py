#!/usr/bin/env python
# coding: utf-8

"""
Author: Gabriel Emery

This script processes MAGIC calibrated data (*_Y_*.root) to perform the muon ring analysis.
Current naming allows LST but first implementation will only handle MAGIC as script for LST are available in lstchain.

Usage:
$ python muon_analysis_LST_or_MAGIC_data.py
--input-file ./data/calibrated/20201216_M1_05093711.001_Y_CrabNebula-W0.40+035.root
--output-dir ./data/muons
--config-file ./config.yaml
"""
import argparse
import yaml
import logging
import numpy as np
from pathlib import Path

from astropy.table import Table
from ctapipe.io import EventSource
from lstchain.image.muon import create_muon_table
from magicctapipe.image import MAGICClean
from magicctapipe.image.muons import perform_muon_analysis

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

pedestal_types = [
    "fundamental",
    "from_extractor",
    "from_extractor_rndm",
]


def magic_muons_from_cal(input_file, output_dir, config, process_run, plots_path):
    """
    Process all event a single telescope MAGIC calibrated data run or subrun to perform the muon ring analysis.

    Parameters
    ----------
    input_file:
    output_dir:
    config:
    process_run:
    plots_path:

    """

    event_source = EventSource(input_url=input_file)
    subarray = event_source.subarray
    obs_id = event_source.obs_ids[0]
    tel_id = event_source.telescope

    # Create the table which will contain the selected muon ring parameters
    muon_parameters = create_muon_table()
    muon_parameters["telescope_name"] = []

    logger.info(f"\nProcess the following data (process_run = {process_run}):")

    # Configure the MAGIC cleaning:
    config_cleaning = config["MAGIC"]["magic_clean"]

    if config_cleaning["find_hotpixels"] == "auto":
        config_cleaning.update({"find_hotpixels": True})

    logger.info("\nConfiguration for the image cleaning:")
    logger.info(config_cleaning)

    ped_type = config_cleaning.pop("pedestal_type")
    i_ped_type = np.where(np.array(pedestal_types) == ped_type)[0][0]

    camera_geom = event_source.subarray.tel[tel_id].camera.geometry
    magic_clean = MAGICClean(camera_geom, config_cleaning)

    # Prepare for saving muons data to an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if process_run:
        output_file = f"{output_dir}/muons_M{tel_id}.Run{obs_id:08}.fits"
    else:
        subrun_id = event_source.metadata["subrun_number"]
        output_file = (
            f"{output_dir}/muons_M{tel_id}.Run{obs_id:08}.{subrun_id[0]:03}.fits"
        )

    # Start processing events:
    logger.info("\nProcessing the events:")

    # load muon analysis config
    muon_config = {}
    if "muon_ring" in config["MAGIC"]:
        muon_config = config["MAGIC"]["muon_ring"]
    # Select the telescope name to be filed in muon_parameters['telescope_name']
    if tel_id == 1:
        tel_name = f"MAGIC-I"
    elif tel_id == 2:
        tel_name = f"MAGIC-II"
    else:
        tel_name = f"MAGIC_?"

    for event in event_source:

        # Apply the image cleaning:
        dead_pixels = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[0]
        try:
            badrms_pixels = event.mon.tel[tel_id].pixel_status.pedestal_failing_pixels[
                i_ped_type
            ]
        except TypeError:
            badrms_pixels = np.zeros(dead_pixels.shape, dtype=bool)
        unsuitable_mask = np.logical_or(dead_pixels, badrms_pixels)

        signal_pixels, image, peak_time = magic_clean.clean_image(
            event.dl1.tel[tel_id].image,
            event.dl1.tel[tel_id].peak_time,
            unsuitable_mask,
        )
        perform_muon_analysis(
            muon_parameters,
            event=event,
            telescope_id=tel_id,
            telescope_name=tel_name,
            image=image,
            subarray=subarray,
            r1_dl1_calibrator_for_muon_rings=None,
            good_ring_config=muon_config,
            data_type="obs",
            plot_rings=(plots_path is not None),
            plots_path=plots_path,
        )

    table = Table(muon_parameters)
    table.write(output_file, format="fits", overwrite=True)
    logger.info(f"\nOutput muons file: {output_file}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        "-i",
        dest="input_file",
        type=str,
        required=True,
        help="Path to an input MAGIC calibrated data file (*_Y_*.root).",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output muon file.",
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
        help="Processes all the sub-run files of the same observation ID at once.",
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
    magic_muons_from_cal(
        args.input_file, args.output_dir, config, args.process_run, args.plots_path
    )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
