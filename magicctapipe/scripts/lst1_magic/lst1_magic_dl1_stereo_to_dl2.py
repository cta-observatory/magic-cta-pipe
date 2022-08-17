#!/usr/bin/env python
# coding: utf-8

"""
This script processes DL1-stereo events and reconstructs the DL2
parameters, i.e., energy, direction and gammaness, with trained RFs.
The RFs are currently applied per telescope combination and per
telescope type.

If the input is real data, the telescope pointing and reconstructed
event arrival directions will be transformed from the Alt/Az to the
RA/Dec coordinate using timestamps.

Usage:
$ python lst1_magic_dl1_stereo_to_dl2.py
--input-file-dl1 ./dl1_stereo/dl1_stereo_LST-1_MAGIC.Run03265.0040.h5
--input-dir-rfs ./rfs
--output-dir ./dl2
"""

import argparse
import glob
import logging
import time
from pathlib import Path

import pandas as pd
from astropy import units as u
from astropy.time import Time
from ctapipe.instrument import SubarrayDescription
from magicctapipe.reco import DirectionRegressor, EnergyRegressor, EventClassifier
from magicctapipe.utils import (
    get_stereo_events,
    save_pandas_to_table,
    transform_altaz_to_radec,
)

__all__ = ["apply_rfs", "dl1_stereo_to_dl2"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

NSEC2SEC = 1e-9


def apply_rfs(event_data, estimator):
    """
    Applies trained RFs to input DL1-stereo data.

    It selects only the events whose telescope combination type is same
    as the input RFs.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events
    estimator: magicctapipe.reco.estimator
        Trained regressor or classifier

    Returns
    -------
    reco_params: pandas.core.frame.DataFrame
        Pandas data frame of the DL2 parameters
    """

    tel_ids = list(estimator.telescope_rfs.keys())

    df_events = event_data.query(
        f"(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})"
    ).copy()

    df_events["multiplicity"] = df_events.groupby(["obs_id", "event_id"]).size()
    df_events.query(f"multiplicity == {len(tel_ids)}", inplace=True)

    reco_params = estimator.predict(df_events)

    return reco_params


def dl1_stereo_to_dl2(input_file_dl1, input_dir_rfs, output_dir):
    """
    Processes DL1-stereo events and reconstructs the DL2 parameters with
    trained RFs.

    Parameters
    ----------
    input_file_dl1: str
        Path to an input DL1-stereo data file
    input_dir_rfs: str
        Path to a directory where trained RFs are stored
    output_dir: str
        Path to a directory where to save an output DL2 data file
    """

    # Load the input DL1-stereo data file:
    logger.info(f"\nInput DL1-stereo data file:\n{input_file_dl1}")

    event_data = pd.read_hdf(input_file_dl1, key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    is_simulation = "true_energy" in event_data.columns
    logger.info(f"\nIs simulation: {is_simulation}")

    event_data = get_stereo_events(event_data)

    # Start reconstructing the DL2 parameters:
    logger.info(f"\nInput RF directory:\n{input_dir_rfs}")

    mask_energy_regressor = f"{input_dir_rfs}/energy_regressors_*.joblib"
    mask_direction_regressor = f"{input_dir_rfs}/direction_regressors_*.joblib"
    mask_event_classifier = f"{input_dir_rfs}/event_classifiers_*.joblib"

    # Reconstruct the energies:
    input_rfs_energy = glob.glob(mask_energy_regressor)
    input_rfs_energy.sort()

    if len(input_rfs_energy) > 0:

        logger.info("\nReconstructing the energies...")
        energy_regressor = EnergyRegressor()

        for input_rfs in input_rfs_energy:

            logger.info(input_rfs)
            energy_regressor.load(input_rfs)

            reco_params = apply_rfs(event_data, energy_regressor)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del energy_regressor

    # Reconstruct the arrival directions:
    input_rfs_direction = glob.glob(mask_direction_regressor)
    input_rfs_direction.sort()

    if len(input_rfs_direction) > 0:

        logger.info("\nReconstructing the arrival directions...")
        direction_regressor = DirectionRegressor()

        for input_rfs in input_rfs_direction:

            logger.info(input_rfs)
            direction_regressor.load(input_rfs)

            reco_params = apply_rfs(event_data, direction_regressor)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del direction_regressor

    # Reconstruct the gammaness:
    input_rfs_class = glob.glob(mask_event_classifier)
    input_rfs_class.sort()

    if len(input_rfs_class) > 0:

        logger.info("\nReconstructing the gammaness...")
        event_classifier = EventClassifier()

        for input_rfs in input_rfs_class:

            logger.info(input_rfs)
            event_classifier.load(input_rfs)

            reco_params = apply_rfs(event_data, event_classifier)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del event_classifier

    # Compute the RA/Dec directions:
    if not is_simulation:

        logger.info("\nTransforming pointing Alt/Az to the RA/Dec coordinate...")

        if "timestamp" in event_data.columns:
            timestamps = Time(
                event_data["timestamp"].to_numpy(), format="unix", scale="utc"
            )

        else:
            # Handle the case when the input is MAGIC-only real data:
            time_sec = event_data["time_sec"].to_numpy()
            time_nanosec = event_data["time_nanosec"].to_numpy() * NSEC2SEC

            timestamps = Time(time_sec + time_nanosec, format="unix", scale="utc")

            event_data["timestamp"] = timestamps.value
            event_data.drop(columns=["time_sec", "time_nanosec"], inplace=True)

        pointing_ra, pointing_dec = transform_altaz_to_radec(
            alt=u.Quantity(event_data["pointing_alt"].values, u.rad),
            az=u.Quantity(event_data["pointing_az"].values, u.rad),
            obs_time=timestamps,
        )

        event_data["pointing_ra"] = pointing_ra.to_value(u.deg)
        event_data["pointing_dec"] = pointing_dec.to_value(u.deg)

        if "reco_alt" in event_data.columns:

            logger.info("Transforming reconstructed Alt/Az to the RA/Dec coordinate...")

            reco_ra, reco_dec = transform_altaz_to_radec(
                alt=u.Quantity(event_data["reco_alt"].values, u.deg),
                az=u.Quantity(event_data["reco_az"].values, u.deg),
                obs_time=timestamps,
            )

            event_data["reco_ra"] = reco_ra.to_value(u.deg)
            event_data["reco_dec"] = reco_dec.to_value(u.deg)

    # Save the data in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    input_file_name = Path(input_file_dl1).name

    output_file_name = input_file_name.replace("dl1_stereo", "dl2")
    output_file = f"{output_dir}/{output_file_name}"

    event_data.reset_index(inplace=True)

    save_pandas_to_table(
        event_data, output_file, group_name="/events", table_name="parameters"
    )

    subarray = SubarrayDescription.from_hdf(input_file_dl1)
    subarray.to_hdf(output_file)

    if is_simulation:
        sim_config = pd.read_hdf(input_file_dl1, key="simulation/config")

        save_pandas_to_table(
            sim_config,
            output_file,
            group_name="/simulation",
            table_name="config",
            mode="a",
        )

    logger.info(f"\nOutput file:\n{output_file}")


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file-dl1",
        "-d",
        dest="input_file_dl1",
        type=str,
        required=True,
        help="Path to an input DL1-stereo data file.",
    )

    parser.add_argument(
        "--input-dir-rfs",
        "-r",
        dest="input_dir_rfs",
        type=str,
        required=True,
        help="Path to a directory where trained RFs are stored.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL2 data file.",
    )

    args = parser.parse_args()

    # Process the input data:
    dl1_stereo_to_dl2(args.input_file_dl1, args.input_dir_rfs, args.output_dir)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
