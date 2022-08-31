#!/usr/bin/env python
# coding: utf-8

"""
This script processes DL1 events and reconstructs the stereo parameters with more
than one telescope information. The quality cuts specified in a configuration file
are applied to events before the reconstruction.

When an input is real data containing LST-1 and MAGIC events, it checks the angular
distance of their pointing directions. Then, it stops the process when the distance
is larger than the limit specified in a configuration file. This is in principle to
avoid the reconstruction of the data taken in too-mispointing conditions, for example,
DL1 data may contain the coincident events which are taken with different wobble
offsets between the systems.

If the "--magic-only" option is given, it reconstructs the stereo parameters using
only MAGIC images.

Usage:
$ python lst1_magic_stereo_reco.py
--input-file ./data/dl1_coincidence/dl1_LST-1_MAGIC.Run03265.0040.h5
--output-dir ./data/dl1_stereo
--config-file ./config.yaml
(--magic-only)
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from astropy.coordinates import Angle, angular_separation
from ctapipe.containers import (
    ArrayEventContainer,
    CameraHillasParametersContainer,
    ImageParametersContainer,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.reco import HillasReconstructor
from magicctapipe.utils import (
    calculate_impact,
    calculate_mean_direction,
    check_tel_combination,
    save_pandas_to_table,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [
    "calculate_pointing_separation",
    "stereo_reconstruction",
]


def calculate_pointing_separation(event_data):
    """
    Calculates the angular distance of the
    LST-1 and MAGIC pointing directions.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of LST-1 and MAGIC events

    Returns
    -------
    theta: astropy.units.quantity.Quantity
        Angular distance of the LST-1 and MAGIC pointing directions
    """

    df_lst = event_data.query("tel_id == 1")

    pointing_az_lst = u.Quantity(df_lst["pointing_az"].to_numpy(), u.rad)
    pointing_alt_lst = u.Quantity(df_lst["pointing_alt"].to_numpy(), u.rad)

    obs_ids = df_lst.index.get_level_values("obs_id").tolist()
    event_ids = df_lst.index.get_level_values("event_id").tolist()

    multi_indices = pd.MultiIndex.from_arrays(
        [obs_ids, event_ids],
        names=["obs_id", "event_id"],
    )

    df_magic = event_data.query("tel_id == [2, 3]")
    df_magic.reset_index(level="tel_id", inplace=True)
    df_magic = df_magic.loc[multi_indices]

    # Calculate the mean of the M1 and M2 pointing directions:
    pointing_az_magic, pointing_alt_magic = calculate_mean_direction(
        lon=df_magic["pointing_az"],
        lat=df_magic["pointing_alt"],
    )

    theta = angular_separation(
        lon1=pointing_az_lst,
        lat1=pointing_alt_lst,
        lon2=pointing_az_magic,
        lat2=pointing_alt_magic,
    )

    return theta


def stereo_reconstruction(
    input_file,
    output_dir,
    config,
    magic_only_analysis=False,
):
    """
    Processes DL1 events and reconstructs the stereo parameters
    with more than one telescope information.

    Parameters
    ----------
    input_file: str
        Path to an input DL1 data file
    output_dir: str
        Path to a directory where to save an output DL1-stereo data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    magic_only_analysis: bool
        If True, it reconstructs the parameters using only MAGIC images
    """

    logger.info(f"Input file:\n{input_file}")
    logger.info(f"\nMAGIC-only analysis: {magic_only_analysis}")

    # Load the input file:
    event_data = pd.read_hdf(input_file, key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    is_simulation = "true_energy" in event_data.columns

    subarray = SubarrayDescription.from_hdf(input_file)
    tel_positions = subarray.positions

    logger.info("\nSubarray configuration:")
    for tel_id in subarray.tel.keys():
        logger.info(
            f"\tTelescope {tel_id}: {subarray.tel[tel_id].name}, "
            f"position = {tel_positions[tel_id]}"
        )

    # Apply the event cuts:
    if magic_only_analysis:
        event_data.query("tel_id > 1", inplace=True)

    config_stereo = config["stereo_reco"]

    logger.info("\nApplying the quality cuts:")
    logger.info(config_stereo["quality_cuts"])

    event_data.query(config_stereo["quality_cuts"], inplace=True)
    event_data["multiplicity"] = event_data.groupby(["obs_id", "event_id"]).size()
    event_data.query("multiplicity > 1", inplace=True)

    combo_types = check_tel_combination(event_data)
    event_data[combo_types.columns[0]] = combo_types

    # Check the angular distance of the LST1 and MAGIC pointing directions:
    if (not is_simulation) and (not magic_only_analysis):

        logger.info(
            "\nChecking the angular distance of the "
            "LST-1 and MAGIC pointing directions..."
        )

        theta = calculate_pointing_separation(event_data)

        if len(theta) == 0:
            logger.info("--> No LST-1 + MAGIC events are found.")

        else:
            theta_max = np.max(theta.to(u.arcmin))
            theta_uplim = u.Quantity(config_stereo["theta_uplim"], u.arcmin)

            if theta_max > theta_uplim:
                logger.info(
                    f"--> The maximum angular distance {theta_max:.3f} is "
                    f"larger than the limit {theta_uplim}. Exiting."
                )
                sys.exit()
            else:
                logger.info(
                    f"--> The maximum angular distance {theta_max:.3f} is "
                    f"smaller than the limit {theta_uplim}."
                )

    # Configure the HillasReconstructor:
    hillas_reconstructor = HillasReconstructor(subarray)

    # Start processing the events:
    logger.info("\nReconstructing the stereo parameters...")

    pointing_az_mean, pointing_alt_mean = calculate_mean_direction(
        lon=event_data["pointing_az"],
        lat=event_data["pointing_alt"],
    )

    group = event_data.groupby(["obs_id", "event_id"]).size()
    obs_ids = group.index.get_level_values("obs_id")
    event_ids = group.index.get_level_values("event_id")

    for i_evt, (obs_id, event_id) in enumerate(zip(obs_ids, event_ids)):

        event = ArrayEventContainer()

        if i_evt % 100 == 0:
            logger.info(f"{i_evt} events")

        event.pointing.array_altitude = pointing_alt_mean[i_evt]
        event.pointing.array_azimuth = pointing_az_mean[i_evt]

        df_evt = event_data.query(f"(obs_id == {obs_id}) & (event_id == {event_id})")

        tel_ids = df_evt.index.get_level_values("tel_id")

        for tel_id in tel_ids:

            df_tel = df_evt.query(f"tel_id == {tel_id}")

            event.pointing.tel[tel_id].altitude = df_tel["pointing_alt"].iloc[0] * u.rad
            event.pointing.tel[tel_id].azimuth = df_tel["pointing_az"].iloc[0] * u.rad

            hillas_params = CameraHillasParametersContainer(
                intensity=float(df_tel["intensity"].iloc[0]),
                x=u.Quantity(df_tel["x"].iloc[0], u.m),
                y=u.Quantity(df_tel["y"].iloc[0], u.m),
                r=u.Quantity(df_tel["r"].iloc[0], u.m),
                phi=Angle(df_tel["phi"].iloc[0], u.deg),
                length=u.Quantity(df_tel["length"].iloc[0], u.m),
                width=u.Quantity(df_tel["width"].iloc[0], u.m),
                psi=Angle(df_tel["psi"].iloc[0], u.deg),
                skewness=float(df_tel["skewness"].iloc[0]),
                kurtosis=float(df_tel["kurtosis"].iloc[0]),
            )

            image_params = ImageParametersContainer(hillas=hillas_params)
            event.dl1.tel[tel_id].parameters = image_params

        # Reconstruct the stereo parameters:
        hillas_reconstructor(event)

        stereo_params = event.dl2.stereo.geometry["HillasReconstructor"]

        if stereo_params.az < 0:
            stereo_params.az += u.Quantity(360, u.deg)

        for tel_id in tel_ids:

            # Calculate the impact parameter:
            impact = calculate_impact(
                core_x=stereo_params.core_x,
                core_y=stereo_params.core_y,
                az=stereo_params.az,
                alt=stereo_params.alt,
                tel_pos_x=tel_positions[tel_id][0],
                tel_pos_y=tel_positions[tel_id][1],
                tel_pos_z=tel_positions[tel_id][2],
            )

            # Set the stereo parameters to the data frame:
            params = {
                "alt": stereo_params.alt.to(u.deg).value,
                "alt_uncert": stereo_params.alt_uncert.to(u.deg).value,
                "az": stereo_params.az.to(u.deg).value,
                "az_uncert": stereo_params.az_uncert.to(u.deg).value,
                "core_x": stereo_params.core_x.to(u.m).value,
                "core_y": stereo_params.core_y.to(u.m).value,
                "core_uncert": stereo_params.core_uncert.to(u.m).value,
                "impact": impact.to(u.m).value,
                "h_max": stereo_params.h_max.to(u.m).value,
                "h_max_uncert": stereo_params.h_max_uncert.to(u.m).value,
                "is_valid": int(stereo_params.is_valid),
            }

            index = (obs_id, event_id, tel_id)
            event_data.loc[index, params.keys()] = params.values()

    n_events_processed = i_evt + 1
    logger.info(f"{n_events_processed} events")

    # Save the data in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    regex = r"dl1_(\S+)\.h5"
    file_name = Path(input_file).name

    if re.fullmatch(regex, file_name):
        parser = re.findall(regex, file_name)[0]

        if magic_only_analysis:
            output_file = f"{output_dir}/dl1_stereo_magic_only_{parser}.h5"
        else:
            output_file = f"{output_dir}/dl1_stereo_{parser}.h5"

    else:
        raise RuntimeError("Could not parse information from the input file name.")

    event_data.reset_index(inplace=True)

    save_pandas_to_table(
        event_data,
        output_file,
        group_name="/events",
        table_name="parameters",
        mode="w",
    )

    if is_simulation:
        sim_config = pd.read_hdf(
            input_file,
            key="simulation/config",
        )

        save_pandas_to_table(
            sim_config,
            output_file,
            group_name="/simulation",
            table_name="config",
            mode="a",
        )

    subarray.to_hdf(output_file)

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
        help="Path to an input DL1 data file.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL1-stereo data file.",
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
        "--magic-only",
        dest="magic_only",
        action="store_true",
        help="Reconstruct the stereo parameters using only MAGIC images.",
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Process the input data:
    stereo_reconstruction(args.input_file, args.output_dir, config, args.magic_only)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
