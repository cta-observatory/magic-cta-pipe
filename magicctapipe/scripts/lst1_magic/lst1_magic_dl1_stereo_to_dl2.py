#!/usr/bin/env python
# coding: utf-8

"""
This script processes DL1-stereo events and reconstructs the DL2
parameters, i.e., energy, direction and gammaness, with trained RFs.

For reconstructing the arrival directions, it uses the MARS-like DISP
method, i.e., select the closest combination with which the sum of the
angular distances of all the head and tail candidates becomes minimum.

Usage:
$ python lst1_magic_dl1_stereo_to_dl2.py
--input-file-dl1 dl1_stereo/dl1_stereo_LST-1_MAGIC.Run03265.0040.h5
--input-dir-rfs rfs
(--output-dir dl2)

Broader usage:
This script is called automatically from the script "DL1_to_DL2.py".
If you want to analyse a target, this is the way to go. See this other script for more details.

"""

import yaml
import argparse
import glob
import itertools
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation
from ctapipe.coordinates import TelescopeFrame
from ctapipe.instrument import SubarrayDescription
from magicctapipe.io import get_stereo_events, save_pandas_data_in_table, telescope_combinations
from magicctapipe.reco import DispRegressor, EnergyRegressor, EventClassifier

__all__ = ["apply_rfs", "reconstruct_arrival_direction", "dl1_stereo_to_dl2"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def apply_rfs(event_data, estimator, config):
    """
    Applies trained RFs to DL1-stereo events, whose telescope
    combination type is same as the RFs.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Data frame of shower events
    estimator: magicctapipe.reco.estimator
        Trained regressor or classifier
    config: dict
        evoked from an yaml file with information about the telescope IDs. Typically called "config_general.yaml"

    Returns
    -------
    reco_params: pandas.core.frame.DataFrame
        Data frame of the shower events with reconstructed parameters
    """
    
    tel_ids = list(estimator.telescope_rfs.keys())
    
    # Extract the events with the same telescope ID
    df_events = event_data.query(f"tel_id == {tel_ids[0]}")

    # Apply the RFs
    reco_params = estimator.predict(df_events)

    return reco_params


def reconstruct_arrival_direction(event_data, tel_descriptions, config):
    """
    Reconstructs the arrival directions of shower events with the
    MARS-like DISP method.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Data frame of shower events
    tel_descriptions: dict
        Telescope descriptions
    config: dict
        dictionary with telescope IDs information
    Returns
    -------
    reco_params: pandas.core.frame.DataFrame
        Data frame of the shower events with reconstructed directions
    """

    params_with_flips = pd.DataFrame()
    
    _, TEL_COMBINATIONS = telescope_combinations(config)
    
    # First of all, we reconstruct the directions of all the head and
    # tail candidates for every telescope image, i.e., the directions
    # separated by the DISP parameter from the image CoG along the
    # shower main axis. The `flip` parameter distinguishes them.

    tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

    for tel_id in tel_ids:
        df_events = event_data.query(f"tel_id == {tel_id}")

        tel_pointing = AltAz(
            alt=u.Quantity(df_events["pointing_alt"], unit="rad"),
            az=u.Quantity(df_events["pointing_az"], unit="rad"),
        )

        tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

        cog_coord = SkyCoord(
            u.Quantity(df_events["x"], unit="m"),
            u.Quantity(df_events["y"], unit="m"),
            frame=tel_descriptions[tel_id].camera.geometry.frame,
        )

        cog_coord = cog_coord.transform_to(tel_frame)

        for flip in [0, 1]:
            psi_flipped = df_events["psi"] + 180 * flip

            event_coord = cog_coord.directional_offset_by(
                position_angle=u.Quantity(psi_flipped, unit="deg"),
                separation=u.Quantity(df_events["reco_disp"], unit="deg"),
            )

            event_coord = event_coord.altaz

            df_altaz = pd.DataFrame(
                data={
                    "flip": flip,
                    "reco_alt": event_coord.alt.to_value("deg"),
                    "reco_az": event_coord.az.to_value("deg"),
                    "combo_type": df_events["combo_type"],
                },
                index=df_events.index,
            )

            params_with_flips = pd.concat([params_with_flips, df_altaz])

    params_with_flips.set_index("flip", append=True, inplace=True)
    params_with_flips.sort_index(inplace=True)

    # Then, we get the flip combination minimizing the angular distances
    # of the head and tail candidates for every shower event. In order
    # to speed up the calculations, here we process the events for every
    # telescope combination types.

    reco_params = pd.DataFrame()

    for combo_type, tel_ids in enumerate(TEL_COMBINATIONS.values()):
        df_events = params_with_flips.query(f"combo_type == {combo_type}")

        n_events = len(df_events.groupby(["obs_id", "event_id"]).size())

        # Here we first define all the possible flip combinations. For
        # example, in case that we have two telescope images, in total
        # 4 combinations are defined as follows:
        #   [(head, head), (head, tail), (tail, head), (tail, tail)]
        # where the i-th element of each tuple means the i-th telescope
        # image. In case of 3 images we have in total 8 combinations.

        flip_combinations = np.array(
            list(itertools.product([0, 1], repeat=len(tel_ids)))
        )

        # Next, we define all the possible 2 telescopes combinations.
        # For example, in case of 3 telescopes, in total 3 combinations
        # are defined as follows:
        #                 [(1, 2), (1, 3), (2, 3)]
        # where the elements of the tuples mean the telescope IDs. In
        # case of 2 telescopes there is only one combination.

        tel_any2_combinations = list(itertools.combinations(tel_ids, 2))

        distances = np.zeros((len(flip_combinations), n_events))

        # Loop over every flip combination
        for i_flip, flip_combo in enumerate(flip_combinations):
            container = {}

            # Set the directions of a given flip combination
            for tel_id, flip in zip(tel_ids, flip_combo):
                container[tel_id] = df_events.query(
                    f"(tel_id == {tel_id}) & (flip == {flip})"
                )

            for tel_id_1, tel_id_2 in tel_any2_combinations:
                # Calculate the distance of the 2-tel combination
                theta = angular_separation(
                    lon1=u.Quantity(container[tel_id_1]["reco_az"], unit="deg"),
                    lat1=u.Quantity(container[tel_id_1]["reco_alt"], unit="deg"),
                    lon2=u.Quantity(container[tel_id_2]["reco_az"], unit="deg"),
                    lat2=u.Quantity(container[tel_id_2]["reco_alt"], unit="deg"),
                )

                # Sum up the distance
                distances[i_flip] += theta.to_value("deg")

        # Extracts the minimum distances and their flip combinations
        distances_min = distances.min(axis=0)
        indices_at_min = distances.argmin(axis=0)

        flips = flip_combinations[indices_at_min].ravel()

        group_size = df_events.groupby(["obs_id", "event_id", "tel_id"]).size()

        obs_ids = group_size.index.get_level_values("obs_id")
        event_ids = group_size.index.get_level_values("event_id")
        tel_ids = group_size.index.get_level_values("tel_id")

        multi_indices = pd.MultiIndex.from_arrays(
            arrays=[obs_ids, event_ids, tel_ids, flips], names=df_events.index.names
        )

        # Keep only the information of the closest combinations
        df_events = df_events.loc[multi_indices]

        # Add the minimum angular distances to the output data frame,
        # since they are useful to separate gamma and hadron events
        # (hadron events tend to have larger distances than gammas)
        df_disp_diffs = pd.DataFrame(
            data={
                "disp_diff_sum": distances_min,
                "disp_diff_mean": distances_min / len(tel_any2_combinations),
            },
            index=df_events.groupby(["obs_id", "event_id"]).size().index,
        )

        df_events = df_events.join(df_disp_diffs)

        reco_params = pd.concat([reco_params, df_events])

    reco_params.reset_index(level="flip", inplace=True)
    reco_params.drop(["flip", "combo_type"], axis=1, inplace=True)
    reco_params.sort_index(inplace=True)

    return reco_params


def dl1_stereo_to_dl2(input_file_dl1, input_dir_rfs, output_dir, config):
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
    config: dict
        dictionary with telescope IDs information
    """
    
    TEL_NAMES, _ = telescope_combinations(config)
    
    # Load the input DL1-stereo data file
    logger.info(f"\nInput DL1-stereo data file: {input_file_dl1}")

    event_data = pd.read_hdf(input_file_dl1, key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    is_simulation = "true_energy" in event_data.columns
    logger.info(f"\nIs simulation: {is_simulation}")

    event_data = get_stereo_events(event_data, config)

    subarray = SubarrayDescription.from_hdf(input_file_dl1)
    tel_descriptions = subarray.tel

    logger.info(f"\nInput RF directory: {input_dir_rfs}")

    mask_energy_regressor = f"{input_dir_rfs}/energy_regressors_*.joblib"
    mask_disp_regressor = f"{input_dir_rfs}/disp_regressors_*.joblib"
    mask_event_classifier = f"{input_dir_rfs}/event_classifiers_*.joblib"

    # Find the energy regressors
    input_files_energy = glob.glob(mask_energy_regressor)
    input_files_energy.sort()

    n_files_energy = len(input_files_energy)

    if n_files_energy > 0:
        logger.info(f"\nIn total {n_files_energy} energy regressor files are found:")

        for input_file_energy in input_files_energy:
            logger.info(f"Applying {input_file_energy}...")

            energy_regressor = EnergyRegressor(TEL_NAMES)
            energy_regressor.load(input_file_energy)

            # Apply the RFs
            reco_params = apply_rfs(event_data, energy_regressor, config)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del energy_regressor

    # Find the DISP regressors
    input_files_dips = glob.glob(mask_disp_regressor)
    input_files_dips.sort()

    n_files_disp = len(input_files_dips)

    if n_files_disp > 0:
        logger.info(f"\nIn total {n_files_disp} DISP regressor files are found:")

        for input_file_disp in input_files_dips:
            logger.info(f"Applying {input_file_disp}...")

            disp_regressor = DispRegressor(TEL_NAMES)
            disp_regressor.load(input_file_disp)

            # Apply the RFs
            reco_params = apply_rfs(event_data, disp_regressor, config)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

        # Reconstruct the arrival directions with the DISP method
        logger.info("\nReconstructing the arrival directions...")

        reco_params = reconstruct_arrival_direction(event_data, tel_descriptions, config)
        event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del disp_regressor

    # Find the event classifiers
    input_files_class = glob.glob(mask_event_classifier)
    input_files_class.sort()

    n_files_class = len(input_files_class)

    if n_files_class > 0:
        logger.info(f"\nIn total {n_files_class} event classifier files are found:")

        for input_file_class in input_files_class:
            logger.info(f"Applying {input_file_class}...")

            event_classifier = EventClassifier(TEL_NAMES)
            event_classifier.load(input_file_class)

            # Apply the RFs
            reco_params = apply_rfs(event_data, event_classifier, config)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del event_classifier

    # In case of MAGIC-only analyses, here we drop `time_sec` and
    # `time_nanosec` but instead set `timestamp`, since the precise
    # timestamps are not needed anymore
    if "time_sec" in event_data.columns:
        time_sec = u.Quantity(event_data["time_sec"], unit="s")
        time_nanosec = u.Quantity(event_data["time_nanosec"], unit="ns")
        timestamps = time_sec + time_nanosec

        event_data["timestamp"] = timestamps.to_value("s")
        event_data.drop(columns=["time_sec", "time_nanosec"], inplace=True)

    event_data.reset_index(inplace=True)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    input_file_name = Path(input_file_dl1).name

    output_file_name = input_file_name.replace("dl1_stereo", "dl2")
    output_file = f"{output_dir}/{output_file_name}"

    save_pandas_data_in_table(
        event_data, output_file, group_name="/events", table_name="parameters"
    )

    # Save the subarray description
    subarray.to_hdf(output_file)

    if is_simulation:
        # Save the simulation configuration
        sim_config = pd.read_hdf(input_file_dl1, key="simulation/config")

        save_pandas_data_in_table(
            input_data=sim_config,
            output_file=output_file,
            group_name="/simulation",
            table_name="config",
            mode="a",
        )

    logger.info(f"\nOutput file: {output_file}")


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file-dl1",
        "-d",
        dest="input_file_dl1",
        type=str,
        required=True,
        help="Path to an input DL1-stereo data file",
    )

    parser.add_argument(
        "--input-dir-rfs",
        "-r",
        dest="input_dir_rfs",
        type=str,
        required=True,
        help="Path to a directory where trained RFs are stored",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL2 data file",
    )
    
    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_general.yaml",
        help="Path to a configuration file",
    )

    args = parser.parse_args()
    
    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)
        
    # Process the input data
    dl1_stereo_to_dl2(args.input_file_dl1, args.input_dir_rfs, args.output_dir, config)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
