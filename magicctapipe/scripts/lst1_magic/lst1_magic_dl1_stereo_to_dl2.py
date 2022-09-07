#!/usr/bin/env python
# coding: utf-8

"""
This script processes DL1-stereo events and reconstructs the DL2
parameters, i.e., energy, direction and gammaness, with trained RFs.
The RFs are applied per telescope combination type and per telescope.

Usage:
$ python lst1_magic_dl1_stereo_to_dl2.py
--input-file-dl1 dl1_stereo/dl1_stereo_LST-1_MAGIC.Run03265.0040.h5
--input-dir-rfs rfs
(--output-dir dl2)
"""

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
from magicctapipe.io import (
    TEL_COMBINATIONS,
    get_stereo_events,
    save_pandas_data_in_table,
)
from magicctapipe.reco import DispRegressor, EnergyRegressor, EventClassifier

__all__ = ["apply_rfs", "reconstruct_arrival_direction", "dl1_stereo_to_dl2"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def apply_rfs(event_data, estimator):
    """
    Applies trained RFs to DL1-stereo events, whose telescope
    combination type is same as the RFs.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Data frame of shower events
    estimator: magicctapipe.reco.estimator
        Trained regressor or classifier

    Returns
    -------
    reco_params: pandas.core.frame.DataFrame
        Data frame of the shower events with reconstructed parameters
    """

    tel_ids = list(estimator.telescope_rfs.keys())
    multiplicity = len(tel_ids)

    df_events = event_data.query(
        f"(tel_id == {tel_ids}) & (multiplicity == {multiplicity})"
    ).copy()

    df_events.dropna(subset=estimator.features, inplace=True)
    df_events["multiplicity"] = df_events.groupby(["obs_id", "event_id"]).size()
    df_events.query(f"multiplicity == {multiplicity}", inplace=True)

    if len(df_events) > 0:
        reco_params = estimator.predict(df_events)

    return reco_params


def reconstruct_arrival_direction(event_data, tel_descriptions):
    """
    Reconstructs the arrival directions of shower events with the
    MARS-like DISP method.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Data frame of shower events
    tel_descriptions: dict
        Telescope descriptions

    Returns
    -------
    reco_params: pandas.core.frame.DataFrame
        Data frame of the shower events with reconstructed directions
    """

    reco_params_flips = pd.DataFrame()

    # First of all, we reconstruct the Alt/Az directions of all the head
    # and tail candidates per telescope, i.e., the directions separated
    # by the DISP parameter from the image CoG along the shower axis.
    # Here the flip parameter (0 or 1) distinguishes the head and tail.

    tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

    for tel_id in tel_ids:

        df_events = event_data.query(f"tel_id == {tel_id}")

        tel_pointing = AltAz(
            alt=u.Quantity(df_events["pointing_alt"].to_numpy(), u.rad),
            az=u.Quantity(df_events["pointing_az"].to_numpy(), u.rad),
        )

        tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

        event_coord = SkyCoord(
            u.Quantity(df_events["x"].to_numpy(), u.m),
            u.Quantity(df_events["y"].to_numpy(), u.m),
            frame=tel_descriptions[tel_id].camera.geometry.frame,
        )

        event_coord = event_coord.transform_to(tel_frame)

        for flip in [0, 1]:

            psi_per_flip = df_events["psi"].to_numpy() + 180 * flip

            event_coord_per_flip = event_coord.directional_offset_by(
                position_angle=u.Quantity(psi_per_flip, u.deg),
                separation=u.Quantity(df_events["reco_disp"].to_numpy(), u.deg),
            )

            event_coord_per_flip = event_coord_per_flip.altaz

            df_altaz_per_flip = pd.DataFrame(
                data={
                    "reco_alt": event_coord_per_flip.alt.to_value(u.deg),
                    "reco_az": event_coord_per_flip.az.to_value(u.deg),
                    "flip": flip,
                },
                index=df_events.index,
            )

            reco_params_flips = reco_params_flips.append(df_altaz_per_flip)

    reco_params_flips.set_index("flip", append=True, inplace=True)
    reco_params_flips.sort_index(inplace=True)

    group_size = reco_params_flips.groupby(["obs_id", "event_id"]).size()
    reco_params_flips["multiplicity"] = group_size

    # Then, we get the flip combination minimizing the sum of the
    # angular distances between the head and tail candidates per shower
    # event. In order to speed up the calculations, here we separate the
    # input events by the telescope combination types.

    reco_params = pd.DataFrame()

    for tel_ids in TEL_COMBINATIONS.values():

        multiplicity = 2 * len(tel_ids)

        df_events = reco_params_flips.query(
            f"(multiplicity == {multiplicity}) & (tel_id == {tel_ids})"
        ).copy()

        df_events["multiplicity"] = df_events.groupby(["obs_id", "event_id"]).size()
        df_events.query(f"multiplicity == {multiplicity}", inplace=True)

        n_events = len(df_events.groupby(["obs_id", "event_id"]).size())

        # Here we first define all the possible flip combinations. For
        # example, in case that we have two telescope images, in total
        # 4 combinations can be defined as follows:
        #   [(head, head), (head, tail), (tail, head), (tail, tail)]
        # where the i-th element of each tuple means the i-th telescope
        # image. In case of 3 images we have in total 8 combinations.

        flip_combinations = np.array(
            list(itertools.product([0, 1], repeat=len(tel_ids)))
        )

        # Next, we define all the possible any 2 telescope combinations.
        # For example, in case of 3 telescopes, in total 3 combinations
        # are defined as follows:
        #                 [(1, 2), (1, 3), (2, 3)]
        # where the elements of the tuples mean the telescope IDs. In
        # case of 2 telescopes there is only one combination.

        tel_any2_combinations = list(itertools.combinations(tel_ids, 2))

        distances = np.zeros((len(flip_combinations), n_events))

        for i_flip, flip_combo in enumerate(flip_combinations):

            container = {}

            # Set the directions of a given flip combination
            for tel_id, flip in zip(tel_ids, flip_combo):
                container[tel_id] = df_events.query(
                    f"(tel_id == {tel_id}) & (flip == {flip})"
                )

            for tel_any2_combo in tel_any2_combinations:

                tel_id_1 = tel_any2_combo[0]
                tel_id_2 = tel_any2_combo[1]

                theta = angular_separation(
                    lon1=u.Quantity(container[tel_id_1]["reco_az"].to_numpy(), u.deg),
                    lat1=u.Quantity(container[tel_id_1]["reco_alt"].to_numpy(), u.deg),
                    lon2=u.Quantity(container[tel_id_2]["reco_az"].to_numpy(), u.deg),
                    lat2=u.Quantity(container[tel_id_2]["reco_alt"].to_numpy(), u.deg),
                )

                # Sum up the distance per flip combination
                distances[i_flip] += theta.to_value(u.deg)

        # Finally, we extract the indices of the flip combinations for
        # each event with which the angular distances become minimum
        distances = np.array(distances)
        distances_min = distances.min(axis=0)

        condition = distances == distances_min
        indices = np.where(condition.transpose())[1]

        flips = flip_combinations[indices].ravel()

        group_size = df_events.groupby(["obs_id", "event_id", "tel_id"]).size()

        obs_ids = group_size.index.get_level_values("obs_id")
        event_ids = group_size.index.get_level_values("event_id")
        tel_ids = group_size.index.get_level_values("tel_id")

        multi_indices = pd.MultiIndex.from_arrays(
            arrays=[obs_ids, event_ids, tel_ids, flips], names=df_events.index.names
        )

        df_events = df_events.loc[multi_indices]

        # Add the minimum angular distances to the output data frame,
        # since they are useful to separate gamma and hadron events
        # (hadron events tend to have larger distances than gammas)
        disp_diffs_sum = pd.Series(
            data=distances_min,
            index=df_events.groupby(["obs_id", "event_id"]).size().index,
            name="disp_diff_sum",
        )

        df_events = df_events.join(disp_diffs_sum)

        reco_params = reco_params.append(df_events)

    reco_params.reset_index(level="flip", inplace=True)
    reco_params.drop(["flip", "multiplicity"], axis=1, inplace=True)
    reco_params.sort_index(inplace=True)

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

    # Load the input DL1-stereo data file
    logger.info(f"\nInput DL1-stereo data file:\n{input_file_dl1}")

    event_data = pd.read_hdf(input_file_dl1, key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    is_simulation = "true_energy" in event_data.columns
    logger.info(f"\nIs simulation: {is_simulation}")

    event_data = get_stereo_events(event_data)

    subarray = SubarrayDescription.from_hdf(input_file_dl1)
    tel_descriptions = subarray.tel

    # Prepare for reconstructing the DL2 parameters
    logger.info(f"\nInput RF directory:\n{input_dir_rfs}")

    mask_energy_regressor = f"{input_dir_rfs}/energy_regressors_*.joblib"
    mask_disp_regressor = f"{input_dir_rfs}/disp_regressors_*.joblib"
    mask_event_classifier = f"{input_dir_rfs}/event_classifiers_*.joblib"

    # Reconstruct the energies
    input_files_energy = glob.glob(mask_energy_regressor)
    input_files_energy.sort()

    n_files_energy = len(input_files_energy)

    if n_files_energy > 0:

        logger.info(f"\nIn total {n_files_energy} energy regressor files are found:")
        energy_regressor = EnergyRegressor()

        for input_file in input_files_energy:

            logger.info(f"Applying {input_file}...")
            energy_regressor.load(input_file)

            reco_params = apply_rfs(event_data, energy_regressor)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del energy_regressor

    # Reconstruct the DISP parameter
    input_files_dips = glob.glob(mask_disp_regressor)
    input_files_dips.sort()

    n_files_disp = len(input_files_dips)

    if n_files_disp > 0:

        logger.info(f"\nIn total {n_files_disp} DISP regressor files are found:")
        disp_regressor = DispRegressor()

        for input_file in input_files_dips:

            logger.info(f"Applying {input_file}...")
            disp_regressor.load(input_file)

            reco_params = apply_rfs(event_data, disp_regressor)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

        # Reconstruct the arrival directions with the DISP method
        logger.info("\nReconstructing the arrival directions...")

        reco_params = reconstruct_arrival_direction(event_data, tel_descriptions)
        event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del disp_regressor

    # Reconstruct the gammaness
    input_files_class = glob.glob(mask_event_classifier)
    input_files_class.sort()

    n_files_class = len(input_files_class)

    if n_files_class > 0:

        logger.info(f"\nIn total {n_files_class} event classifier files are found:")
        event_classifier = EventClassifier()

        for input_file in input_files_class:

            logger.info(f"Applying {input_file}...")
            event_classifier.load(input_file)

            reco_params = apply_rfs(event_data, event_classifier)
            event_data.loc[reco_params.index, reco_params.columns] = reco_params

    del event_classifier

    # In case of the MAGIC-only analysis, here we drop the "time_sec"
    # and "time_nanosec" parameters but instead set the "timestamp"
    # parameter, since the precise timestamps are not needed anymore
    if "time_sec" in event_data.columns:

        time_sec = event_data["time_sec"].to_numpy() * u.s
        time_nanosec = event_data["time_nanosec"].to_numpy() * u.ns
        timestamps = time_sec + time_nanosec

        event_data["timestamp"] = timestamps.to_value(u.s)
        event_data.drop(columns=["time_sec", "time_nanosec"], inplace=True)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    input_file_name = Path(input_file_dl1).name

    output_file_name = input_file_name.replace("dl1_stereo", "dl2")
    output_file = f"{output_dir}/{output_file_name}"

    event_data.reset_index(inplace=True)

    save_pandas_data_in_table(
        event_data, output_file, group_name="/events", table_name="parameters"
    )

    subarray.to_hdf(output_file)

    if is_simulation:
        sim_config = pd.read_hdf(input_file_dl1, key="simulation/config")

        save_pandas_data_in_table(
            data=sim_config,
            output_file=output_file,
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

    # Process the input data
    dl1_stereo_to_dl2(args.input_file_dl1, args.input_dir_rfs, args.output_dir)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
