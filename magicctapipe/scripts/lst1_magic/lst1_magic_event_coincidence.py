#!/usr/bin/env python
# coding: utf-8

"""
This script searches for coincident events from LST-1 and MAGIC joint
observation data offline using their timestamps. It applies time offsets
and the coincidence window to LST-1 events, and checks the event
coincidence within the offset region specified in the configuration
file. Since the optimal time offset changes depending on the telescope
distance along the pointing direction, it requires to input a subrun
file for LST data, whose observation time is usually around 10 seconds
so the change of the distance is negligible. Only the parameters common
to both LST-1 and MAGIC data will be saved to an output file.

The MAGIC standard stereo analysis discards the events when one of the
telescope images cannot survive the cleaning or fail to reconstruct the
DL1 parameters. However, it's possible to perform the stereo analysis if
LST-1 sees these events. Thus, it checks the event coincidence for each
telescope combination (i.e., LST-1 + M1 and LST-1 + M2) and keeps the
events coincident with LST-1 events. Non-coincident MAGIC events are
discarded since according to simulations they are mostly hadron.

Please note that for the data taken before 12th June 2021, a coincidence
peak should be found around the time offset of -3.1 us. For the data
taken after that date, however, there is an additional global offset and
so the peak is shifted to the time offset of -6.5 us. Thus, it would be
needed to tune the offset scan region depending on the date when the
data were taken. The reason of the shift is under investigation.

Unless there is any particular reason, please use the default half width
300 ns for the coincidence window, which is optimized to reduce the
accidental coincidence rate as much as possible by keeping the number of
actual coincident events.

Usage:
$ python lst1_magic_event_coincidence.py
--input-file-lst dl1/LST-1/dl1_LST-1.Run03265.0040.h5
--input-dir-magic dl1/MAGIC
--output-dir dl1_coincidence
--config-file config.yaml
"""

import argparse
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from ctapipe.instrument import SubarrayDescription
from magicctapipe.io import (
    TEL_NAMES,
    get_stereo_events,
    load_lst_dl1_data_file,
    load_magic_dl1_data_files,
    save_pandas_data_in_table,
)

__all__ = ["event_coincidence"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The conversion factor from seconds to nanoseconds
SEC2NSEC = 1e9

# The final digit of timestamps
TIME_ACCURACY = 100 * u.ns

# The telescope positions defined in a simulation
TEL_POSITIONS = {
    1: [-8.09, 77.13, 0.78] * u.m,  # LST-1
    2: [39.3, -62.55, -0.97] * u.m,  # MAGIC-I
    3: [-31.21, -14.57, 0.2] * u.m,  # MAGIC-II
}


def event_coincidence(input_file_lst, input_dir_magic, output_dir, config):
    """
    Searches for coincident events from LST-1 and MAGIC joint
    observation data offline using their timestamps.

    Parameters
    ----------
    input_file_lst: str
        Path to an input LST-1 DL1 data file
    input_dir_magic: str
        Path to a directory where input MAGIC DL1 data files are stored
    output_dir: str
        Path to a directory where to save an output DL1 data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_coincidence = config["event_coincidence"]

    # Load the input LST-1 data file
    logger.info(f"\nInput LST-1 file:\n{input_file_lst}")

    data_lst, subarray_lst = load_lst_dl1_data_file(input_file_lst)

    # Load the input MAGIC data files
    logger.info(f"\nInput MAGIC directory:\n{input_dir_magic}")

    data_magic, subarray_magic = load_magic_dl1_data_files(input_dir_magic)

    # Exclude the parameters non-common to LST-1 and MAGIC data
    timestamp_type_lst = config_coincidence["timestamp_type_lst"]
    logger.info(f"\nLST timestamp type: {timestamp_type_lst}")

    data_lst.rename(columns={timestamp_type_lst: "timestamp"}, inplace=True)

    params_lst = set(data_lst.columns) ^ set(["timestamp"])
    params_magic = set(data_magic.columns) ^ set(["time_sec", "time_nanosec"])
    params_non_common = list(params_lst ^ params_magic)

    data_lst.drop(params_non_common, axis=1, errors="ignore", inplace=True)
    data_magic.drop(params_non_common, axis=1, errors="ignore", inplace=True)

    # Prepare for the event coincidence
    window_half_width = u.Quantity(config_coincidence["window_half_width"])
    window_half_width = u.Quantity(window_half_width.to(u.ns).round(), dtype=int)

    logger.info(f"\nCoincidence window half width: {window_half_width}")

    offset_start = u.Quantity(config_coincidence["time_offset"]["start"])
    offset_stop = u.Quantity(config_coincidence["time_offset"]["stop"])

    logger.info(f"\nTime offsets:\n\tstart: {offset_start}\n\tstop: {offset_stop}")

    time_offsets = np.arange(
        start=offset_start.to_value(u.ns).round(),
        stop=offset_stop.to_value(u.ns).round(),
        step=TIME_ACCURACY.to_value(u.ns).round(),
    )

    time_offsets = u.Quantity(time_offsets.round(), unit=u.ns, dtype=int)

    event_data = pd.DataFrame()
    features = pd.DataFrame()
    profiles = pd.DataFrame(data={"time_offset": time_offsets.to_value(u.us).round(1)})

    # Arrange the LST timestamps. They are stored in the UNIX format in
    # units of seconds with 17 digits, 10 digits for the integral part
    # and 7 digits for the fractional part (up to 100 ns order). For the
    # coincidence search, however, it is too long to precisely find
    # coincident events if we keep using the default data type "float64"
    # due to a rounding issue. Thus, here we scale the timestamps to the
    # unit of nanoseconds and use the "int64" type, which can keep a
    # value up to ~20 digits. In order to precisely scale the timestamps
    # we use the "Decimal" module.

    time_lst = np.array([Decimal(str(time)) for time in data_lst["timestamp"]])
    time_lst *= Decimal(str(SEC2NSEC))  # Conversion from seconds to nanoseconds
    time_lst = u.Quantity(time_lst, unit=u.ns, dtype=int)

    # Check the event coincidence per telescope combination
    tel_ids = np.unique(data_magic.index.get_level_values("tel_id"))

    for tel_id in tel_ids:

        tel_name = TEL_NAMES[tel_id]
        df_magic = data_magic.query(f"tel_id == {tel_id}").copy()

        # Arrange the MAGIC timestamps to the same scale as LST-1
        seconds = np.array([Decimal(str(time)) for time in df_magic["time_sec"]])
        nseconds = np.array([Decimal(str(time)) for time in df_magic["time_nanosec"]])

        time_magic = seconds * Decimal(str(SEC2NSEC)) + nseconds
        time_magic = u.Quantity(time_magic, unit=u.ns, dtype=int)

        df_magic["timestamp"] = time_magic.to_value(u.s)
        df_magic.drop(["time_sec", "time_nanosec"], axis=1, inplace=True)

        # Extract the MAGIC events taken when LST-1 observed
        logger.info(f"\nExtracting the {tel_name} events taken when LST-1 observed...")

        mask = np.logical_and(
            time_magic >= time_lst[0] + time_offsets[0] - window_half_width,
            time_magic <= time_lst[-1] + time_offsets[-1] + window_half_width,
        )

        n_events_magic = np.count_nonzero(mask)

        if n_events_magic == 0:
            logger.warning(f"--> No {tel_name} events are found. Skipping...")
            continue

        logger.info(f"--> {n_events_magic} events are found.")

        df_magic = df_magic.iloc[mask]
        time_magic = time_magic[mask]

        # Start checking the event coincidence. The time offsets and the
        # coincidence window are applied to the LST-1 events, and the
        # MAGIC events existing in the window, including the edges, are
        # recognized as the coincident events. At first, we scan the
        # number of coincident events in each time offset and find the
        # offset maximizing the number of events. Then, we compute the
        # average offset weighted by the number of events around the
        # maximizing offset. Finally, we again check the coincidence at
        # the average offset and then save the coincident events.

        # Note that there are two conditions for the event coincidence.
        # The first one includes both edges of the coincidence window,
        # and the other one includes only the right edge. The latter
        # is used to estimate the number of coincident events between
        # the time offset steps.

        n_events_lst = len(time_lst)

        n_coincidence = np.zeros(len(time_offsets), dtype=int)
        n_coincidence_btwn = np.zeros(len(time_offsets), dtype=int)

        logger.info("\nChecking the event coincidence...")

        for i_step, time_offset in enumerate(time_offsets):

            time_lolim = time_lst + time_offset - window_half_width
            time_uplim = time_lst + time_offset + window_half_width

            for i_evt in range(n_events_lst):

                condition_lo = time_magic.value >= time_lolim[i_evt].value
                condition_lo_wo_eq = time_magic.value > time_lolim[i_evt].value
                condition_hi = time_magic.value <= time_uplim[i_evt].value

                # Check the coincidence including the both edges
                coincidence_mask = np.logical_and(condition_lo, condition_hi)

                if np.count_nonzero(coincidence_mask) == 1:
                    n_coincidence[i_step] += 1

                # Check the coincidence including only the right edge
                coincidence_mask_btwn = np.logical_and(condition_lo_wo_eq, condition_hi)

                if np.count_nonzero(coincidence_mask_btwn) == 1:
                    n_coincidence_btwn[i_step] += 1

            logger.info(
                f"time offset: {time_offset.to(u.us).round(1)} "
                f"--> {n_coincidence[i_step]} events"
            )

        # Sometimes there are more than one time offsets maximizing the
        # number of coincident events, so here we calculate the mean
        offset_at_max = time_offsets[n_coincidence == n_coincidence.max()].mean()

        # The half width of the average region is defined as the "full"
        # width of the coincidence window, since the width of the
        # coincidence distribution becomes larger than that of the
        # coincidence window due to the uncertainty of the timestamps
        mask = np.logical_and(
            time_offsets >= np.round(offset_at_max - 2 * window_half_width),
            time_offsets <= np.round(offset_at_max + 2 * window_half_width),
        )

        average_offset = np.average(time_offsets[mask], weights=n_coincidence[mask])

        # The number of coincident events at the average offset can be
        # estimated from the ones for the time offset steps
        n_events_at_avg = n_coincidence_btwn[time_offsets < average_offset][-1]
        percentage = np.round(100 * n_events_at_avg / n_events_magic, 1)

        logger.info(
            f"\nAverage offset: {average_offset.to(u.us).round(3)}"
            f"\n--> Number of coincident events: {n_events_at_avg}"
            f"\n--> Ratio over the {tel_name} events: "
            f"{n_events_at_avg}/{n_events_magic} = {percentage}%"
        )

        # Check again the coincidence at the offset where the same
        # result is expected as the average offset
        optimized_offset = time_offsets[time_offsets < average_offset][-1]

        time_lolim = time_lst + optimized_offset - window_half_width
        time_uplim = time_lst + optimized_offset + window_half_width

        indices_lst = []
        indices_magic = []

        for i_evt in range(n_events_lst):

            # Use only the right edge condition
            coincidence_mask = np.logical_and(
                time_magic.value > time_lolim[i_evt].value,
                time_magic.value <= time_uplim[i_evt].value,
            )

            if np.count_nonzero(coincidence_mask) == 1:
                indices_lst.append(i_evt)
                indices_magic.append(np.where(coincidence_mask)[0][0])

        multi_indices_magic = df_magic.iloc[indices_magic].index
        obs_ids_magic = multi_indices_magic.get_level_values("obs_id_magic")
        event_ids_magic = multi_indices_magic.get_level_values("event_id_magic")

        # Keep only the LST-1 events coincident with the MAGIC events
        df_lst = data_lst.iloc[indices_lst].copy()
        df_lst["obs_id_magic"] = obs_ids_magic
        df_lst["event_id_magic"] = event_ids_magic
        df_lst.reset_index(inplace=True)
        df_lst.set_index(["obs_id_magic", "event_id_magic", "tel_id"], inplace=True)

        # Arrange the data frames
        coincidence_id = "1" + str(tel_id)  # Combination of the telescope IDs

        df_feature = pd.DataFrame(
            data={
                "coincidence_id": [int(coincidence_id)],
                "window_half_width": [window_half_width.value],
                "unix_time": [df_lst["timestamp"].mean()],
                "pointing_alt_lst": [df_lst["pointing_alt"].mean()],
                "pointing_alt_magic": [df_magic["pointing_alt"].mean()],
                "pointing_az_lst": [df_lst["pointing_az"].mean()],
                "pointing_az_magic": [df_magic["pointing_az"].mean()],
                "average_offset": [average_offset.to_value(u.us)],
                "n_coincidence": [n_events_at_avg],
                "n_events_magic": [n_events_magic],
            }
        )

        df_profile = pd.DataFrame(
            data={
                "time_offset": time_offsets.to_value(u.us).round(1),
                f"n_coincidence_tel{coincidence_id}": n_coincidence,
                f"n_coincidence_btwn_tel{coincidence_id}": n_coincidence_btwn,
            }
        )

        event_data = pd.concat([event_data, df_lst, df_magic])
        features = features.append(df_feature)
        profiles = pd.merge(left=profiles, right=df_profile, on="time_offset")

    if event_data.empty:
        logger.warning("\nNo coincident events are found. Exiting...\n")
        sys.exit()

    event_data.sort_index(inplace=True)
    event_data.drop_duplicates(inplace=True)

    # It sometimes happen that even if it is a MAGIC-stereo event, only
    # M1 or M2 event is coincident with a LST-1 event. In that case we
    # keep both M1 and M2 events, since they are recognized as the same
    # shower event by the MAGIC-stereo hardware trigger.

    # It also happens that a MAGIC-stereo event is coincident with
    # different LST-1 events, i.e., M1 event is coincident with a LST-1
    # event that is different from the one that M2 event is coincident.
    # Here we drop such kind of events at the moment.

    # Finally, we drop the MAGIC-stereo events non-coincident with LST-1
    # since according to simulations they are mostly hadronic origin.

    group_mean = event_data.groupby(["obs_id_magic", "event_id_magic"]).mean()

    event_data["obs_id"] = group_mean["obs_id"]
    event_data["event_id"] = group_mean["event_id"]

    event_data.dropna(subset=["obs_id", "event_id"], inplace=True)

    event_data = event_data.astype({"obs_id": int, "event_id": int})
    event_data.reset_index(inplace=True)
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)

    event_data = get_stereo_events(event_data)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    input_file_name = Path(input_file_lst).name

    output_file_name = input_file_name.replace("LST-1", "LST-1_MAGIC")
    output_file = f"{output_dir}/{output_file_name}"

    event_data.reset_index(inplace=True)

    save_pandas_data_in_table(
        event_data, output_file, group_name="/events", table_name="parameters", mode="w"
    )

    save_pandas_data_in_table(
        features, output_file, group_name="/coincidence", table_name="feature", mode="a"
    )

    save_pandas_data_in_table(
        profiles, output_file, group_name="/coincidence", table_name="profile", mode="a"
    )

    tel_descriptions = {
        1: subarray_lst.tel[1],  # LST-1
        2: subarray_magic.tel[2],  # MAGIC-I
        3: subarray_magic.tel[3],  # MAGIC-II
    }

    subarray_lst1_magic = SubarrayDescription(
        "LST1-MAGIC-Array", TEL_POSITIONS, tel_descriptions
    )

    subarray_lst1_magic.to_hdf(output_file)

    logger.info(f"\nOutput file:\n{output_file}")


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file-lst",
        "-l",
        dest="input_file_lst",
        type=str,
        required=True,
        help="Path to an input LST-1 DL1 data file.",
    )

    parser.add_argument(
        "--input-dir-magic",
        "-m",
        dest="input_dir_magic",
        type=str,
        required=True,
        help="Path to a directory where input MAGIC DL1 data files are stored.",
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

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Check the event coincidence
    event_coincidence(
        args.input_file_lst, args.input_dir_magic, args.output_dir, config
    )

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
