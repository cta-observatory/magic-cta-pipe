#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script finds coincident events from LST-1 and MAGIC joint observation data offline using their timestamps.
The MAGIC standard stereo analysis discards shower events when one of the telescope images cannot survive the cleaning
or fail to compute the DL1 parameters. However, it's possible to perform the stereo analysis if LST-1 sees these events.
Thus, the script checks the event coincidence for each telescope combination (i.e., LST-1 + M1 and LST-1 + M2) using each
MAGIC timestamp. Then, it saves the events containing more than two telescope information to an output file.

Time offsets and the coincidence window apply to LST-1 events, and the script searches for coincident events
within the offset region specified in a configuration file. A peak is usually found in -10 < offset < 0 [us].
Since the optimal time offset changes depending on the telescope distance along the pointing direction,
it requires to input a sub-run file for LST data, whose observation duration is usually ~10 seconds.

Usage:
$ python lst1_magic_event_coincidence.py
--input-file-lst ./data/dl1/LST-1/dl1_LST-1.Run03265.0040.h5
--input-dir-magic ./data/dl1/MAGIC
--output-dir ./data/dl1_coincidence
--config-file ./config.yaml
"""

import re
import sys
import glob
import time
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal
from astropy import units as u
from astropy.time import Time
from ctapipe.instrument import SubarrayDescription
from magicctapipe.utils import (
    check_tel_combination,
    save_pandas_to_table,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

nsec2sec = 1e-9
usec2sec = 1e-6
sec2usec = 1e6

accuracy_time = 1e-7   # final digit of a timestamp, unit: [sec]

tel_names = {
    1: 'LST-1',
    2: 'MAGIC-I',
    3: 'MAGIC-II',
}

tel_positions = {
    1: u.Quantity([-8.09, 77.13, 0.78], u.m),
    2: u.Quantity([39.3, -62.55, -0.97], u.m),
    3: u.Quantity([-31.21, -14.57, 0.2], u.m),
}

__all__ = [
    'event_coincidence',
]


def load_lst_data(input_file, type_lst_time):
    """
    Load an input LST-1 data file.

    Parameters
    ----------
    input_file: str
        Path to an input LST-1 data file
    type_lst_time: str
        Type of LST-1 timestamp used for the event coincidence

    Returns
    -------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing LST-1 events
    subarray: SubarrayDescription
        LST-1 subarray description
    """

    logger.info('\nLoading the input LST-1 data file:')
    logger.info(input_file)

    base_name = Path(input_file).resolve().name
    regex = r'(\S+)_LST-1\.\S+\.h5'

    data_level = re.findall(regex, base_name)[0]
    data = pd.read_hdf(input_file, key=f'{data_level}/event/telescope/parameters/LST_LSTCam')

    # Exclude the non-reconstructed events:
    data.dropna(subset=['intensity', 'time_gradient', 'alt_tel'], inplace=True)

    # Check the duplication of event IDs and exclude them if they exist.
    # ToBeChecked: if the duplication still happens in the recent data or not:
    event_ids, counts = np.unique(data['event_id'], return_counts=True)

    if np.any(counts > 1):
        event_ids_dup = event_ids[counts > 1].tolist()
        data.query(f'event_id != {event_ids_dup}', inplace=True)

        logger.warning('\nExclude the following events due to the duplication of the event IDs:')
        logger.warning(event_ids_dup)

    logger.info(f'LST-1: {len(data)} events')

    # Rename the column names:
    params_rename = {
        'obs_id': 'obs_id_lst',
        'event_id': 'event_id_lst',
        'alt_tel': 'pointing_alt',
        'az_tel': 'pointing_az',
        'leakage_pixels_width_1': 'pixels_width_1',
        'leakage_pixels_width_2': 'pixels_width_2',
        'leakage_intensity_width_1': 'intensity_width_1',
        'leakage_intensity_width_2': 'intensity_width_2',
        'time_gradient': 'slope',
        type_lst_time: 'timestamp',
    }

    data.rename(columns=params_rename, inplace=True)
    data.set_index(['obs_id_lst', 'event_id_lst', 'tel_id'], inplace=True)

    # Change the units of the parameters:
    optics = pd.read_hdf(input_file, key='configuration/instrument/telescope/optics')
    focal_length = optics['equivalent_focal_length'][0]

    data['length'] = focal_length * np.tan(np.deg2rad(data['length']))
    data['width'] = focal_length * np.tan(np.deg2rad(data['width']))
    data['phi'] = np.rad2deg(data['phi'])
    data['psi'] = np.rad2deg(data['psi'])

    # Read the subarray description:
    subarray = SubarrayDescription.from_hdf(input_file)

    return data, subarray


def load_magic_data(input_dir):
    """
    Load input MAGIC data files.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input MAGIC data files are stored

    Returns
    -------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing MAGIC events
    subarray: SubarrayDescription
        MAGIC subarray description
    """

    file_paths = glob.glob(f'{input_dir}/dl*.h5')
    file_paths.sort()

    if len(file_paths) == 0:
        logger.error('\nCould not find MAGIC data files in the input directory. Exiting.')
        sys.exit()

    # Load and merge the input files:
    logger.info('\nLoading the following MAGIC data files:')

    data = pd.DataFrame()

    for path in file_paths:
        logger.info(path)
        df = pd.read_hdf(path, key='events/parameters')
        data = data.append(df)

    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data.sort_index(inplace=True)

    telescope_ids = np.unique(data.index.get_level_values('tel_id'))

    for tel_id in telescope_ids:
        tel_name = tel_names[tel_id]
        n_events = len(data.query(f'tel_id == {tel_id}'))
        logger.info(f'{tel_name}: {n_events} events')

    # Read the subarray description:
    subarray = SubarrayDescription.from_hdf(file_paths[0])

    return data, subarray


def event_coincidence(
    input_file_lst,
    input_dir_magic,
    output_dir,
    config,
    keep_all_params=False,
):
    """
    Find coincident events from LST-1 and MAGIC
    joint observation data offline using their timestamps.

    Parameters
    ----------
    input_file_lst: str
        Path to an input LST-1 data file
    input_dir_magic: str
        Path to a directory where input MAGIC data files are stored
    output_dir: str
        Path to a directory where to save an output coincidence data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    keep_all_params: bool
        If true, it also saves the parameters
        non-common to LST-1 and MAGIC events (default: false)
    """

    config_evco = config['event_coincidence']

    # Load the input files:
    data_lst, subarray_lst = load_lst_data(input_file_lst, config_evco['type_lst_time'])
    data_magic, subarray_magic = load_magic_data(input_dir_magic)

    tel_descriptions = {
        1: subarray_lst.tel[1],      # LST-1
        2: subarray_magic.tel[2],    # MAGIC-I
        3: subarray_magic.tel[3],    # MAGIC-II
    }

    subarray_lst1_magic = SubarrayDescription('LST1-MAGIC-Array', tel_positions, tel_descriptions)

    if not keep_all_params:

        params_lst = set(data_lst.columns) ^ set(['timestamp'])
        params_magic = set(data_magic.columns) ^ set(['time_sec', 'time_nanosec'])
        params_non_common = list(params_lst ^ params_magic)

        data_lst.drop(params_non_common, axis=1, errors='ignore', inplace=True)
        data_magic.drop(params_non_common, axis=1, errors='ignore', inplace=True)

    # Configure the event coincidence:
    logger.info('\nConfiguration for the event coincidence:')
    logger.info(config_evco)

    precision = int(-np.log10(accuracy_time))

    window_width = config_evco['window_width'] * usec2sec

    bins_offset = np.arange(
        config_evco['offset_start'] * usec2sec,
        config_evco['offset_stop'] * usec2sec,
        step=accuracy_time,
    )

    # Arrange the LST timestamp. It is originally stored in the UNIX format with 17 digits,
    # which is too long to precisely find coincident events due to the rounding issue.
    # Thus, here we arrange the timestamp so that it starts from an observation day, i.e.,　subtract
    # the UNIX time of an observation day from the timestamp. As a result, the timestamp becomes
    # ten digits and can be safely handled by keeping the precision.

    # The UNIX time of an observation day can be obtained by rounding the first event timing information.
    # Here we use the Decimal module to safely subtract the UNIX time of an observation day.
    # Then, we get the timestamp back to the float type because the coincidence algorithm takes time if we keep
    # using the Decimal module. It if confirmed that using the float type doesn't change the results:

    obs_day_mjd = np.round(Time(data_lst['timestamp'].iloc[0], format='unix', scale='utc').mjd)
    obs_day_unix = np.round(Time(obs_day_mjd, format='mjd', scale='utc').unix)

    time_lst_unix = np.array([Decimal(str(time)) for time in data_lst['timestamp']])
    time_lst = np.float64(time_lst_unix - Decimal(str(obs_day_unix)))

    # Check the event coincidence per telescope combination:
    telescope_ids = np.unique(data_magic.index.get_level_values('tel_id'))

    df_events = pd.DataFrame()
    df_features = pd.DataFrame()
    df_profile = pd.DataFrame(
        data={'offset_usec': bins_offset * sec2usec},
    )

    for tel_id in telescope_ids:

        tel_name = tel_names[tel_id]
        df_magic = data_magic.query(f'tel_id == {tel_id}')

        # Arrange the MAGIC timestamp to the same format of the LST timestamp:
        time_sec = df_magic['time_sec'].to_numpy() - obs_day_unix
        time_nanosec = df_magic['time_nanosec'].to_numpy() * nsec2sec

        time_magic = np.round(time_sec + time_nanosec, decimals=precision)

        # Extract the MAGIC events taken when LST observed:
        logger.info(f'\nExtracting the {tel_name} events within the LST-1 observation time window...')

        condition = np.logical_and(
            time_magic > time_lst[0] + bins_offset[0] - window_width,
            time_magic < time_lst[-1] + bins_offset[-1] + window_width,
        )

        n_events_magic = np.count_nonzero(condition)

        if n_events_magic == 0:
            logger.warning(f'--> No {tel_name} events are found. Skipping.')
            continue

        logger.info(f'--> {n_events_magic} events are found. Checking the event coincidence...\n')

        df_magic = df_magic.iloc[condition]
        time_magic = time_magic[condition]

        # Start checking the event coincidence. The time offsets and the coincidence window apply to the LST-1 events,
        # and the MAGIC events existing in the window (including the edges) are recognized as coincident events.
        # At first, we scan the number of coincident events in each time offset and find the time offset
        # where the number of events becomes maximum. Then, we compute the averaged offset weighted by the
        # number of events around the maximizing offset. Finally, we again check the coincidence at the
        # average offset and store the coincident events in the data container.

        # Note that there are two conditions for the event coincidence. The first one includes
        # both edges of the coincidence window, and the other one includes only the right edge.
        # The latter means the number of coincident events between the time offset steps:

        n_events_lst = len(time_lst)

        n_events_stereo = np.zeros(len(bins_offset), dtype=int)
        n_events_stereo_btwn = np.zeros(len(bins_offset), dtype=int)

        for i_off, offset in enumerate(bins_offset):

            time_lolim = np.round(time_lst + offset - window_width/2, decimals=precision)
            time_uplim = np.round(time_lst + offset + window_width/2, decimals=precision)

            for i_ev in range(n_events_lst):

                # Check the coincidence including both edges:
                condition = np.logical_and(
                    time_magic >= time_lolim[i_ev],
                    time_magic <= time_uplim[i_ev],
                )

                if np.count_nonzero(condition) == 1:
                    n_events_stereo[i_off] += 1

                # Check the coincidence including only the right edge:
                condition_btwn = np.logical_and(
                    time_magic > time_lolim[i_ev],
                    time_magic <= time_uplim[i_ev],
                )

                if np.count_nonzero(condition_btwn) == 1:
                    n_events_stereo_btwn[i_off] += 1

            logger.info(f'time offset: {offset * sec2usec:.1f} [us]  -->  {n_events_stereo[i_off]} events')

        offsets_at_max = bins_offset[n_events_stereo == np.max(n_events_stereo)]

        mask = np.logical_and(
            bins_offset >= np.round(offsets_at_max[0] - window_width, decimals=precision),
            bins_offset <= np.round(offsets_at_max[-1] + window_width, decimals=precision),
        )

        offset_avg = np.average(bins_offset[mask], weights=n_events_stereo[mask])
        n_events_at_avg = n_events_stereo_btwn[bins_offset < offset_avg][-1]

        logger.info(f'\nAveraged offset: {offset_avg * sec2usec:.3f} [us]')
        logger.info(f'--> Number of coincident events: {n_events_at_avg}')
        logger.info(f'--> Ratio over the {tel_name} events: {n_events_at_avg}/{n_events_magic} ' \
                    f'= {n_events_at_avg / n_events_magic * 100:.1f}%')

        # Check the coincidence at the averaged offset:
        offset = bins_offset[bins_offset < offset_avg][-1]

        time_lolim = np.round(time_lst - window_width/2 + offset, decimals=precision)
        time_uplim = np.round(time_lst + window_width/2 + offset, decimals=precision)

        indices_lst = []
        indices_magic = []

        for i_ev in range(n_events_lst):

            condition = np.logical_and(
                time_magic > time_lolim[i_ev],
                time_magic <= time_uplim[i_ev],
            )

            if np.count_nonzero(condition) == 1:
                indices_lst.append(i_ev)
                indices_magic.append(np.where(condition)[0][0])

        # Arrange the data frames:
        df_lst = data_lst.iloc[indices_lst].copy()
        df_lst['obs_id'] = df_magic.iloc[indices_magic].index.get_level_values('obs_id')
        df_lst['event_id'] = df_magic.iloc[indices_magic].index.get_level_values('event_id')
        df_lst.reset_index(inplace=True)
        df_lst.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

        df_magic['timestamp'] = df_magic['time_sec'] + df_magic['time_nanosec'] * nsec2sec
        df_magic.drop(['time_sec', 'time_nanosec'], axis=1, inplace=True)

        coincidence_id = '1' + str(tel_id)   # Combination of the telescope IDs used for the coincidence

        features_per_combo = pd.DataFrame({
            'coincidence_id': [int(coincidence_id)],
            'mean_time_unix': [df_lst['timestamp'].mean()],
            'mean_alt_lst': [df_lst['pointing_alt'].mean()],
            'mean_alt_magic': [df_magic['pointing_alt'].mean()],
            'mean_az_lst': [df_lst['pointing_az'].mean()],
            'mean_az_magic': [df_magic['pointing_az'].mean()],
            'offset_avg_usec': [offset_avg * sec2usec],
            'n_coincidence': [n_events_at_avg],
            'n_magic': [n_events_magic],
        })

        profile_per_combo = pd.DataFrame({
            'offset_usec': bins_offset * sec2usec,
            f'n_coincidence_tel{coincidence_id}': n_events_stereo,
            f'n_coincidence_btwn_tel{coincidence_id}': n_events_stereo_btwn,
        })

        df_events = pd.concat([df_events, df_lst, df_magic])
        df_features = df_features.append(features_per_combo)
        df_profile = pd.merge(left=df_profile, right=profile_per_combo, on='offset_usec')

    if df_events.empty:
        logger.warning('\nNo coincident events are found. Exiting.\n')
        sys.exit()

    df_events.sort_index(inplace=True)
    df_events.drop_duplicates(inplace=True)

    # Sometimes it happens that one MAGIC stereo event is coincident with two
    # different LST-1 events, and at the moment we exclude that kind of events:
    df_events['multiplicity'] = df_events.groupby(['obs_id', 'event_id']).size()
    df_events = df_events.query('multiplicity == [2, 3]')

    # Set the event types:
    combo_types = check_tel_combination(df_events)
    df_events = df_events.join(combo_types)

    # Save in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file_lst).resolve().name
    regex = r'(\S+)_LST-1\.(\S+)\.h5'

    parser = re.findall(regex, base_name)[0]
    output_file = f'{output_dir}/{parser[0]}_LST-1_MAGIC.{parser[1]}.h5'

    df_events.reset_index(inplace=True)

    save_pandas_to_table(df_events, output_file, '/events', 'parameters')
    save_pandas_to_table(df_features, output_file, '/coincidence', 'features')
    save_pandas_to_table(df_profile, output_file, '/coincidence', 'profile')

    subarray_lst1_magic.to_hdf(output_file)

    logger.info('\nOutput file:')
    logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file-lst', '-l', dest='input_file_lst', type=str, required=True,
        help='Path to an input LST-1 data file.',
    )

    parser.add_argument(
        '--input-dir-magic', '-m', dest='input_dir_magic', type=str, required=True,
        help='Path to a directory where input MAGIC data files are stored.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output coincidence data file.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.',
    )

    parser.add_argument(
        '--keep-all-params', dest='keep_all_params', action='store_true',
        help='Keeps all the parameters of LST-1 and MAGIC events.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # Check the event coincidence:
    event_coincidence(
        args.input_file_lst,
        args.input_dir_magic,
        args.output_dir,
        config,
        args.keep_all_params,
    )

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
