#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script finds coincident events from LST-1 and MAGIC joint observation data offline using their timestamps.
The MAGIC standard stereo analysis discards shower events when one of the telescope images cannot survive the
cleaning or fail to compute the DL1 parameters. However, it's possible to perform the stereo analysis if LST-1 sees these events.
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
import tables
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal
from astropy import units as u
from astropy.time import Time
from ctapipe.instrument import SubarrayDescription

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

nsec2sec = 1e-9
sec2usec = 1e6

accuracy_time = 1e-7   # final digit of timestamp, unit: [sec]

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

tel_combinations = {
    'm1_m2': [2, 3],   # event_type = 0
    'lst1_m1': [1, 2],   # event_type = 1
    'lst1_m2': [1, 3],   # event_type = 2
    'lst1_m1_m2': [1, 2, 3],   # event_type = 3
}

__all__ = [
    'event_coincidence',
]


def load_lst_data(input_file, type_lst_time):
    """
    Loads an input LST-1 data file.

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

    # Try to read DL2 data, but if it does not exist, read DL1 data:
    try:
        data = pd.read_hdf(input_file, key=f'dl2/event/telescope/parameters/LST_LSTCam')
    except:
        data = pd.read_hdf(input_file, key=f'dl1/event/telescope/parameters/LST_LSTCam')

    logger.info(f'LST-1: {len(data)} events')

    # Exclude the events that lstchain could not reconstruct:
    logger.info('\nExcluding non-reconstructed events...')
    params_basic = ['intensity', 'time_gradient', 'alt_tel', 'az_tel']

    data.dropna(subset=params_basic, inplace=True)
    logger.info(f'--> LST-1: {len(data)} events')

    # It sometimes happens that cosmic and pedestal events have the same event IDs,
    # so let's check the duplication of the IDs and exclude them if they exist.
    # ToBeChecked: if the duplication still happens in the latest data or not:
    event_ids, counts = np.unique(data['event_id'], return_counts=True)

    if np.any(counts > 1):
        event_ids_dup = event_ids[counts > 1].tolist()
        logger.info(f'\nExcluding the following events due to the duplication of event IDs:\n{event_ids_dup}')

        data.query(f'event_id != {event_ids_dup}', inplace=True)
        logger.info(f'--> LST-1: {len(data)} events')

    # Keep only the specified type of timestamp and rename it to "timestamp":
    params_time = np.array(['dragon_time', 'tib_time', 'ucts_time', 'trigger_time'])
    params_time_drop = params_time[params_time != type_lst_time]

    data.drop(params_time_drop, axis=1, inplace=True)
    data.rename(columns={type_lst_time: 'timestamp'}, inplace=True)

    # Rename the column names, which lstchain renamed, to the default ones to make them consistent with MC data.
    # In addition, put the suffix "_lst" to the observation/event IDs to avoid a confusion from the MAGIC ones:
    params_rename = {
        'obs_id': 'obs_id_lst',
        'event_id': 'event_id_lst',
        'leakage_pixels_width_1': 'pixels_width_1',
        'leakage_pixels_width_2': 'pixels_width_2',
        'leakage_intensity_width_1': 'intensity_width_1',
        'leakage_intensity_width_2': 'intensity_width_2',
        'time_gradient': 'slope',
    }

    data.rename(columns=params_rename, inplace=True)
    data.set_index(['obs_id_lst', 'event_id_lst', 'tel_id'], inplace=True)

    # Change the units of parameters, that lstchain changed, to the default ones to make them consistent with MC data:
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
    Loads input MAGIC data files.

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

    if file_paths == []:
        logger.error('\nCould not find MAGIC data files in the input directory. Exiting.')
        sys.exit()

    # Load and merge the input files:
    logger.info('\nLoading the following MAGIC data files:')

    data = pd.DataFrame()

    for path in file_paths:
        logger.info(path)
        df = pd.read_hdf(path, key='events/params')
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
    Finds coincident events from LST-1 and MAGIC
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
    logger.info(f'\nConfiguration for the event coincidence:\n{config_evco}')

    # Configure the event coincidence:
    precision = int(-np.log10(accuracy_time))
    window_width = config_evco['window_width']

    bins_offset = np.arange(
        start=config_evco['offset_start'],
        stop=config_evco['offset_stop'],
        step=accuracy_time,
    )

    bins_offset = np.round(bins_offset, decimals=precision)

    # Loading the input files:
    data_lst, subarray_lst = load_lst_data(input_file_lst, config_evco['type_lst_time'])
    data_magic, subarray_magic = load_magic_data(input_dir_magic)

    tel_descriptions = {
        1: subarray_lst.tel[1],      # LST-1
        2: subarray_magic.tel[2],    # MAGIC-I
        3: subarray_magic.tel[3],    # MAGIC-II
    }

    subarray_lst1_magic = SubarrayDescription('LST1-MAGIC-Array', tel_positions, tel_descriptions)

    if not keep_all_params:
        # Exclude the parameters non-common to both LST and MAGIC data except the event timing information:
        params_lst = set(data_lst.columns) ^ set(['timestamp'])
        params_magic = set(data_magic.columns) ^ set(['time_sec', 'time_nanosec'])
        params_non_common = list(params_lst ^ params_magic)

        data_lst.drop(params_non_common, axis=1, errors='ignore', inplace=True)
        data_magic.drop(params_non_common, axis=1, errors='ignore', inplace=True)

    # Arrange the LST timestamp. It is originally stored in the UNIX format with 17 digits,
    # which is too long to precisely find coincident events due to the rounding issue.
    # Thus, here we arrange the timestamp so that it starts from an observation day, i.e.,　subtract
    # the UNIX time of an observation day from the timestamp. As a result, the timestamp becomes
    # ten digits and can be safely handled by keeping the precision.

    # Get the UNIX time of an observation day by rounding the first event timing information:
    obs_day_mjd = np.round(Time(data_lst['timestamp'].iloc[0], format='unix', scale='utc').mjd)
    obs_day_unix = np.round(Time(obs_day_mjd, format='mjd', scale='utc').unix)

    # Use the Decimal module to safely subtract the UNIX time of an observation day:
    time_lst_unix = np.array([Decimal(str(time)) for time in data_lst['timestamp']])
    time_lst = time_lst_unix - Decimal(str(obs_day_unix))

    # Get back to the float format, because the coincidence algorithm takes time if we keep
    # using the Decimal module. It if confirmed that using the float type doesn't change the results:
    time_lst = time_lst.astype(float)

    df_events = pd.DataFrame()
    df_features = pd.DataFrame()
    df_profile = pd.DataFrame(
        data={'offset_usec': bins_offset * sec2usec},
    )

    # Perform the event coincidence per telescope combination:
    telescope_ids = np.unique(data_magic.index.get_level_values('tel_id'))

    for tel_id in telescope_ids:

        tel_name = tel_names[tel_id]
        df_magic = data_magic.query(f'tel_id == {tel_id}')

        # Arrange MAGIC timestamp. The "time_sec" and "time_nanosec" are the integral and fractional
        # part of the timestamp, respectively. Here we arrange it to the same format of the LST timestamp:
        time_sec = df_magic['time_sec'].to_numpy() - obs_day_unix
        time_nanosec = df_magic['time_nanosec'].to_numpy() * nsec2sec

        time_magic = np.round(time_sec + time_nanosec, decimals=precision)

        # Extract the MAGIC events taken when LST observes:
        logger.info(f'\nExtracting the {tel_name} events within the LST-1 observation time window...')

        condition = np.logical_and(
            time_magic > time_lst[0] + bins_offset[0] - window_width,
            time_magic < time_lst[-1] + bins_offset[-1] + window_width,
        )

        n_events_magic = np.count_nonzero(condition)

        if n_events_magic == 0:
            logger.info(f'--> No {tel_name} events are found within the LST-1 observation time window. Skipping.')
            continue

        logger.info(f'--> {n_events_magic} events are found. Checking the event coincidence...\n')

        df_magic = df_magic.iloc[condition]
        time_magic = time_magic[condition]

        # Start the event coincidence. The time offsets and the coincidence window apply to the LST-1 events,
        # and the MAGIC events existing in the window (including the edges) are recognized as coincident events.
        # At first, we scan the number of coincident events in each time offset, and find the time offset
        # where the number of events becomes maximum. Then, we compute the averaged offset weighted by the
        # number of events around the maximizing offset. Finally, we again check coincident events at the
        # average offset and store the events in the data container.

        # Note that there are two conditions for the event coincidence. The first one includes
        # both edges of the coincidence window, and the other one includes only the right edge.
        # The latter means the number of coincident events between the time offset steps:

        n_events_lst = len(time_lst)

        n_events_stereo = np.zeros(len(bins_offset), dtype=np.int)
        n_events_stereo_btwn = np.zeros(len(bins_offset), dtype=np.int)

        for i_off, offset in enumerate(bins_offset):

            time_lolim = np.round(time_lst + offset - window_width/2, decimals=precision)
            time_uplim = np.round(time_lst + offset + window_width/2, decimals=precision)

            # Loop over the LST-1 events to check the coincident events. It is also possible to create
            # a 2D array with the shape (n_events_lst, n_events_magic) to check the coincidence all at once.
            # However, handling an array of ~10 million elements takes time and it is faster to loop over the events:

            for i_ev in range(n_events_lst):

                # Check the coincidence including both edges:
                condition = np.logical_and(
                    time_magic >= time_lolim[i_ev],
                    time_magic <= time_uplim[i_ev],
                )

                if np.count_nonzero(condition) == 1:
                    n_events_stereo[i_off] += int(1)

                # Check the coincidence including only the right edge:
                condition_btwn = np.logical_and(
                    time_magic > time_lolim[i_ev],
                    time_magic <= time_uplim[i_ev],
                )

                if np.count_nonzero(condition_btwn) == 1:
                    n_events_stereo_btwn[i_off] += int(1)

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
        indices_lst = []
        indices_magic = []

        offset = bins_offset[bins_offset < offset_avg][-1]

        time_lolim = np.round(time_lst - window_width/2 + offset, decimals=precision)
        time_uplim = np.round(time_lst + window_width/2 + offset, decimals=precision)

        for i_ev in range(n_events_lst):

            condition = np.logical_and(
                time_magic > time_lolim[i_ev],
                time_magic <= time_uplim[i_ev],
            )

            if np.count_nonzero(condition) == 1:
                indices_lst.append(i_ev)
                indices_magic.append(np.where(condition)[0][0])

        # Arrange the data frames:
        df_lst = data_lst.iloc[indices_lst]
        df_lst['obs_id'] = df_magic.iloc[indices_magic].index.get_level_values('obs_id')
        df_lst['event_id'] = df_magic.iloc[indices_magic].index.get_level_values('event_id')
        df_lst.reset_index(inplace=True)
        df_lst.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

        df_magic['timestamp'] = df_magic['time_sec'] + df_magic['time_nanosec'] * nsec2sec
        df_magic.drop(['time_sec', 'time_nanosec'], axis=1, inplace=True)

        # Define the coincidence ID, which is the combination of the telescope IDs:
        coincidence_id = '1' + str(tel_id)

        features_per_combo = pd.DataFrame({
            'coincidence_id': [int(coincidence_id)],
            'mean_time_unix': [df_lst['timestamp'].mean()],
            'mean_alt_lst': [df_lst['alt_tel'].mean()],
            'mean_alt_magic': [df_magic['alt_tel'].mean()],
            'mean_az_lst': [df_lst['az_tel'].mean()],
            'mean_az_magic': [df_magic['az_tel'].mean()],
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

        df_profile = pd.merge(
            left=df_profile,
            right=profile_per_combo,
            on='offset_usec',
        )

    if df_events.empty:
        logger.info('\nNo coincident events are found. Exiting.\n')
        sys.exit()

    # Since we merge two data frames, LST-1 + M1 and LST-1 + M2,
    # the LST-1 events are duplicated and so here we exclude them:
    df_events.sort_index(inplace=True)
    df_events.drop_duplicates(inplace=True)

    # Keep only the events containing two or three telescope information.
    # Sometimes it happens that one MAGIC stereo event is coincident with two
    # different LST-1 events, and at the moment we exclude that kind of events:
    df_events['multiplicity'] = df_events.groupby(['obs_id', 'event_id']).size()
    df_events = df_events.query('multiplicity == [2, 3]')

    # Check the number of coincident events per event type:
    n_events_total = len(df_events.groupby(['obs_id', 'event_id']).size())
    logger.info(f'\nIn total {n_events_total} stereo events are found:')

    for event_type, (tel_combo, tel_ids) in enumerate(tel_combinations.items()):

        df = df_events.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        df['multiplicity'] = df.groupby(['obs_id', 'event_id']).size()
        df.query(f'multiplicity == {len(tel_ids)}', inplace=True)

        n_events = len(df.groupby(['obs_id', 'event_id']).size())
        logger.info(f'{tel_combo}: {n_events:.0f} events ({n_events / n_events_total * 100:.1f}%)')

        df_events.loc[df.index, 'event_type'] = event_type

    # Prepare for saving the data to an output file.
    # Here we parse run information from the input file name:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file_lst).resolve().name
    regex = r'(\S+)_LST-1\.(\S+)\.h5'

    if re.fullmatch(regex, base_name):
        parser = re.findall(regex, base_name)[0]
        output_file = f'{output_dir}/{parser[0]}_LST-1_MAGIC.{parser[1]}.h5'

    # Save in the output file:
    with tables.open_file(output_file, mode='w') as f_out:

        df_events.reset_index(inplace=True)
        event_values = [tuple(array) for array in df_events.to_numpy()]
        dtypes = np.dtype([(name, dtype) for name, dtype in zip(df_events.dtypes.index, df_events.dtypes)])

        event_table = np.array(event_values, dtype=dtypes)
        f_out.create_table('/events', 'params', createparents=True, obj=event_table)

    df_features.to_hdf(output_file, key='coincidence/features', mode='a', format='table')
    df_profile.to_hdf(output_file, key='coincidence/profile', mode='a', format='table')

    # Save the subarray description:
    subarray_lst1_magic.to_hdf(output_file)

    logger.info('\nOutput file:')
    logger.info(output_file)

    logger.info('\nDone.')


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

    event_coincidence(
        args.input_file_lst,
        args.input_dir_magic,
        args.output_dir,
        config,
        args.keep_all_params,
    )

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
