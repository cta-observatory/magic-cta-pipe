#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script searches for coincident events from LST-1 and MAGIC joint observation data offline using their timestamps.
Time offsets and the coincidence window are applied to LST-1 events, and the event coincidence is checked
within an offset region specified in a configuration file. A peak is usually found in -10 < offset < 0 [us].
Since the optimal time offset changes depending on the telescope distance along the pointing direction,
it requires to input a subrun file for LST data, whose observation duration is usually around 10 seconds.

The MAGIC standard stereo analysis discards shower events when one of telescope images cannot survive the image cleaning
or fail to reconstruct the DL1 parameters. However, it's possible to perform the stereo analysis if LST-1 sees the events.
Thus, it checks the event coincidence for each telescope combination (i.e., LST-1 + M1 and LST-1 + M2) and keeps the events
coincident with LST-1 events (non-coincident MAGIC events are discarded at the moment).

If the "--keep-all-params" option is given, it keeps parameters non-common to LST-1 and MAGIC events.

Usage:
$ python lst1_magic_event_coincidence.py
--input-file-lst ./data/dl1/LST-1/dl1_LST-1.Run03265.0040.h5
--input-dir-magic ./data/dl1/MAGIC
--output-dir ./data/dl1_coincidence
--config-file ./config.yaml
(--keep-all-params)
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
from ctapipe.containers import EventType
from ctapipe.instrument import SubarrayDescription
from ctapipe.coordinates import CameraFrame
from lstchain.reco.utils import add_delta_t_key
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

nominal_foclen_lst = 28   # unit: [m]
effective_foclen_lst = 29.30565   # unit: [m]

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
    'load_lst_data_file',
    'load_magic_data_file',
    'event_coincidence',
]


def load_lst_data_file(input_file):
    """
    Loads an input LST-1 data file.

    Parameters
    ----------
    input_file: str
        Path to an input LST-1 data file

    Returns
    -------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of LST-1 events
    subarray: ctapipe.instrument.subarray.SubarrayDescription
        LST-1 subarray description
    """

    logger.info('\nLoading the input LST-1 data file:')
    logger.info(input_file)

    try:
        # Try to load DL2 data at first:
        event_data = pd.read_hdf(input_file, key='dl2/event/telescope/parameters/LST_LSTCam')
    except:
        # Load DL1 data:
        event_data = pd.read_hdf(input_file, key='dl1/event/telescope/parameters/LST_LSTCam')

    event_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    event_data.sort_index(inplace=True)

    # Add the arrival time differences of consecutive events:
    event_data = add_delta_t_key(event_data)

    # Exclude non-reconstructed events:
    event_data.dropna(subset=['intensity', 'time_gradient', 'alt_tel', 'az_tel'], inplace=True)

    # Exclude interleaved events:
    event_type_subarray = EventType.SUBARRAY.value
    event_data.query(f'event_type == {event_type_subarray}', inplace=True)

    # Check the duplications of event IDs and exclude them if they exist.
    # ToBeChecked: if the duplication still happens in recent data or not:
    event_ids, counts = np.unique(event_data.index.get_level_values('event_id'), return_counts=True)

    if np.any(counts > 1):
        event_ids_dup = event_ids[counts > 1].tolist()

        logger.warning('\nExclude the following events due to the duplications of event IDs:')
        logger.warning(event_ids_dup)

        event_data.query(f'event_id != {event_ids_dup}', inplace=True)

    logger.info(f'LST-1: {len(event_data)} events')

    # Rename the column names:
    event_data.rename(
        columns={'delta_t': 'time_diff',
                 'alt_tel': 'pointing_alt',
                 'az_tel': 'pointing_az',
                 'leakage_pixels_width_1': 'pixels_width_1',
                 'leakage_pixels_width_2': 'pixels_width_2',
                 'leakage_intensity_width_1': 'intensity_width_1',
                 'leakage_intensity_width_2': 'intensity_width_2',
                 'time_gradient': 'slope'},
        inplace=True,
    )

    # Change the units of some parameters:
    optics = pd.read_hdf(input_file, key='configuration/instrument/telescope/optics')
    focal_length = optics['equivalent_focal_length'][0]

    event_data['length'] = focal_length * np.tan(np.deg2rad(event_data['length']))   # [deg] -> [m]
    event_data['width'] = focal_length * np.tan(np.deg2rad(event_data['width']))   # [deg] -> [m]

    event_data['phi'] = np.rad2deg(event_data['phi'])   # [rad] -> [deg]
    event_data['psi'] = np.rad2deg(event_data['psi'])   # [rad] -> [deg]

    # Read the subarray description:
    subarray = SubarrayDescription.from_hdf(input_file)

    if focal_length == nominal_foclen_lst:
        # Set the effective focal length to the subarray:
        subarray.tel[1].optics.equivalent_focal_length = u.Quantity(effective_foclen_lst, u.m)
        subarray.tel[1].camera.geometry.frame = CameraFrame(focal_length=u.Quantity(effective_foclen_lst, u.m))

    return event_data, subarray


def load_magic_data_file(input_dir):
    """
    Loads input MAGIC data files.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input MAGIC data files are stored

    Returns
    -------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of MAGIC events
    subarray: ctapipe.instrument.subarray.SubarrayDescription
        MAGIC subarray description
    """

    file_mask = f'{input_dir}/dl*.h5'

    input_files = glob.glob(file_mask)
    input_files.sort()

    if len(input_files) == 0:
        raise FileNotFoundError('Could not find MAGIC data files in the input directory.')

    # Load the input files:
    logger.info('\nLoading the following MAGIC data files:')

    data_list = []

    for input_file in input_files:
        logger.info(input_file)

        df_events = pd.read_hdf(input_file, key='events/parameters')
        data_list.append(df_events)

    event_data = pd.concat(data_list)

    event_data.rename(
        columns={'obs_id': 'obs_id_magic',
                 'event_id': 'event_id_magic'},
        inplace=True,
    )

    event_data.set_index(['obs_id_magic', 'event_id_magic', 'tel_id'], inplace=True)
    event_data.sort_index(inplace=True)

    tel_ids = np.unique(event_data.index.get_level_values('tel_id'))

    for tel_id in tel_ids:
        tel_name = tel_names[tel_id]
        n_events = len(event_data.query(f'tel_id == {tel_id}'))

        logger.info(f'{tel_name}: {n_events} events')

    # Read the subarray description:
    subarray = SubarrayDescription.from_hdf(input_files[0])

    return event_data, subarray


def event_coincidence(input_file_lst, input_dir_magic,
                      output_dir, config, keep_all_params=False):
    """
    Searches for coincident events from LST-1 and MAGIC
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
        If True, it keeps parameters non-common to LST-1 and MAGIC events
    """

    config_evco = config['event_coincidence']

    logger.info('\nConfiguration for the event coincidence:')
    logger.info(config_evco)

    type_lst_time = config_evco['type_lst_time']

    offset_steps = np.arange(
        config_evco['offset_start'] * usec2sec,
        config_evco['offset_stop'] * usec2sec,
        step=accuracy_time,
    )

    precision = int(-np.log10(accuracy_time))
    offset_steps = np.round(offset_steps, decimals=precision)

    window_width = config_evco['window_width'] * usec2sec

    # Load the input files:
    data_lst, subarray_lst = load_lst_data_file(input_file_lst)
    data_magic, subarray_magic = load_magic_data_file(input_dir_magic)

    if not keep_all_params:
        # Exclude parameters non-common to LST-1 and MAGIC events (except timing information):
        params_lst = set(data_lst.columns) ^ set([type_lst_time])
        params_magic = set(data_magic.columns) ^ set(['time_sec', 'time_nanosec'])
        params_non_common = list(params_lst ^ params_magic)

        data_lst.drop(params_non_common, axis=1, errors='ignore', inplace=True)
        data_magic.drop(params_non_common, axis=1, errors='ignore', inplace=True)

    # Arrange the LST timestamp. It is originally stored in the UNIX format with 17 digits,
    # but it is too long to precisely find coincident events due to the rounding issue.
    # Thus, here we arrange the timestamp so that it starts from an observation day,
    # i.e., subtract the UNIX time of an observation day from the timestamp.
    # As a result, the timestamp becomes ten digits and so can be safely handled by keeping the precision.

    # The UNIX time of an observation day is obtained by rounding the first event MJD time.
    # Here we use the Decimal module to safely subtract the UNIX time of an observation day.
    # Then, we get the timestamp back to the float type since the event coincidence takes time if we keep using the Decimal module.
    # It if confirmed that using the float type doesn't change the results:

    time_first_event = Time(data_lst[type_lst_time].iloc[0], format='unix', scale='utc')

    obs_date = Time(np.round(time_first_event.mjd), format='mjd', scale='utc')
    obs_date_unix = np.round(obs_date.unix)

    time_lst_unix = np.array([Decimal(str(time)) for time in data_lst[type_lst_time]])
    time_lst = np.float64(time_lst_unix - Decimal(str(obs_date_unix)))

    # Check the event coincidence per telescope combination:
    tel_ids = np.unique(data_magic.index.get_level_values('tel_id'))

    event_data = pd.DataFrame()
    features = pd.DataFrame()
    profiles = pd.DataFrame({'offset_usec': offset_steps * sec2usec})

    for tel_id in tel_ids:

        tel_name = tel_names[tel_id]
        df_magic = data_magic.query(f'tel_id == {tel_id}')

        # Arrange the MAGIC timestamp to the same scale as the LST timestamp:
        time_sec = df_magic['time_sec'].to_numpy() - obs_date_unix
        time_nanosec = df_magic['time_nanosec'].to_numpy() * nsec2sec

        time_magic = np.round(time_sec + time_nanosec, decimals=precision)

        # Extract the MAGIC events taken when LST-1 observed:
        logger.info(f'\nExtracting the {tel_name} events taken when LST-1 observed...')

        mask = np.logical_and(
            time_magic > time_lst[0] + offset_steps[0] - window_width,
            time_magic < time_lst[-1] + offset_steps[-1] + window_width,
        )

        n_events_magic = np.count_nonzero(mask)

        if n_events_magic == 0:
            logger.warning(f'--> No {tel_name} events are found. Skipping.')
            continue

        logger.info(f'--> {n_events_magic} events are found.')

        df_magic = df_magic.iloc[mask]
        time_magic = time_magic[mask]

        # Start checking the event coincidence. The time offsets and the coincidence window are applied to the LST-1 events,
        # and the MAGIC events existing in the window (including the edges) are recognized as coincident events.
        # At first, we scan the number of coincident events in each time offset and find the offset maximizing the number of events.
        # Then, we compute the average offset weighted by the number of events around the maximizing offset.
        # Finally, we again check the coincidence at the average offset and save the coincident events.

        # Note that there are two conditions for the event coincidence.
        # The first one includes both edges of the coincidence window, and the other one includes only the right edge.
        # The latter means the number of coincident events between the time offset steps:

        n_events_lst = len(time_lst)

        n_events_stereo = np.zeros(len(offset_steps), dtype=int)
        n_events_stereo_btwn = np.zeros(len(offset_steps), dtype=int)

        logger.info('\nChecking the event coincidence...')

        for i_step, offset in enumerate(offset_steps):

            time_lolim = np.round(time_lst + offset - window_width/2, decimals=precision)
            time_uplim = np.round(time_lst + offset + window_width/2, decimals=precision)

            for i_evt in range(n_events_lst):

                # Check the coincidence including the both edges:
                condition = np.logical_and(
                    time_magic >= time_lolim[i_evt],
                    time_magic <= time_uplim[i_evt],
                )

                if np.count_nonzero(condition) == 1:
                    n_events_stereo[i_step] += 1

                # Check the coincidence including only the right edge:
                condition_btwn = np.logical_and(
                    time_magic > time_lolim[i_evt],
                    time_magic <= time_uplim[i_evt],
                )

                if np.count_nonzero(condition_btwn) == 1:
                    n_events_stereo_btwn[i_step] += 1

            logger.info(f'time offset: {offset * sec2usec:.1f} [us]  -->  {n_events_stereo[i_step]} events')

        offsets_at_max = offset_steps[n_events_stereo == np.max(n_events_stereo)]

        mask = np.logical_and(
            offset_steps >= np.round(offsets_at_max[0] - window_width, decimals=precision),
            offset_steps <= np.round(offsets_at_max[-1] + window_width, decimals=precision),
        )

        offset_avg = np.average(offset_steps[mask], weights=n_events_stereo[mask])

        n_events_at_avg = n_events_stereo_btwn[offset_steps < offset_avg][-1]
        ratio = n_events_at_avg / n_events_magic

        logger.info(f'\nAverage offset: {offset_avg * sec2usec:.3f} [us]')
        logger.info(f'--> Number of coincident events: {n_events_at_avg}')
        logger.info(f'--> Ratio over the {tel_name} events: {n_events_at_avg}/{n_events_magic} = {ratio * 100:.1f}%')

        # Check the coincidence at the average offset:
        offset = offset_steps[offset_steps < offset_avg][-1]

        time_lolim = np.round(time_lst - window_width/2 + offset, decimals=precision)
        time_uplim = np.round(time_lst + window_width/2 + offset, decimals=precision)

        indices_lst = []
        indices_magic = []

        for i_evt in range(n_events_lst):

            condition = np.logical_and(
                time_magic > time_lolim[i_evt],
                time_magic <= time_uplim[i_evt],
            )

            if np.count_nonzero(condition) == 1:
                indices_lst.append(i_evt)
                indices_magic.append(np.where(condition)[0][0])

        # Arrange the data frames:
        multi_indices_magic = df_magic.iloc[indices_magic].index

        df_lst = data_lst.iloc[indices_lst].copy()
        df_lst['obs_id_magic'] = multi_indices_magic.get_level_values('obs_id_magic')
        df_lst['event_id_magic'] = multi_indices_magic.get_level_values('event_id_magic')

        df_lst.reset_index(inplace=True)
        df_lst.set_index(['obs_id_magic', 'event_id_magic', 'tel_id'], inplace=True)

        df_lst.rename(columns={type_lst_time: 'timestamp'}, inplace=True)

        df_magic['timestamp'] = df_magic['time_sec'] + df_magic['time_nanosec'] * nsec2sec
        df_magic.drop(['time_sec', 'time_nanosec'], axis=1, inplace=True)

        coincidence_id = '1' + str(tel_id)   # Combination of the telescope IDs used for the coincidence

        df_feature = pd.DataFrame({
            'coincidence_id': [int(coincidence_id)],
            'unix_time': [df_lst['timestamp'].mean()],
            'pointing_alt_lst': [df_lst['pointing_alt'].mean()],
            'pointing_alt_magic': [df_magic['pointing_alt'].mean()],
            'pointing_az_lst': [df_lst['pointing_az'].mean()],
            'pointing_az_magic': [df_magic['pointing_az'].mean()],
            'offset_avg_usec': [offset_avg * sec2usec],
            'n_coincidence': [n_events_at_avg],
            'n_events_magic': [n_events_magic],
        })

        df_profile = pd.DataFrame({
            'offset_usec': offset_steps * sec2usec,
            f'n_coincidence_tel{coincidence_id}': n_events_stereo,
            f'n_coincidence_btwn_tel{coincidence_id}': n_events_stereo_btwn,
        })

        event_data = pd.concat([event_data, df_lst, df_magic])
        features = features.append(df_feature)
        profiles = pd.merge(left=profiles, right=df_profile, on='offset_usec')

    if event_data.empty:
        logger.warning('\nNo coincident events are found. Exiting.\n')
        sys.exit()

    event_data.sort_index(inplace=True)
    event_data.drop_duplicates(inplace=True)

    group_mean = event_data.groupby(['obs_id_magic', 'event_id_magic']).mean()

    event_data['obs_id'] = group_mean['obs_id']
    event_data['event_id'] = group_mean['event_id']

    # Exclude the MAGIC events non-coincident with any of the LST-1 events:
    event_data.dropna(subset=['obs_id', 'event_id'], inplace=True)

    event_data['obs_id'] = event_data['obs_id'].astype(int)
    event_data['event_id'] = event_data['event_id'].astype(int)

    event_data.reset_index(inplace=True)
    event_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

    # Sometimes it happens that one MAGIC stereo event is coincident with two different LST-1 events,
    # and at the moment we exclude that kind of events:
    event_data['multiplicity'] = event_data.groupby(['obs_id', 'event_id']).size()
    event_data.query('multiplicity == [2, 3]', inplace=True)

    # Set the event types:
    combo_types = check_tel_combination(event_data)
    event_data = event_data.join(combo_types)

    # Save the data in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    regex = r'(\S+)_LST-1\.(\S+)\.h5'
    file_name = Path(input_file_lst).resolve().name

    if re.fullmatch(regex, file_name):
        parser = re.findall(regex, file_name)[0]
        output_file = f'{output_dir}/{parser[0]}_LST-1_MAGIC.{parser[1]}.h5'
    else:
        raise RuntimeError('Could not parse information from the input file name.')

    event_data.reset_index(inplace=True)
    save_pandas_to_table(event_data, output_file, group_name='/events', table_name='parameters', mode='w')

    save_pandas_to_table(features, output_file, group_name='/coincidence', table_name='features', mode='a')
    save_pandas_to_table(profiles, output_file, group_name='/coincidence', table_name='profile', mode='a')

    tel_descriptions = {
        1: subarray_lst.tel[1],     # LST-1
        2: subarray_magic.tel[2],   # MAGIC-I
        3: subarray_magic.tel[3],   # MAGIC-II
    }

    subarray_lst1_magic = SubarrayDescription('LST1-MAGIC-Array', tel_positions, tel_descriptions)
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
        help='Keep parameters non-common to LST-1 and MAGIC events.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # Check the event coincidence:
    event_coincidence(args.input_file_lst, args.input_dir_magic,
                      args.output_dir, config, args.keep_all_params)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
