#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

Find the coincident events from LST-1 and MAGIC joint observation data by offline with their timestamps.
The coincidence is checked for each telescope combination, i.e., (LST1 + M1) and (LST1 + M2), 
and the events finally containing more than two telescopes information are saved to an output file.

Usage:
$ python lst1_magic_event_coincidence.py 
--input-file-lst "./data/dl1/LST-1/dl1_LST-1.Run02923.0040.h5"
--input-file-magic "./data/dl1/MAGIC/dl1_magic_run05093174_to_05093175_merged.h5"
--output-file "./data/dl1_coincidence/dl1_lst1_magic_run02923.0040.h5"
--config-file "./config.yaml"
"""

import sys
import h5py
import time
import yaml
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
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

tel_positions = {
    1: u.Quantity([-8.09, 77.13, 0.78], u.m),   # LST-1
    2: u.Quantity([39.3, -62.55, -0.97], u.m),  # MAGIC-I
    3: u.Quantity([-31.21, -14.57, 0.2], u.m)   # MAGIC-II
}

__all__ = ['event_coincidence']


def load_lst_data(input_file, type_lst_time):

    logger.info(f'\nLoading the LST-1 data file:\n{input_file}')

    with h5py.File(input_file, 'r') as f:
        data_level = 'dl2' if ('dl2' in f.keys()) else 'dl1'

    data = pd.read_hdf(input_file, key=f'{data_level}/event/telescope/parameters/LST_LSTCam')
    logger.info(f'LST-1: {len(data)} events')

    # --- check the duplication of event IDs ---
    event_ids, counts = np.unique(data['event_id'].values, return_counts=True)

    if np.any(counts > 1):

        event_ids_dup = event_ids[counts > 1].tolist()
        logger.info(f'\nExclude the following events due to the duplication of event IDs: {event_ids_dup}')

        data.query(f'event_id != {event_ids_dup}', inplace=True)
        logger.info(f'--> LST-1: {len(data)} events')

    # --- change the column names ---
    params_rename = {
        'obs_id': 'obs_id_lst',
        'event_id': 'event_id_lst',
        'leakage_pixels_width_1': 'pixels_width_1',
        'leakage_pixels_width_2': 'pixels_width_2',
        'leakage_intensity_width_1': 'intensity_width_1',
        'leakage_intensity_width_2': 'intensity_width_2',
        'time_gradient': 'slope'
    } 

    data.rename(columns=params_rename, inplace=True)
    data.set_index(['obs_id_lst', 'event_id_lst', 'tel_id'], inplace=True)

    # --- remove the unnecessary columns ---
    params_drop = [
        'log_intensity', 'mc_type', 'tel_pos_x', 'tel_pos_y', 'tel_pos_z',
        'calibration_id', 'trigger_type', 'ucts_trigger_type', 'mc_core_distance',
        'concentration_cog', 'concentration_core', 'concentration_pixel', 'wl', 'event_type'
    ]

    for param in params_drop:
        if param in data.columns:
            data.drop(param, axis=1, inplace=True)

    # --- define the timestamps ---
    params_time = np.array(['dragon_time', 'tib_time', 'ucts_time', 'trigger_time'])

    data.drop(params_time[params_time != type_lst_time], axis=1, inplace=True)
    data.rename(columns={type_lst_time: 'timestamp'}, inplace=True)

    # --- drop the non-reconstructed events ---
    logger.info('\nDropping the non-reconstructed events...')

    data.dropna(subset=['intensity', 'slope', 'alt_tel', 'az_tel'], inplace=True)
    logger.info(f'--> LST-1: {len(data)} events')

    # --- change the unit from degree to meter ---
    optics = pd.read_hdf(input_file, key='configuration/instrument/telescope/optics')
    foclen = optics['equivalent_focal_length'].values[0]

    data['length'] = foclen * np.tan(np.deg2rad(data['length']))
    data['width'] = foclen * np.tan(np.deg2rad(data['width']))

    # --- change the unit from radian to degree ---
    data['alt_tel'] = np.rad2deg(data['alt_tel'])
    data['az_tel'] = np.rad2deg(data['az_tel'])
    data['phi'] = np.rad2deg(data['phi'])
    data['psi'] = np.rad2deg(data['psi'])

    # --- load the subarray description ---
    subarray = SubarrayDescription.from_hdf(input_file)

    return data, subarray


def load_magic_data(input_file):

    logger.info(f'\nLoading the MAGIC data file:\n{input_file}')

    data = pd.read_hdf(input_file, key='events/params')
    data['tel_id'] += 1   # M1: tel_id -> 2,  M2: tel_id -> 3
    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data.sort_index(inplace=True)

    telescope_ids = np.unique(data.index.get_level_values('tel_id'))

    for tel_id in telescope_ids:

        tel_name = 'M1' if (tel_id == 2) else 'M2'
        n_events = len(data.query(f'tel_id == {tel_id}'))

        logger.info(f'{tel_name}: {n_events} events')

    subarray = SubarrayDescription.from_hdf(input_file)

    return data, subarray


def event_coincidence(input_file_lst, input_file_magic, output_file, config):

    accuracy_time = 1e-7  # unit: [sec]
    precision = int(-np.log10(accuracy_time))

    config_evco = config['event_coincidence']
    logger.info(f'\nConfiguration for the event coincidence:\n{config_evco}')

    # --- load the input data files ---
    data_lst, subarray_lst = load_lst_data(input_file_lst, config_evco['type_lst_time'])
    data_magic, subarray_magic = load_magic_data(input_file_magic)

    tel_descriptions = {
        1: subarray_lst.tels[1],     # LST-1
        2: subarray_magic.tels[1],   # MAGIC-I
        3: subarray_magic.tels[2]    # MAGIC-II
    }

    subarray = SubarrayDescription('LST-MAGIC-Array', tel_positions, tel_descriptions)
    
    # --- arrange the LST-1 timestamps ---
    time_lst_unix = np.array([Decimal(str(time)) for time in data_lst['timestamp'].values])

    obs_day = np.round(Time(time_lst_unix[0], format='unix', scale='utc').mjd)
    obs_day = np.round(Time(obs_day, format='mjd', scale='utc').unix)

    time_lst = time_lst_unix - Decimal(str(obs_day))
    time_lst = np.round(time_lst.astype(float), decimals=precision)

    # --- check the event coincidence ---
    window_width = config_evco['window_width']

    bins_offset = np.arange(config_evco['offset_start'], config_evco['offset_stop'], step=accuracy_time)
    bins_offset = np.round(bins_offset, decimals=precision)

    df_events = pd.DataFrame()
    df_features = pd.DataFrame()
    df_profile = pd.DataFrame(data={'offset_usec': bins_offset * sec2usec})

    telescope_ids = np.unique(data_magic.index.get_level_values('tel_id'))

    for tel_id in telescope_ids:
        
        tel_name = 'M1' if (tel_id == 2) else 'M2'
        df_magic = data_magic.query(f'tel_id == {tel_id}')

        time_sec = np.round(df_magic['time_sec'].values - obs_day)
        time_nanosec = np.round(df_magic['time_nanosec'].values * nsec2sec, decimals=precision)

        time_magic = np.round(time_sec + time_nanosec, decimals=precision)

        logger.info(f'\nExtracting the {tel_name} events within the LST-1 observation time window...')

        condition_lo = (time_magic > (time_lst[0] + bins_offset[0] - window_width))
        condition_hi = (time_magic < (time_lst[-1] + bins_offset[-1] + window_width))

        condition = (condition_lo & condition_hi)

        if np.sum(condition) == 0:
            logger.info(f'--> No {tel_name} events are found within the LST-1 observation time window. Skipping.')
            continue

        n_events_magic = np.sum(condition)
        logger.info(f'--> {n_events_magic} events are found. Checking the event coincidence...\n')
        
        df_magic = df_magic.iloc[condition]
        time_magic = time_magic[condition]

        n_events_lst = len(time_lst)

        n_events_stereo = np.zeros(len(bins_offset), dtype=np.int)
        n_events_stereo_btwn = np.zeros(len(bins_offset), dtype=np.int)

        for i_off, offset in enumerate(bins_offset):

            time_lim_lo = np.round(time_lst + offset - window_width/2, decimals=precision)
            time_lim_hi = np.round(time_lst + offset + window_width/2, decimals=precision)

            for i_ev in range(n_events_lst):

                condition_lo = ( time_lim_lo[i_ev] <= time_magic )
                condition_hi = ( time_magic <= time_lim_hi[i_ev] )

                if np.count_nonzero(condition_lo & condition_hi) == 1:
                    n_events_stereo[i_off] += int(1)

                condition_lo_wo_equal = ( time_lim_lo[i_ev] < time_magic )

                if np.count_nonzero(condition_lo_wo_equal & condition_hi) == 1:
                    n_events_stereo_btwn[i_off] += int(1)

            logger.info(f'time offset: {offset * sec2usec:.1f} [us]  -->  {n_events_stereo[i_off]} events')

        n_events_max = np.max(n_events_stereo)
        index_at_max = np.where(n_events_stereo == n_events_max)[0][0]
        offset_at_max = bins_offset[index_at_max]

        offset_lo = np.round(offset_at_max - window_width, decimals=precision)
        offset_hi = np.round(offset_at_max + window_width, decimals=precision)

        condition = (offset_lo <= bins_offset) & (bins_offset <= offset_hi)
        offset_avg = np.average(bins_offset[condition], weights=n_events_stereo[condition])

        n_events_at_avg = n_events_stereo_btwn[bins_offset < offset_avg][-1]

        logger.info(f'\nAveraged offset: {offset_avg * sec2usec:.3f} [us]')
        logger.info(f'--> Number of coincidences: {n_events_at_avg}')
        logger.info(f'--> Ratio over {tel_name} events: {n_events_at_avg}/{n_events_magic} ' \
                    f'= {n_events_at_avg/n_events_magic * 100:.1f}%')

        # --- check the event coincidence with the averaged offset ---
        indices_magic = []
        indices_lst = []

        offset = bins_offset[bins_offset < offset_avg][-1]
        
        time_lim_lo = np.round(time_lst - window_width/2 + offset, decimals=precision)
        time_lim_hi = np.round(time_lst + window_width/2 + offset, decimals=precision)

        for i_ev in range(n_events_lst):

            condition_lo = ( time_lim_lo[i_ev] < time_magic )
            condition_hi = ( time_magic <= time_lim_hi[i_ev] )

            if np.count_nonzero(condition_lo & condition_hi) == 1:

                index_magic = np.where(condition_lo & condition_hi)[0][0]
                indices_magic.append(index_magic)
                indices_lst.append(i_ev)
                
        # --- arrange the data frames ---
        df_lst = data_lst.iloc[indices_lst]
        df_lst['obs_id'] = df_magic.iloc[indices_magic].index.get_level_values('obs_id')
        df_lst['event_id'] = df_magic.iloc[indices_magic].index.get_level_values('event_id')
        df_lst.reset_index(inplace=True)
        df_lst.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

        df_magic['timestamp'] = df_magic['time_sec'] + df_magic['time_nanosec'] * nsec2sec
        df_magic.drop(['time_sec', 'time_nanosec'], axis=1, inplace=True)

        features_per_combo = pd.DataFrame(
            data={'tel_combo': [f'lst1_{tel_name.lower()}'],
                  'mean_time_unix': [df_lst['timestamp'].mean()],
                  'mean_alt_lst': [df_lst['alt_tel'].mean()],
                  'mean_alt_magic': [df_magic['alt_tel'].mean()],
                  'mean_az_lst': [df_lst['az_tel'].mean()],
                  'mean_az_magic': [df_magic['az_tel'].mean()],
                  'offset_avg_usec': [offset_avg * sec2usec],
                  'n_coincidence': [n_events_at_avg], 
                  'n_magic': [n_events_magic]}
        )

        profile_per_combo = pd.DataFrame(
            data={'offset_usec': bins_offset * sec2usec,
                  f'n_coincidence_lst1_{tel_name.lower()}': n_events_stereo,
                  f'n_coincidence_btwn_lst1_{tel_name.lower()}': n_events_stereo_btwn}
        )

        df_events = pd.concat([df_events, df_lst, df_magic])
        df_features = df_features.append(features_per_combo)
        df_profile = pd.merge(left=df_profile, right=profile_per_combo, on='offset_usec')

    if df_events.empty:
        logger.info('\nNo coincident events are found. Exiting.\n')
        sys.exit()

    # --- check the number of coincident events ---
    df_events.sort_index(inplace=True)
    df_events.drop_duplicates(inplace=True)

    df_events['multiplicity'] = df_events.groupby(['obs_id', 'event_id']).size()
    df_events = df_events.query('multiplicity == [2, 3]')

    tel_combinations = {
        'm1_m2': [2, 3],
        'lst1_m1': [1, 2],
        'lst1_m2': [1, 3],
        'lst1_m1_m2': [1, 2, 3]
    }

    n_events_total = len(df_events.groupby(['obs_id', 'event_id']).size()) 
    logger.info(f'\nIn total {n_events_total} stereo events are found:') 

    for tel_combo, tel_ids in tel_combinations.items():
        
        df = df_events.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        n_events = np.sum(df.groupby(['obs_id', 'event_id']).size().values == len(tel_ids))
        logger.info(f'{tel_combo}: {n_events} events ({n_events / n_events_total * 100:.1f}%)')

    # --- save in the output file ---
    df_events.to_hdf(output_file, key='events/params', mode='w') 
    df_features.to_hdf(output_file, key='coincidence/features', mode='a')
    df_profile.to_hdf(output_file, key='coincidence/profile', mode='a')

    subarray.to_hdf(output_file)
    
    logger.info(f'\nOutput data file: {output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file-lst', '-l', dest='input_file_lst', type=str,
        help='Path to an input LST-1 DL1 or DL2 data file.'
    )

    parser.add_argument(
        '--input-file-magic', '-m', dest='input_file_magic', type=str,
        help='Path to an input MAGIC DL1 or DL2 data file.'
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./dl1_coincidence.h5',
        help='Path to an output DL1-coincidence data file.'
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.'
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    event_coincidence(args.input_file_lst, args.input_file_magic, args.output_file, config)
    
    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
