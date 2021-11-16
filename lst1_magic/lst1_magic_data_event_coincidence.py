#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import os
import sys
import glob
import yaml
import time
import h5py
import argparse
import warnings
import numpy as np 
import pandas as pd
from pathlib import Path
from decimal import Decimal
from astropy.time import Time

warnings.simplefilter('ignore')

__all__ = ['event_coincidence']


def load_lst_data(input_data, type_lst_time):

    print(f'\nLoading the LST-1 data file: {input_data}')

    with h5py.File(input_data, 'r') as f:
        keys = f.keys()

    data_level = 'dl2' if ('dl2' in keys) else 'dl1'

    data_lst = pd.read_hdf(input_data, key=f'{data_level}/event/telescope/parameters/LST_LSTCam')

    print(f'LST-1: {len(data_lst)} events')

    # --- check duplication of event IDs ---
    event_ids, counts = np.unique(data_lst['event_id'].values, return_counts=True)

    if np.sum(counts > 1):

        print(f'\nExclude the following events due to the duplication of event IDs: {event_ids[counts > 1]}')
        data_lst.query(f'event_id != {list(event_ids[counts > 1])}', inplace=True)

        print(f'--> LST-1: {len(data_lst)} events')

    # --- change the column names to default ones ---
    column_names = {
        'obs_id': 'obs_id_lst',
        'event_id': 'event_id_lst',
        'leakage_pixels_width_1': 'pixels_width_1', 
        'leakage_pixels_width_2': 'pixels_width_2', 
        'leakage_intensity_width_1': 'intensity_width_1',
        'leakage_intensity_width_2': 'intensity_width_2',
        'time_gradient': 'slope'
    } 

    data_lst.rename(columns=column_names, inplace=True)
    data_lst.set_index(['obs_id_lst', 'event_id_lst', 'tel_id'], inplace=True)

    # --- remove unnecessary columns ---
    columns_list = [
        'log_intensity', 'n_pixels', 'mc_type', 'tel_pos_x', 'tel_pos_y', 'tel_pos_z',
        'calibration_id', 'trigger_type', 'ucts_trigger_type', 'mc_core_distance',
        'concentration_cog', 'concentration_core', 'concentration_pixel', 'wl'
    ]

    for column in columns_list:
        if column in data_lst.columns:
            data_lst.drop(column, axis=1, inplace=True)

    # --- define the timestamps ---
    data_lst.rename(columns={type_lst_time: 'timestamp'}, inplace=True)

    time_names = np.array(['dragon_time', 'tib_time', 'ucts_time', 'trigger_time'])
    data_lst.drop(time_names[time_names != type_lst_time], axis=1, inplace=True)
   
    # --- change the unit from [deg] to [m] ---
    optics = pd.read_hdf(input_data, key='configuration/instrument/telescope/optics')
    foclen = optics['equivalent_focal_length'].values[0]

    data_lst['length'] = foclen * np.tan(np.deg2rad(data_lst['length'].values))
    data_lst['width'] = foclen * np.tan(np.deg2rad(data_lst['width'].values))

    # --- change the unit from [rad] to [deg] ---
    data_lst['phi'] = np.rad2deg(data_lst['phi'].values)
    data_lst['psi'] = np.rad2deg(data_lst['psi'].values)

    return data_lst


def load_magic_data(input_data_mask):

    print('\nLoading the following MAGIC data files:')

    data_magic = pd.DataFrame()

    paths_list = glob.glob(input_data_mask)
    paths_list.sort()

    for path in paths_list:

        print(path)

        df = pd.read_hdf(path, key='events/params')
        df['tel_id'] = df['tel_id'].values + 1  # MAGIC-I -> 2, MAGIC-II -> 3 
        data_magic = pd.concat([data_magic, df])

    data_magic.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data_magic.sort_index(inplace=True)

    data_magic.drop(['slope_err', 'intercept_err', 'deviation'], axis=1, inplace=True)

    for tel_id, tel_name in zip([2, 3], ['MAGIC-I', 'MAGIC-II']):
        n_events = len(data_magic.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}:  {n_events} events')

    return data_magic


def event_coincidence(input_data_lst, input_data_mask_magic, output_data, config):

    sec2us = 1e6
    ms2sec = 1e-3 
    ns2sec = 1e-9  

    accuracy_time = 1e-7  # unit: [sec]
    precision = int(np.log10(1/accuracy_time))

    print(f'\nConfiguration for the event coincidence:\n{config}')

    # --- load the LST-1 input data ---
    data_lst = load_lst_data(input_data_lst, config['type_lst_time'])

    # --- load the MAGIC input data ---
    data_magic = load_magic_data(input_data_mask_magic)
    
    # --- arange the LST-1 timestamps ---
    mjd = data_magic['mjd'].values[0]
    obs_day = Time(mjd, format='mjd', scale='utc')

    time_lst_tmp = data_lst['timestamp'].values.astype(str)
    time_lst_tmp = np.array([Decimal(time) for time in time_lst_tmp]) 

    time_lst = time_lst_tmp - Decimal(str(obs_day.unix))
    time_lst = time_lst.astype(float)

    # --- check the event coincidence ---
    df_events = {}
    df_profile = {}
    df_features = {}

    for tel_id, tel_name in zip([2, 3], ['MAGIC-I', 'MAGIC-II']):
        
        df_magic = data_magic.query(f'tel_id == {tel_id}')

        time_magic = df_magic['millisec'].values * ms2sec + df_magic['nanosec'].values * ns2sec
        time_magic = np.round(time_magic, precision)

        print(f'\nExtracting the {tel_name} events within the LST-1 data observation time window...')

        window_width = config['window_width']

        bins_offset = np.arange(
            start=config['offset_start'], stop=config['offset_stop'], step=accuracy_time
        )

        bins_offset = np.round(bins_offset, precision)

        condition_lo = (time_magic > (time_lst[0] + bins_offset[0] - window_width))
        condition_hi = (time_magic < (time_lst[-1] + bins_offset[-1] + window_width))
        
        condition = (condition_lo & condition_hi)

        if np.sum(condition) == 0:
            print(f'--> No {tel_name} events are found within the LST-1 data observation time window. Exiting.\n')
            sys.exit()

        else:
            n_events_magic = np.sum(condition)
            print(f'--> {n_events_magic} events are found.')
        
        df_magic = df_magic.iloc[condition]
        time_magic = time_magic[condition]

        print('\nChecking the event coincidence...')

        n_events_lst = len(time_lst)

        n_events_stereo = np.zeros(len(bins_offset), dtype=np.int)
        n_events_stereo_btwn = np.zeros(len(bins_offset), dtype=np.int)

        for i_off, offset in enumerate(bins_offset): 

            time_lim_lo = np.round(time_lst + offset - window_width/2, precision)
            time_lim_hi = np.round(time_lst + offset + window_width/2, precision)
            
            for i_ev in range(n_events_lst):
                
                condition_lo = ( time_lim_lo[i_ev] <= time_magic )
                condition_hi = ( time_magic <= time_lim_hi[i_ev] )

                if np.count_nonzero(condition_lo & condition_hi) == 1:
                    n_events_stereo[i_off] += int(1)
                    
                condition_lo_wo_equal = ( time_lim_lo[i_ev] < time_magic )

                if np.count_nonzero(condition_lo_wo_equal & condition_hi) == 1:
                    n_events_stereo_btwn[i_off] += int(1)

            print(f'time_offset = {offset*sec2us:.01f} [us]  -->  {n_events_stereo[i_off]} events')

        if np.all(n_events_stereo == 0):
            print(f'\nNo coicident events of LST-1 and {tel_name} are found. Exiting.\n')
            sys.exit()

        n_events_max = np.max(n_events_stereo)
        index_at_max = np.where(n_events_stereo == n_events_max)[0][0]
        offset_at_max = bins_offset[index_at_max]

        offset_lo = np.round(offset_at_max - window_width, precision)
        offset_hi = np.round(offset_at_max + window_width, precision)

        condition = (offset_lo <= bins_offset) & (bins_offset <= offset_hi)
        offset_avg = np.average(bins_offset[condition], weights=n_events_stereo[condition])

        n_events_at_avg = n_events_stereo_btwn[bins_offset < offset_avg][-1]
        ratio = n_events_at_avg/n_events_magic

        print(f'\nAveraged offset = {offset_avg*sec2us:.3f} [us]')
        print(f'--> Number of coincidences = {n_events_at_avg}')
        print(f'--> Ratio of the coincidences = {n_events_at_avg}/{n_events_magic} = {ratio*100:.1f}%')

        # --- check the event coincidence with the averaged offset --- 
        indices_magic = []
        indices_lst = []

        offset = bins_offset[bins_offset < offset_avg][-1]
        time_lim_lo = np.round(time_lst - window_width/2 + offset, precision)
        time_lim_hi = np.round(time_lst + window_width/2 + offset, precision)

        for i_ev in range(n_events_lst):

            condition_lo = ( time_lim_lo[i_ev] < time_magic )
            condition_hi = ( time_magic <= time_lim_hi[i_ev] )
            
            if np.count_nonzero(condition_lo & condition_hi) == 1:
                
                index_magic = np.where(condition_lo & condition_hi)[0][0]
                indices_magic.append(index_magic)
                indices_lst.append(i_ev)

        obs_ids_lst = data_lst.iloc[indices_lst].index.get_level_values('obs_id_lst')
        event_ids_lst = data_lst.iloc[indices_lst].index.get_level_values('event_id_lst')

        obs_ids_magic = df_magic.iloc[indices_magic].index.get_level_values('obs_id')
        event_ids_magic = df_magic.iloc[indices_magic].index.get_level_values('event_id')

        # --- arrange data frames ---
        df_lst = data_lst.iloc[indices_lst]

        df_lst['obs_id'] = obs_ids_magic
        df_lst['event_id'] = event_ids_magic

        df_lst.reset_index(inplace=True)
        df_lst.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

        df_magic.loc[(obs_ids_magic, event_ids_magic, tel_id), 'obs_id_lst'] = obs_ids_lst
        df_magic.loc[(obs_ids_magic, event_ids_magic, tel_id), 'event_id_lst'] = event_ids_lst

        df_magic['timestamp'] = Time(df_magic['mjd'].values, format='mjd').unix + \
                    df_magic['millisec'].values * ms2sec + df_magic['nanosec'].values * ns2sec

        df_magic.drop(['mjd', 'millisec', 'nanosec'], axis=1, inplace=True)

        df_events[tel_name] = pd.concat([df_lst, df_magic])
        df_events[tel_name].sort_index(inplace=True)

        container_profile = {
            'offset_us': bins_offset * sec2us,
            f'n_coincidence_m{tel_id}': n_events_stereo,
            f'n_coincidence_btwn_m{tel_id}': n_events_stereo_btwn
        }

        df_profile[tel_name] = pd.DataFrame(container_profile)

        mean_time_unix = np.mean(df_events[tel_name]['timestamp'].values)
        mean_alt_lst = np.mean(np.rad2deg(df_lst['alt_tel'].values))
        mean_alt_magic = np.mean(np.rad2deg(df_magic['alt_tel'].values))

        container_features = {
            'mean_time_unix': [mean_time_unix],
            'mean_alt_lst': [mean_alt_lst],
            'mean_alt_magic': [mean_alt_magic],
            'n_magic': [n_events_magic],
            'n_coincidence': [n_events_at_avg], 
            'ratio': [ratio], 
            'offset_avg_us': [offset_avg * sec2us]
        }

        df_features[tel_name] = pd.DataFrame(container_features, index=[tel_name])

    # --- check the number of coincident events ---
    data_stereo = pd.concat([df_events['MAGIC-I'], df_events['MAGIC-II']])
    data_stereo.sort_index(inplace=True)
    data_stereo.drop_duplicates(inplace=True)

    data_stereo['multiplicity'] = data_stereo.groupby(['obs_id', 'event_id']).size()
    data_stereo = data_stereo.query('multiplicity == [2, 3]')

    print('\nEvents with 2 tels info:')

    for tel_id, tel_name in zip([2, 3], ['MAGIC-I', 'MAGIC-II']):
        
        df = data_stereo.query(f'(tel_id == [1, {tel_id}]) & (multiplicity == 2)')
        n_events = np.sum(df.groupby(['obs_id', 'event_id']).size().values == 2)
        print(f'LST-1 + {tel_name}: {n_events} events')

    df = data_stereo.query(f'(tel_id == [2, 3]) & (multiplicity == 2)')
    n_events = np.sum(df.groupby(['obs_id', 'event_id']).size().values == 2)
    print(f'MAGIC-I + MAGIC-II: {n_events} events')

    print('\nEvents with 3 tels info:')

    n_events = len(data_stereo.query(f'multiplicity == 3'))/3
    print(f'LST-1 + MAGIC-I + MAGIC-II: {n_events:.0f} events')

    n_events = len(data_stereo.groupby(['obs_id', 'event_id']).size()) 
    print(f'\nIn total {n_events} stereo events are found.') 

    # --- save the data frames ---
    output_dir = str(Path(output_data).parent)
    os.makedirs(output_dir, exist_ok=True)

    data_stereo.to_hdf(output_data, key='events/params', mode='w') 

    data_profile = pd.merge(df_profile['MAGIC-I'], df_profile['MAGIC-II'], on='offset_us')
    data_profile.to_hdf(output_data, key='coincidence/profile', mode='a')

    data_features = pd.concat([df_features['MAGIC-I'], df_features['MAGIC-II']])
    data_features.to_hdf(output_data, key='coincidence/features', mode='a')
    
    print(f'\nOutput data: {output_data}')


def main():

    start_time = time.time()

    arg_parser = argparse.ArgumentParser() 

    arg_parser.add_argument(
        '--input-data-lst', '-l', dest='input_data_lst', type=str, 
        help='Path to a LST-1 DL1 or DL2 data file.'
    )

    arg_parser.add_argument(
        '--input-data-magic', '-m', dest='input_data_magic', type=str, 
        help='Path to MAGIC data files with HDF format.'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str,  
        help='Path to an output data file. The output directory will be created if it does not exist.'  
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, 
        help='Path to a configuration file.'
    )

    args = arg_parser.parse_args()

    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    event_coincidence(
        args.input_data_lst, args.input_data_magic, args.output_data, config_lst1_magic['event_coincidence'],
    )
    
    print('\nDone.')
    print(f'\nelapsed time = {time.time() - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
