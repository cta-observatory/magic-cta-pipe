#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import re
import sys
import glob
import yaml
import time
import argparse
import warnings
import numpy as np 
import pandas as pd
from decimal import Decimal
from astropy.time import Time

warnings.simplefilter('ignore')

__all__ = ['event_coincidence']


def load_lst_data(data_path):

    print(f'\nLoading the LST-1 data file: {data_path}')

    re_parser = re.findall('(\w+)_LST-1.Run(\d+)\.(\d+)\.h5', data_path)[0]
    data_level = re_parser[0]

    data_lst = pd.read_hdf(
        data_path, key=f'{data_level}/event/telescope/parameters/LST_LSTCam'
    )
    
    print(f'LST-1: {len(data_lst)} events')

    # --- change the column names ---
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

    # --- remove unnecessary columns ---
    column_names = [
        'log_intensity', 'n_pixels', 'concentration_cog', 'concentration_core', 
        'concentration_pixel', 'mc_type', 'mc_core_distance', 
        'tel_pos_x', 'tel_pos_y', 'tel_pos_z', 
    ]

    data_lst = data_lst.drop(column_names, axis=1)

    # --- change the unit from [deg] to [m] ---
    optics = pd.read_hdf(data_path, key='configuration/instrument/telescope/optics')
    foclen = optics['equivalent_focal_length'].values[0]

    data_lst['length'] = foclen * np.tan(np.deg2rad(data_lst['length'].values))
    data_lst['width'] = foclen * np.tan(np.deg2rad(data_lst['width'].values))

    # --- change the unit from [rad] to [deg] ---
    data_lst['phi'] = np.rad2deg(data_lst['phi'].values)
    data_lst['psi'] = np.rad2deg(data_lst['psi'].values)

    # --- set the index ---
    data_lst.set_index(['obs_id_lst', 'event_id_lst', 'tel_id'], inplace=True)

    return data_lst


def load_magic_data(data_path):

    print('\nLoading the following MAGIC data files:')

    data_magic = pd.DataFrame()

    paths_list = glob.glob(data_path)
    paths_list.sort()

    for path in paths_list:

        print(path)

        df = pd.read_hdf(path, key='events/params')
        data_magic = pd.concat([data_magic, df])

    data_magic.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data_magic.sort_index(inplace=True)

    for tel_id, tel_name in zip([1, 2], ['MAGIC-I', 'MAGIC-II']):
        n_events = len(data_magic.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}:  {n_events} events')

    # --- apply multiplicity cut ---
    print('\nApplying the multiplicity cut (= 2)...')

    multiplicity = data_magic.groupby(['obs_id', 'event_id']).size()

    data_magic['multiplicity'] = multiplicity
    data_magic = data_magic.query('multiplicity == 2')
    data_magic = data_magic.drop('multiplicity', axis=1)

    for tel_id, tel_name in zip([1, 2], ['MAGIC-I', 'MAGIC-II']):
        n_events = len(data_magic.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}:  {n_events} events')

    return data_magic


def event_coincidence(data_path_lst, data_path_magic, config):

    sec2us = 1e6
    ms2sec = 1e-3
    ns2sec = 1e-9

    accuracy_time = 1e-7
    decimals = int(np.log10(1/accuracy_time))

    print(f'\nConfiguration for the event coincidence:\n{config}')

    # --- load the input LST-1 data ---
    data_lst = load_lst_data(data_path_lst)

    # --- load the input MAGIC data ---
    data_magic = load_magic_data(data_path_magic)
    
    # --- get the LST-1 timestamps ---
    mjd = data_magic['mjd'].values[0]
    obs_day = Time(mjd, format='mjd', scale='utc')

    type_lst_time = config['type_lst_time']

    time_lst_unix = list(map(str, data_lst[type_lst_time].values))  
    time_lst_unix = np.array(list(map(Decimal, time_lst_unix)))

    time_lst = time_lst_unix - Decimal(str(obs_day.unix))
    time_lst = np.array(list(map(float, time_lst)))

    # --- get the MAGIC timestamps ---
    df_magic = {
        1: data_magic.query('tel_id == 1'), 
        2: data_magic.query('tel_id == 2'), 
    }

    if config['type_magic_time'] == 'MAGIC-I':
        time_magic = df_magic[1]['millisec'].values * ms2sec + df_magic[1]['nanosec'].values * ns2sec

    elif config['type_magic_time'] == 'MAGIC-II':
        time_magic = df_magic[2]['millisec'].values * ms2sec + df_magic[2]['nanosec'].values * ns2sec

    time_magic = np.round(time_magic, decimals)

    # --- extract events --- 
    print('\nExtracting the MAGIC-stereo events within the LST-1 data observation time window...')

    window_width = config['window_width']

    bins_offset = np.arange(
        start=config['offset_start'], stop=config['offset_stop'], step=accuracy_time
    )

    bins_offset = np.round(bins_offset, decimals)

    condition_lo = (time_magic > (time_lst[0] + bins_offset[0] - window_width))
    condition_hi = (time_magic < (time_lst[-1] + bins_offset[-1] + window_width))
    
    condition = (condition_lo & condition_hi)

    if np.sum(condition) == 0:
        print('--> No MAGIC-stereo events are found within the LST-1 data observation time window. ' \
              'Please check your MAGIC and LST-1 input data. Exiting.')
        sys.exit()

    else:
        n_events_magic = np.sum(condition)
        print(f'--> {n_events_magic} MAGIC-stereo events are found. Continuing.\n')
    
    df_magic[1] = df_magic[1].iloc[condition]
    df_magic[2] = df_magic[2].iloc[condition]

    time_magic = time_magic[condition]

    # --- check the event coincidence ---
    print('Checking the event coincidence...')

    n_events_lst = len(time_lst)

    n_events_stereo = np.zeros(len(bins_offset), dtype=np.int)
    n_events_stereo_btwn = np.zeros(len(bins_offset), dtype=np.int)

    for i_off, offset in enumerate(bins_offset): 

        time_lim_lo = np.round(time_lst + offset - window_width/2, decimals)
        time_lim_hi = np.round(time_lst + offset + window_width/2, decimals)
        
        for i_ev in range(n_events_lst):
            
            condition_lo = ( time_lim_lo[i_ev] <= time_magic )
            condition_hi = ( time_magic <= time_lim_hi[i_ev] )

            if np.count_nonzero(condition_lo & condition_hi) == 1:
                n_events_stereo[i_off] += int(1)
                
            condition_lo_wo_equal = ( time_lim_lo[i_ev] < time_magic )

            if np.count_nonzero(condition_lo_wo_equal & condition_hi) == 1:
                n_events_stereo_btwn[i_off] += int(1)

        print(f'time_offset = {offset*sec2us:.01f} [us]  -->  {n_events_stereo[i_off]} events')

    n_events_max = np.max(n_events_stereo)
    index_at_max = np.where(n_events_stereo == n_events_max)[0][0]
    offset_at_max = bins_offset[index_at_max]

    offset_lo = np.round(offset_at_max - window_width, decimals)
    offset_hi = np.round(offset_at_max + window_width, decimals)

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
    time_lim_lo = np.round(time_lst - window_width/2 + offset, decimals)
    time_lim_hi = np.round(time_lst + window_width/2 + offset, decimals)

    for i_ev in range(n_events_lst):

        condition_lo = ( time_lim_lo[i_ev] < time_magic )
        condition_hi = ( time_magic <= time_lim_hi[i_ev] )
        
        if np.count_nonzero(condition_lo & condition_hi) == 1:
            
            index_magic = np.where(condition_lo & condition_hi)[0][0]
            indices_magic.append(index_magic)
            indices_lst.append(i_ev)

    obs_ids_lst = data_lst.iloc[indices_lst].index.get_level_values('obs_id_lst')
    event_ids_lst = data_lst.iloc[indices_lst].index.get_level_values('event_id_lst')

    obs_ids_magic = df_magic[1].iloc[indices_magic].index.get_level_values('obs_id')
    event_ids_magic = df_magic[1].iloc[indices_magic].index.get_level_values('event_id')

    # --- arrange the LST-1 data frame ---
    df_lst = data_lst.iloc[indices_lst]

    df_lst['obs_id'] = obs_ids_magic
    df_lst['event_id'] = event_ids_magic

    df_lst.reset_index(inplace=True)
    df_lst.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

    # --- arange the MAGIC data frame ---
    for tel_id in [1, 2]:
        
        df_magic[tel_id].loc[(obs_ids_magic, event_ids_magic, tel_id), 'obs_id_lst'] = obs_ids_lst
        df_magic[tel_id].loc[(obs_ids_magic, event_ids_magic, tel_id), 'event_id_lst'] = event_ids_lst

        df_magic[tel_id].reset_index(inplace=True)
        df_magic[tel_id]['tel_id'] = tel_id + 1   # MAGIC-I -> 2, MAGIC-II -> 3
        df_magic[tel_id].set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

    # --- make a coincident events list ---
    if config['keep_magic_stereo']:
        print('\nKeeping the non-coincident MAGIC-stereo events...')

    else:
        print('\nDiscarding the non-coincident MAGIC-stereo events...')
        df_magic[1] = df_magic[1].iloc[indices_magic]
        df_magic[2] = df_magic[2].iloc[indices_magic]

    data_stereo = pd.concat([df_lst, df_magic[1], df_magic[2]])
    data_stereo.sort_index(inplace=True)

    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):
        n_events = len(data_stereo.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}:  {n_events} events')

    data_stereo['offset_us'] = offset_avg * sec2us

    return data_stereo


def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser() 

    arg_parser.add_argument(
        '--input-data-lst', '-l', dest='input_data_lst', type=str, 
        help='Path to an input LST-1 DL1 data file, e.g., dl1_LST-1.Run02923.0000.h5'
    )

    arg_parser.add_argument(
        '--input-data-magic', '-m', dest='input_data_magic', type=str, 
        help='Path to input MAGIC DL1 data file(s), e.g., dl1_run*.h5'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str,  
        help='Path and name of an output data file with HDF5 format, e.g., dl1_lst1_magic.h5'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, 
        help='Path to a config file with yaml format, e.g., config.yaml'
    )

    args = arg_parser.parse_args()

    # --- perform the event coincidence ---
    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    data_stereo = event_coincidence(
        args.input_data_lst, args.input_data_magic, config_lst1_magic['event_coincidence']
    )

    # --- store the coincident events list ---
    data_stereo.to_hdf(args.output_data, key='events/params')

    print(f'\nOutput data file: {args.output_data}')
    
    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
