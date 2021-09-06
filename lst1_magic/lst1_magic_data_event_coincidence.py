#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import re
import sys
import yaml
import time
import glob
import argparse
import warnings
import numpy as np 
import pandas as pd
from decimal import Decimal
from astropy.time import Time

warnings.simplefilter('ignore')

__all__ = [
    'load_lst_data',
    'load_magic_data', 
    'event_coincidence',
]

def load_lst_data(input_file):

    print(f'\nLoading the LST-1 data file: {input_file}')

    re_parser = re.findall("(\w+)_LST-1.Run(\d+)\.(\d+)\.h5", input_file)[0]
    data_level = re_parser[0]

    data_lst = pd.read_hdf(
        input_file, key=f'{data_level}/event/telescope/parameters/LST_LSTCam'
    )

    print(f'LST-1: {len(data_lst)} events')

    data_lst.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

    # --- change the column names ---
    column_names = {
        'leakage_pixels_width_1': 'pixels_width_1', 
        'leakage_pixels_width_2': 'pixels_width_2', 
        'leakage_intensity_width_1': 'intensity_width_1',
        'leakage_intensity_width_2': 'intensity_width_2',
        'time_gradient': 'slope'
    } 

    data_lst.rename(columns=column_names, inplace=True)

    # --- change the unit from [deg] to [m] ---
    optics_lst = pd.read_hdf(input_file, key='configuration/instrument/telescope/optics')
    foclen_lst = optics_lst['equivalent_focal_length'].values[0]

    data_lst['length'] = foclen_lst * np.tan(np.deg2rad(data_lst['length'].values))
    data_lst['width'] = foclen_lst * np.tan(np.deg2rad(data_lst['width'].values))

    # --- change the unit from [rad] to [deg] ---
    data_lst['phi'] = np.rad2deg(data_lst['phi'].values)
    data_lst['psi'] = np.rad2deg(data_lst['psi'].values)

    return data_lst

def load_magic_data(input_mask):

    data_paths = glob.glob(input_mask)
    data_paths.sort()

    if data_paths == []:
        print('\nError: No MAGIC input files found. Please check your input argument. Exiting.')
        sys.exit()

    data_magic = pd.DataFrame()

    print('\nLoading the MAGIC data files...')

    for path in data_paths:
        print(path)
        df = pd.read_hdf(path, key='events/params')
        data_magic = pd.concat([data_magic, df])

    data_magic.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data_magic.sort_index(inplace=True)

    for tel_id, tel_name in zip([1, 2], ['MAGIC-I', 'MAGIC-II']):
        df_tel = data_magic.query(f'tel_id == {tel_id}')
        print(f'{tel_name}:  {len(df_tel)} events')

    print('\nApplying the multiplicity cut (= 2)...')

    multiplicity = data_magic.groupby(['obs_id', 'event_id']).size()
    data_magic['multiplicity'] = multiplicity
    data_magic = data_magic.query('multiplicity == 2')
    data_magic.drop('multiplicity', axis=1)

    for tel_id, tel_name in zip([1, 2], ['MAGIC-I', 'MAGIC-II']):
        df_tel = data_magic.query(f'tel_id == {tel_id}')
        print(f'{tel_name}:  {len(df_tel)} events')

    # --- change the column names ---
    column_names = {
        'obs_id': 'obs_id_magic',
        'event_id': 'event_id_magic'
    }

    data_magic.reset_index(inplace=True)
    data_magic.rename(columns=column_names, inplace=True)
    data_magic.set_index(['obs_id_magic', 'event_id_magic', 'tel_id'], inplace=True)
    data_magic.sort_index(inplace=True)

    return data_magic

def event_coincidence(data_lst, data_magic, output_file, config_file):

    sec2us = 1e6
    sec2ns = 1e9
    ms2sec = 1e-3
    ns2sec = 1e-9

    accuracy_time = 1e-7
    bins_step = accuracy_time
    decimals = int(np.log10(1/accuracy_time))

    # --- load config file ---
    config = yaml.safe_load(open(config_file, "r"))

    print('\nCoincidence configuration:\n {}'.format(config['coincidence']))

    type_lst_time = config['coincidence']['type_lst_time']
    tel_id_magic = config['coincidence']['tel_id_magic']
    window_width = float(config['coincidence']['window_width'])

    config_bins = config['coincidence']['bins_offset']

    bins_offset = np.arange(
        float(config_bins['start']), float(config_bins['stop']), bins_step
    )

    bins_offset = np.round(bins_offset, decimals)

    # --- extract events ---
    print('\nExtracting the MAGIC-stereo events within the LST-1 data observation time window...')

    obs_day = Time(data_magic['mjd'].values[0], format='mjd', scale='utc')

    time_lst_unix = list(map(str, data_lst[type_lst_time].values))  
    time_lst_unix = np.array(list(map(Decimal, time_lst_unix)))

    time_lst = time_lst_unix - Decimal(str(obs_day.unix))
    time_lst = np.array(list(map(float, time_lst)))

    df_magic = {
        1: data_magic.query('tel_id == 1'),
        2: data_magic.query('tel_id == 2')
    }

    time_magic = df_magic[tel_id_magic]['millisec'].values * ms2sec +\
                 df_magic[tel_id_magic]['nanosec'].values * ns2sec

    time_magic = np.round(time_magic, decimals)

    condition_lo = (time_magic > (time_lst[0] + bins_offset[0] - window_width))
    condition_hi = (time_magic < (time_lst[-1] + bins_offset[-1] + window_width))
    
    condition = (condition_lo & condition_hi)

    if np.sum(condition) == 0:
        print('--> No MAGIC-stereo events within the LST-1 data observation time window. ' \
              'Please check your MAGIC and LST-1 input files. Exiting.')
        sys.exit()

    else:
        n_events_lst = len(time_lst)
        n_events_magic = np.sum(condition)
        print(f'--> {n_events_magic} MAGIC-stereo events are found. Continuing.\n')
    
    df_magic[1] = df_magic[1].iloc[condition]
    df_magic[2] = df_magic[2].iloc[condition]
    time_magic = time_magic[condition]

    # --- check coincidence ---
    print('Checking the coincidence...')

    n_events_stereo = np.zeros(len(bins_offset), dtype=np.int)
    n_events_stereo_btwn = np.zeros(len(bins_offset), dtype=np.int)

    for i_off, offset in enumerate(bins_offset): 

        time_lim_lo = np.round(time_lst + offset - window_width/2, decimals)
        time_lim_hi = np.round(time_lst + offset + window_width/2, decimals)
        
        for i_ev in range(n_events_lst):
            
            condition_lo = ( time_lim_lo[i_ev] <=  time_magic )
            condition_hi = ( time_magic <= time_lim_hi[i_ev] )

            if np.count_nonzero(condition_lo & condition_hi) == 1:
                n_events_stereo[i_off] += int(1)
                
            condition_lo_wo_equal = ( time_lim_lo[i_ev] <  time_magic )

            if np.count_nonzero(condition_lo_wo_equal & condition_hi) == 1:
                n_events_stereo_btwn[i_off] += int(1)

        print(f'time_offset = {offset*sec2us:.01f} [us]  -->  {n_events_stereo[i_off]} events')

    n_events_max = np.max(n_events_stereo)
    index_max = np.where(n_events_stereo == n_events_max)[0][0]
    offset_at_max = bins_offset[index_max]

    offset_lo = np.round(offset_at_max - window_width, decimals)
    offset_hi = np.round(offset_at_max + window_width, decimals)

    condition = (offset_lo <= bins_offset) & (bins_offset <= offset_hi)
    offset_avg = np.average(bins_offset[condition], weights=n_events_stereo[condition])

    n_events_at_avg = n_events_stereo_btwn[bins_offset < offset_avg][-1]
    ratio = n_events_at_avg/n_events_magic

    print(f'\nAveraged offset = {offset_avg*sec2us:.3f} [us]')
    print(f'--> Number of coincidences = {n_events_at_avg}')
    print(f'--> Ratio of the coincidences = {n_events_at_avg}/{n_events_magic} = {ratio*100:.1f}%\n')

    # --- make coincident events list --- 
    indices_magic = []
    indices_lst = []

    print('Making the coincident event list...')

    offset = bins_offset[bins_offset < offset_avg][-1]
    time_lim_lo = np.round(time_lst - window_width/2 + offset, decimals)
    time_lim_hi = np.round(time_lst + window_width/2 + offset, decimals)

    for i_ev in range(n_events_lst):

        condition_lo = ( time_lim_lo[i_ev] <  time_magic )
        condition_hi = ( time_magic <= time_lim_hi[i_ev] )
        
        if np.count_nonzero(condition_lo & condition_hi) == 1:
            
            index_magic = np.where(condition_lo & condition_hi)[0][0]
            indices_magic.append(index_magic)
            indices_lst.append(i_ev)

    df_lst = data_lst.iloc[indices_lst]
    df_lst['offset_avg'] = offset_avg

    obs_ids_lst = df_lst.index.get_level_values('obs_id')
    event_ids_lst = df_lst.index.get_level_values('event_id')

    for tel_id in [1, 2]:
        df_magic[tel_id] = df_magic[tel_id].iloc[indices_magic]
        df_magic[tel_id]['obs_id'] = obs_ids_lst
        df_magic[tel_id]['event_id'] = event_ids_lst
        df_magic[tel_id].reset_index(inplace=True)

        if tel_id == 1:
            df_magic[tel_id]['tel_id'] = np.repeat(config['tel_ids']['MAGIC-I'], len(df_magic[tel_id]))
        
        elif tel_id == 2:
            df_magic[tel_id]['tel_id'] = np.repeat(config['tel_ids']['MAGIC-II'], len(df_magic[tel_id]))

        df_magic[tel_id].set_index(['obs_id', 'event_id', 'tel_id'], inplace=True) 

    data_stereo = pd.concat([df_lst, df_magic[1], df_magic[2]])
    data_stereo.sort_index(inplace=True)

    # --- store the coincident events list ---
    print(f'--> {output_file}')
    data_stereo.to_hdf(output_file, key='events/params')

    return data_stereo

# ============
# === Main ===
# ============

def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser() 

    arg_parser.add_argument(
        '--input-file-lst', '-il', dest='input_file_lst', type=str, 
        help='Path to a LST-1 DL1 file, f.g. dl1_LST-1.Run02923.0000.h5'
    )

    arg_parser.add_argument(
        '--input-dir-magic', '-im', dest='input_dir_magic', type=str, 
        help='Path to an input directory that contains MAGIC DL1 files.'
    )

    arg_parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./dl1_coincidence.h5', 
        help='Path and name of the output file with HDF5 format.'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml', 
        help='Path to the config file with yaml format.'
    )

    args = arg_parser.parse_args()

    # --- load LST data ---
    data_lst = load_lst_data(args.input_file_lst)

    # --- load MAGIC data ---
    input_mask_magic = args.input_dir_magic + '/*.h5'
    data_magic = load_magic_data(input_mask_magic)

    # --- perform event coincidence ---
    data_stereo = event_coincidence(data_lst, data_magic, args.output_file, args.config_file)

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')

if __name__ == '__main__':
    main()
