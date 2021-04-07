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

start_time = time.time()

sec2us = 1e6
sec2ns = 1e9
ms2sec = 1e-3
ns2sec = 1e-9

# ========================
# === Get the argument ===
# ========================

arg_parser = argparse.ArgumentParser() 

arg_parser.add_argument('--input-file-lst', '-il', dest='input_file_lst', type=str, help='Path to the LST-1 input file')
arg_parser.add_argument('--input-dir-magic', '-im', dest='input_dir_magic', type=str, help='Path to the MAGIC input directory')
arg_parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./coincidence.h5', help='Path to the output file')
arg_parser.add_argument('--config-file', '-c', dest='config_file', type=str, default='./config.yaml', help='Path to the config file')

args = arg_parser.parse_args()

# ===========================
# === Load the MAGIC data ===
# ===========================

data_mask = args.input_dir_magic + '/*.h5'
data_paths_magic = glob.glob(data_mask)
data_paths_magic.sort()

if data_paths_magic == []:
    print('\nError: Failed to find the MAGIC input files. Please check your input directory. Exiting.')
    sys.exit()

data_magic = pd.DataFrame()

print('\nLoading the MAGIC data files...')

for path in data_paths_magic:
    print(path)
    df = pd.read_hdf(path, key='events/params')
    data_magic = pd.concat([data_magic, df])

col_renames = {
    'obs_id': 'obs_id_magic',
    'event_id': 'event_id_magic'
}

data_magic.reset_index(inplace=True)
data_magic.rename(columns=col_renames, inplace=True)

data_magic.set_index(['obs_id_magic', 'event_id_magic', 'tel_id'], inplace=True)
data_magic.sort_index(inplace=True)

for tel_id, tel_name in zip([1, 2], ['MAGIC-I', 'MAGIC-II']):
    df = data_magic.query(f'tel_id == {tel_id}')
    print(f'{tel_name}:  {len(df)} events')

print('\nApplying the mutiplicity cut (= 2)...')

multiplicity = data_magic.groupby(['obs_id_magic', 'event_id_magic']).size()
data_magic['multiplicity'] = multiplicity
data_magic = data_magic.query('multiplicity == 2')

for tel_id, tel_name in zip([1, 2], ['MAGIC-I', 'MAGIC-II']):
    df = data_magic.query(f'tel_id == {tel_id}')
    print(f'{tel_name}:  {len(df)} events')

# ===========================
# === Load the LST-1 data ===
# ===========================

print(f'\nLoading the LST-1 data file: {args.input_file_lst}')

re_parser = re.findall("(\w+)_LST-1.Run(\d+)\.(\d+)\.h5", args.input_file_lst)[0]
data_level = re_parser[0]

data_lst = pd.read_hdf(args.input_file_lst, key=f'{data_level}/event/telescope/parameters/LST_LSTCam')
n_events_lst = len(data_lst)
print(f'LST-1: {n_events_lst} events')

data_lst.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

col_renames = {
    'leakage_pixels_width_1': 'pixels_width_1', 
    'leakage_pixels_width_2': 'pixels_width_2', 
    'leakage_intensity_width_1': 'intensity_width_1',
    'leakage_intensity_width_2': 'intensity_width_2',
    'time_gradient': 'slope'
} 

data_lst.rename(columns=col_renames, inplace=True)

foclen_lst = 28  # unit: [m]
data_lst['length'] = foclen_lst * np.tan(np.deg2rad(data_lst['length'].values))
data_lst['width'] = foclen_lst * np.tan(np.deg2rad(data_lst['width'].values))

# ===================================
# === Check the event coincidence ===
# ===================================

print('\nCoincidence configuration:\n {}'.format(config['coincidence']))

config = yaml.safe_load(open(args.config_file, "r"))

type_timestamp = config['coincidence']['timestamp_lst']
window = float(config['coincidence']['window'])

config_bins = config['coincidence']['bins_offset']
offset_bins = np.round(np.arange(float(config_bins['start']), float(config_bins['stop']), float(config_bins['step'])), 7)

print('\nExtracting the MAGIC-stereo events within the LST-1 data observation time window...')

obs_day = Time(data_magic['mjd'].values[0], format='mjd', scale='utc')
time_lst_tmp = np.array(list(map(Decimal, list(map(str, data_lst[type_timestamp].values)))))
time_lst = np.array(list(map(float, time_lst_tmp - Decimal(str(obs_day.unix)))))

df_magic = {
    1: data_magic.query('tel_id == 1'),
    2: data_magic.query('tel_id == 2')
}

time_magic_tmp = np.round(df_magic[1]['millisec'].values * ms2sec + df_magic[1]['nanosec'].values * ns2sec, 7)

condition = (time_magic_tmp > (time_lst[0] + offset_bins[0] - window)) & (time_magic_tmp < (time_lst[-1] + offset_bins[-1] + window))

if np.sum(condition) == 0:
    print('--> No MAGIC-stereo events within the LST-1 data observation time window. Please check your MAGIC and LST-1 input files. Exiting.')
    sys.exit()
else:
    print(f'--> {np.sum(condition)} MAGIC-stereo events are found. Continuing.\n')
    n_events_magic = np.sum(condition)
    df_magic[1] = df_magic[1].iloc[condition]
    df_magic[2] = df_magic[2].iloc[condition]

time_magic = time_magic_tmp[condition]

print('Checking the coincidence...')

nums_stereo = np.zeros(len(offset_bins), dtype=np.int)

for i_off, offset in enumerate(offset_bins): 
    
    time_lim_lo = np.round(time_lst + offset - window/2, 7)
    time_lim_hi = np.round(time_lst + offset + window/2, 7)
    
    for i_ev in range(n_events_lst):
        
        condition_lo = ( time_lim_lo[i_ev] <  time_magic )
        condition_hi = ( time_magic <= time_lim_hi[i_ev] )

        if np.count_nonzero(condition_lo & condition_hi) == 1:
            nums_stereo[i_off] += int(1)
            
    print(f'time_offset = {offset*sec2us:.01f} [us]  -->  {nums_stereo[i_off]} events')

index_max = np.where(nums_stereo == np.max(nums_stereo))[0]
index_max = np.append(index_max, index_max[-1]+1)
offset_max = np.average(offset_bins[index_max])

indices = np.where((offset_bins >= offset_max - window) & (offset_bins <= offset_max + window))
offset_avg = np.average(offset_bins[indices], weights=nums_stereo[indices])
num_avg = nums_stereo[offset_bins < offset_avg][-1]

print(f'\nAveraged time-offset = {offset_avg*sec2us:.2f} [µs]')
print(f'--> Number of coincidence = {num_avg}')
print(f'--> Fraction of the coincidence = {num_avg}/{n_events_magic} = {num_avg/n_events_magic*100:.1f}%\n')

# ======================================
# === Make the coincident event list ===
# ======================================

indices_magic = []
indices_lst = []
obs_ids_lst = []
event_ids_lst = []

print('Making the coincident event list...')

offset = offset_bins[offset_bins < offset_avg][-1]
time_lim_lo = np.round(time_lst - window/2 + offset, 7)
time_lim_hi = np.round(time_lst + window/2 + offset, 7)

for i_ev in range(n_events_lst):

    condition_lo = ( time_lim_lo[i_ev] <  time_magic )
    condition_hi = ( time_magic <= time_lim_hi[i_ev] )
    
    if np.count_nonzero(condition_lo & condition_hi) == 1:
        
        index_magic = np.where(condition_lo & condition_hi)[0][0]
        indices_magic.append(index_magic)
        indices_lst.append(i_ev)
        obs_ids_lst.append(data_lst.iloc[i_ev].name[0])
        event_ids_lst.append(data_lst.iloc[i_ev].name[1])

df_lst = data_lst.iloc[indices_lst]

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
data_stereo['offset_avg'] = offset_avg

for tel_name in config['tel_ids']:
    df = data_stereo.query('tel_id == {}'.format(config['tel_ids'][tel_name]))
    print(f'{tel_name}: {len(df)} events')

data_stereo.to_hdf(args.output_file, key='events/params')

end_time = time.time()
print(f'\nDone. elapsed_time = {end_time - start_time:.0f} [sec]')