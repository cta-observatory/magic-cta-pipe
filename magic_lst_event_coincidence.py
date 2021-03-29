#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import re
import sys
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

# ========================
# === Get the argument ===
# ========================

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--input-file-lst', '-il', dest='input_file_lst', type=str, help='Path to the LST-1 input file')
arg_parser.add_argument('--input-dir-magic', '-im', dest='input_dir_magic', type=str, 
                    help='Path to the directory that contains MAGIC input files')
arg_parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./coincidence.h5', help='Path to the output file')

args = arg_parser.parse_args()

# ===========================
# === Load the MAGIC data ===
# ===========================

data_mask = str(args.input_dir_magic) + '/*.h5'
data_paths_magic = glob.glob(data_mask)

if data_paths_magic == []:
    print('\nNot accessible to the input files. Please check the path to the files. Exiting.')
    sys.exit()

data_magic = pd.DataFrame()

print('\nLoading the MAGIC dataset...')

for path in data_paths_magic:
    print(path)
    df = pd.read_hdf(path, key='dl1/hillas_params')
    data_magic = pd.concat([data_magic, df])

data_magic.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
data_magic.sort_index(inplace=True)

for tel_id in [1, 2]:
    df = data_magic.query(f'tel_id == {tel_id}')
    print(f'MAGIC{tel_id}:  {len(df)} events')

print('\nApplying the multiplicity cut...')

multiplicity = data_magic['intensity'].groupby(['obs_id', 'event_id']).size()
data_magic['multiplicity'] = multiplicity
data_magic = data_magic.query('multiplicity == 2')

for tel_id in [1, 2]:
    df = data_magic.query(f'tel_id == {tel_id}')
    print(f'MAGIC{tel_id}:  {len(df)} events')

# remane the columns as to match those of LST-1 DL1 file
param_names = {'tel_alt': 'alt_tel', 
                'tel_az': 'az_tel', 
                'pixels_width_1': 'leakage_pixels_width_1', 
                'pixels_width_2': 'leakage_pixels_width_2', 
                'intensity_width_1': 'leakage_intensity_width_1', 
                'intensity_width_2': 'leakage_intensity_width_2',
                'slope': 'time_gradient'}

data_magic.rename(columns=param_names, inplace=True)

# ===========================
# === Load the LST-1 data ===
# ===========================

print('\nLoading the LST-1 dataset...')

re_parser = re.findall("(\w+)_LST-1.Run(\d+)\.(\d+)\.h5", args.input_file_lst)[0]
data_level = re_parser[0]
run_id_lst = int(re_parser[1])
subrun_id_lst = int(re_parser[2])

print(f'data_level = {data_level}, run_id_lst = {run_id_lst}, subrun_id_lst = {subrun_id_lst}')

data_lst = pd.read_hdf(args.input_file_lst, key=f'{data_level}/event/telescope/parameters/LST_LSTCam')
n_events_lst = len(data_lst)
print(f'LST-1: {n_events_lst} events')

data_lst['run_id_lst'] = np.repeat(run_id_lst, n_events_lst)
data_lst['subrun_id_lst'] = np.repeat(subrun_id_lst, n_events_lst)
data_lst.rename(columns={'event_id': 'event_id_lst'}, inplace=True)
data_lst.set_index(['run_id_lst', 'subrun_id_lst', 'event_id_lst', 'tel_id'], inplace=True)

# ===================================
# === Check the event coincidence ===
# ===================================

window = 6e-7  # optimized coincidence window, unit: [sec]
offset_bins = np.round(np.arange(-5e-6, 0, 1e-7), 7)  # region to scan the time offset, unit: [sec]

sec2us = 1e6
sec2ns = 1e9

print('\nExtracting the MAGIC events within the LST1 data observation time window...')

obs_day = Time(data_magic['mjd'].values[0], format='mjd', scale='utc')
time_lst_tmp = np.array(list(map(Decimal, list(map(str, data_lst['dragon_time'].values)))))  # dragon_time is the most stable timestamp so far 
time_lst = np.array(list(map(float, time_lst_tmp - Decimal(str(obs_day.unix)))))

df_magic = {1: data_magic.query('tel_id == 1'),
            2: data_magic.query('tel_id == 2')}

ms2sec = 1e-3
ns2sec = 1e-9

time_magic_tmp = np.round(df_magic[1]['millisec'].values*ms2sec + df_magic[1]['nanosec'].values*ns2sec, 7)

condition = (time_magic_tmp > time_lst[0] + offset_bins[0] - window) & (time_magic_tmp < time_lst[-1] + offset_bins[-1] + window)

if np.sum(condition) == 0:
    print('--> No MAGIC events within the LST data observation time window. Check your MAGIC and LST input files. Exiting.')
    sys.exit()
else:
    print(f'--> {np.sum(condition)} MAGIC events are found within the LST1 observation window.\n')
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

        if np.count_nonzero(condition_lo&condition_hi) == 1:
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
run_ids_lst = []
subrun_ids_lst = []
event_ids_lst = []

print('Making the coincident event list...')

offset = offset_bins[offset_bins < offset_avg][-1]
time_lim_lo = np.round(time_lst - window/2 + offset, 7)
time_lim_hi = np.round(time_lst + window/2 + offset, 7)

for i_ev in range(n_events_lst):

    condition_lo = ( time_lim_lo[i_ev] <  time_magic )
    condition_hi = ( time_magic <= time_lim_hi[i_ev] )
    
    if np.count_nonzero(condition_lo&condition_hi) == 1:
        
        index_magic = np.where(condition_lo&condition_hi)[0][0]
        indices_magic.append(index_magic)
        indices_lst.append(i_ev)
        run_ids_lst.append(data_lst.iloc[i_ev].name[0])
        subrun_ids_lst.append(data_lst.iloc[i_ev].name[1])
        event_ids_lst.append(data_lst.iloc[i_ev].name[2])

df_lst = data_lst.iloc[indices_lst]

foclen_lst = 28  # unit: [m]
df_lst['length'] = foclen_lst*np.tan(np.deg2rad(df_lst['length'].values))
df_lst['width'] = foclen_lst*np.tan(np.deg2rad(df_lst['width'].values))

for tel_id in [1, 2]:
    df_magic[tel_id] = df_magic[tel_id].iloc[indices_magic]
    df_magic[tel_id]['run_id_lst'] = run_ids_lst
    df_magic[tel_id]['subrun_id_lst'] = subrun_ids_lst
    df_magic[tel_id]['event_id_lst'] = event_ids_lst
    df_magic[tel_id].reset_index(inplace=True)
    df_magic[tel_id]['tel_id'] = np.repeat(tel_id+4, len(df_magic[tel_id]))   # convert MAGIC tel_id to from 1,2 to 5,6
    df_magic[tel_id].rename(columns={'obs_id': 'run_id_magic', 'event_id': 'event_id_magic'}, inplace=True)
    df_magic[tel_id].set_index(['run_id_lst', 'subrun_id_lst', 'event_id_lst', 'tel_id'], inplace=True)

data_stereo = pd.concat([df_lst, df_magic[1], df_magic[2]])
data_stereo = data_stereo.sort_index()   
data_stereo['offset_avg'] = np.repeat(offset_avg, len(data_stereo))

data_stereo.to_hdf(args.output_file, key='dl1/hillas_params')

for tel_id, tel_name in zip([1, 5, 6], ['LST1', 'MAGIC1', 'MAGIC2']):
    df = data_stereo.query(f'tel_id == {tel_id}')
    print(f'{tel_name}:  {len(df)} events')

print('\nDone.')

end_time = time.time()

print(f'\nelapsed_time = {end_time-start_time:.0f} [sec]')
