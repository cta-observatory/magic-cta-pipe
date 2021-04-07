#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import sys
import time
import warnings
import argparse
import numpy as np 
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, Angle
from astropy.coordinates.angle_utilities import angular_separation
from ctapipe.io import event_source
from ctapipe.reco import HillasReconstructor
from ctapipe.containers import HillasParametersContainer, ReconstructedShowerContainer
from ctapipe.reco.reco_algorithms import InvalidWidthException

warnings.simplefilter('ignore')

start_time = time.time()

# ========================
# === Get the argument ===
# ========================

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', '-i', dest='input_file', type=str, help='Path to the input file')
parser.add_argument('--output-file', '-o', dest='output_file', type=str, help='Path to the output file')

args = parser.parse_args()

# =====================================
# === Load the DL1 coincidence file ===
# =====================================

print('\nLoading the DL1 coincidence file...')

data_stereo = pd.read_hdf(args.input_file, key='dl1/hillas_params')
print(args.input_file)

for tel_id, tel_name in zip([1, 5, 6], ['LST-1', 'MAGIC1', 'MAGIC2']):
    df = data_stereo.query(f'tel_id == {tel_id}')
    print(f'{tel_name}: {len(df)} events')

# ====================================
# === Check the pointing direction ===
# ====================================

print('\nChecking the pointing direction...')

az_lst = data_stereo.query('tel_id == 1')['az_tel'].values*u.rad
alt_lst = data_stereo.query('tel_id == 1')['alt_tel'].values*u.rad
az_magic = data_stereo.query('tel_id == 5')['az_tel'].values*u.rad
alt_magic = data_stereo.query('tel_id == 5')['alt_tel'].values*u.rad

theta = angular_separation(az_lst, alt_lst, az_magic, alt_magic)
theta = theta.to(u.deg).value

theta_lim = 2/60  # 2 arcmin, preliminary value
condition = (theta > theta_lim)

if np.sum(condition) != 0:
    print(f'--> Angular separation is larger than {theta_lim*60} arcmin. ' \
                'Input events may be taken with different pointing direction. Please check the input file. Exiting.')
    sys.exit()
else:
    print(f'--> The angular separation is less than {theta_lim*60} arcmin. Continuing.')

# =======================================
# === Calculate the Stereo parameters ===
# =======================================

# Extract the "subarray" configuration by loading the simtel file with event_source:
data_path_simtel = '/home/yoshiki.ohtani/workspace/Data/LaPalma/4LSTs_MAGIC/gamma/zenith_20deg/' \
                    'south_pointing/run1/sim_telarray_v3_trans_80%/cta-prod5-lapalma_4LSTs_MAGIC/0.0deg/Data/' \
                    'gamma_20deg_180deg_run1___cta-prod5-lapalma_4LSTs_MAGIC_desert-2158m_mono.simtel.gz'

source = event_source(input_url=data_path_simtel)
subarray = source.subarray

run_id = np.unique(data_stereo.index.get_level_values('run_id_lst').values)[0]
subrun_id = np.unique(data_stereo.index.get_level_values('subrun_id_lst').values)[0]
event_ids = np.unique(data_stereo.index.get_level_values('event_id_lst').values)
n_events = len(event_ids)

container = {}
for param in ReconstructedShowerContainer().keys():
    if param == 'tel_ids':
        container['num_tels'] = []
    else:
        container[param] = []

horizon_frame = AltAz()
hillas_reconstructor = HillasReconstructor()
    
print('\nCalculating the stereo parameters...')

for i_ev, event_id in enumerate(event_ids):
    
    if i_ev%100 == 0:
        print(f'{i_ev}/{n_events} events')
    
    df_ev = data_stereo.query(f'event_id_lst == {event_id}')
    
    # define the array_pointing object 
    alt_tel = u.Quantity(df_ev.query('tel_id == 1')['alt_tel'].values[0], u.rad)
    az_tel = u.Quantity(df_ev.query('tel_id == 1')['az_tel'].values[0], u.rad)
    array_pointing = SkyCoord(alt=alt_tel, az=az_tel, frame=horizon_frame)
    
    hillas_params = {}

    for tel_id in [1, 5, 6]:

        # define the hillas parameters container
        df_tel = df_ev.query(f'tel_id == {tel_id}')
        hillas_params[tel_id] = HillasParametersContainer()
        hillas_params[tel_id].x = u.Quantity(df_tel['x'].values[0], u.m)
        hillas_params[tel_id].y = u.Quantity(df_tel['y'].values[0], u.m)
        hillas_params[tel_id].r = u.Quantity(df_tel['r'].values[0], u.m)
        hillas_params[tel_id].length = u.Quantity(df_tel['length'].values[0], u.m)
        hillas_params[tel_id].width = u.Quantity(df_tel['width'].values[0], u.m)
        hillas_params[tel_id].phi = Angle(df_tel['phi'].values[0], u.rad)
        hillas_params[tel_id].psi = Angle(df_tel['psi'].values[0], u.rad)
        hillas_params[tel_id].intensity = df_tel['intensity'].values[0]
        hillas_params[tel_id].skewness = df_tel['skewness'].values[0]
        hillas_params[tel_id].kurtosis = df_tel['kurtosis'].values[0]
    
    # calculate the stereo parameters
    try:
        stereo_params = hillas_reconstructor.predict(hillas_params, subarray, array_pointing)
    except InvalidWidthException:
        print(f'--> event ID {event_id}: HillasContainer contains width = 0 or nan. Stereo parameter calculation skipped.')
        stereo_params = ReconstructedShowerContainer()
    
    for param in stereo_params.keys():
        if 'astropy' in str(type(stereo_params[param])):
            container[param].append(stereo_params[param].value)
        elif param == 'tel_ids':
            container['num_tels'].append(len(stereo_params['tel_ids']))
        else:
            container[param].append(stereo_params[param])

print(f'{i_ev+1}/{n_events} events')

# store the parameter in the data frame
for tel_id in [1, 5, 6]:
    for param in container.keys():
        data_stereo.loc[(slice(None), slice(None), slice(None), tel_id), param] = container[param]

data_stereo.to_hdf(args.output_file, key='dl1/hillas_params')

print('\nDone.')    

end_time = time.time()
print(f'\nelapsed_time = {end_time-start_time:.0f} [sec]')