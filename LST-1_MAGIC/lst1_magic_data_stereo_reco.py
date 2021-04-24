#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import sys
import time
import yaml
import argparse
import warnings
import numpy as np 
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, Angle, EarthLocation
from astropy.coordinates.angle_utilities import angular_separation
from ctapipe.io import event_source
from ctapipe.reco import HillasReconstructor
from ctapipe.containers import HillasParametersContainer, ReconstructedShowerContainer
from ctapipe.reco.reco_algorithms import InvalidWidthException

def calc_impact(core_x, core_y, az, alt, tel_pos_x, tel_pos_y, tel_pos_z): 
    t = (tel_pos_x - core_x) * np.cos(alt) * np.cos(az) - (tel_pos_y - core_y) * np.cos(alt) * np.sin(az) + tel_pos_z * np.sin(alt)    
    impact = np.sqrt((core_x - tel_pos_x + t * np.cos(alt) * np.cos(az))**2 + \
                     (core_y - tel_pos_y - t * np.cos(alt) * np.sin(az))**2 + (t * np.sin(alt) - tel_pos_z)**2)
    return impact

deg2arcmin = 60

warnings.simplefilter('ignore')

start_time = time.time()

# ========================
# === Get the argument ===
# ========================

parser = argparse.ArgumentParser()

parser.add_argument('--input-file', '-i', dest='input_file', type=str, 
    help='Path to the input file. The DL1 coincidence file with HDF5 format is needed, f.g dl1_coincidence_lst1_magic_Run02923.0000.h5')

parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./dl1_stereo.h5', 
    help='Path and name of the output file with HDF5 format.')

parser.add_argument('--config-file', '-c', dest='config_file', type=str, default='./config.yaml', help='Path to the config file')

args = parser.parse_args()

config = yaml.safe_load(open(args.config_file, "r"))

print('\nTelescope IDs: {}'.format(config['tel_ids']))

allowed_tel_ids = []
for tel_name in config['tel_ids']:
    allowed_tel_ids.append(config['tel_ids'][tel_name])

# ===========================
# === Load the input file === 
# ===========================

print(f'\nLoading {args.input_file}')

data_stereo = pd.read_hdf(args.input_file, key='events/params')
data_stereo.sort_index(inplace=True)

for tel_name in config['tel_ids']:
    df = data_stereo.query('tel_id == {}'.format(config['tel_ids'][tel_name]))
    print(f'{tel_name}: {len(df)} events')

# ====================================
# === Check the angular separation ===
# ====================================

print('\nChecking the pointing direction...')

df_lst = data_stereo.query('tel_id == {}'.format(config['tel_ids']['LST-1']))
df_magic = data_stereo.query('tel_id == {}'.format(config['tel_ids']['MAGIC-I']))

theta = angular_separation(df_lst['az_tel'].values*u.rad, df_lst['alt_tel'].values*u.rad,
                            df_magic['az_tel'].values*u.rad, df_magic['alt_tel'].values*u.rad)

theta = theta.to(u.deg).value

theta_lim = config['stereo_reco']['theta_lim']
condition = (theta*deg2arcmin > theta_lim)

if np.sum(condition) > 0:
    print(f'--> Angular separation is larger than {theta_lim} arcmin. ' \
                'Input events may be taken with different pointing direction. Please check the input data. Exiting.')
    sys.exit()
else:
    print(f'--> The angular separation is less than {theta_lim} arcmin. Continuing.')
        
# ===========================
# === Define the subarray ===
# ===========================

data_path_simtel = config['stereo_reco']['simtel_path']

source = event_source(input_url=data_path_simtel)
subarray = source.subarray

tel_positions = subarray.positions
tel_positions_cog = {}

tels_pos_x = []
tels_pos_y = []
tels_pos_z = []

for tel_id in allowed_tel_ids:
    tels_pos_x.append(tel_positions[tel_id].value[0])
    tels_pos_y.append(tel_positions[tel_id].value[1])
    tels_pos_z.append(tel_positions[tel_id].value[2])

for tel_id in tel_positions.keys():
    pos_x_cog = tel_positions[tel_id][0].value - np.mean(tels_pos_x)
    pos_y_cog = tel_positions[tel_id][1].value - np.mean(tels_pos_y)
    pos_z_cog = tel_positions[tel_id][2].value - np.mean(tels_pos_z)
    
    tel_positions_cog[tel_id] = [pos_x_cog, pos_y_cog, pos_z_cog]*u.m
    
subarray.positions = tel_positions_cog

print('\nTelescope positions:')
for tel_name in config['tel_ids']:
    print('{}: {}'.format(tel_name, tel_positions_cog[config['tel_ids'][tel_name]]))

# =======================================
# === Calculate the stereo parameters ===
# =======================================

hillas_reconstructor = HillasReconstructor()
horizon_frame = AltAz()

# === initialize the event container ===
container = {}
for param in ReconstructedShowerContainer().keys():
    if param == 'tel_ids':
        container['num_tels'] = []
    else:
        container[param] = []

event_ids = np.unique(data_stereo.index.get_level_values('event_id').values)
n_events = len(event_ids)

print('\nReconstructing the stereo parameters...')
    
for i_ev, event_id in enumerate(event_ids):
    
    if i_ev%100 == 0:
        print(f'{i_ev}/{n_events} events')
    
    df_ev = data_stereo.query(f'event_id == {event_id}')
    
    # === define the array_pointing object === 
    alt_tel = u.Quantity(df_ev['alt_tel'].values[0], u.rad)
    az_tel = u.Quantity(df_ev['az_tel'].values[0], u.rad)
    array_pointing = SkyCoord(alt=alt_tel, az=az_tel, frame=horizon_frame)
    
    hillas_params = {}
    
    for tel_id in allowed_tel_ids:    
    
        # === define the hillas parameter containers === 
        df_tel = df_ev.query(f'tel_id == {tel_id}')
        hillas_params[tel_id] = HillasParametersContainer()
        hillas_params[tel_id].x = u.Quantity(df_tel['x'].values[0], u.m)
        hillas_params[tel_id].y = u.Quantity(df_tel['y'].values[0], u.m)
        hillas_params[tel_id].r = u.Quantity(df_tel['r'].values[0], u.m)
        hillas_params[tel_id].length = u.Quantity(df_tel['length'].values[0], u.m)
        hillas_params[tel_id].width = u.Quantity(df_tel['width'].values[0], u.m)
        hillas_params[tel_id].phi = Angle(df_tel['phi'].values[0], u.deg)
        hillas_params[tel_id].psi = Angle(df_tel['psi'].values[0], u.deg)
        hillas_params[tel_id].intensity = df_tel['intensity'].values[0]
        hillas_params[tel_id].skewness = df_tel['skewness'].values[0]
        hillas_params[tel_id].kurtosis = df_tel['kurtosis'].values[0]
    
    # === calculate the stereo parameters ===
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

for param in container.keys():
    container[param] = np.array(container[param])

container['az'][container['az'] < 0] += 2*np.pi

# === convert the "Alt/Az" to "RA/Dec" === 
print(f'\nTransforming "Alt/Az" to "RA/Dec" direction...')

config_loc = config['obs_location']
location = EarthLocation.from_geodetic(lat=config_loc['lat']*u.deg, lon=config_loc['lon']*u.deg, height=config_loc['height']*u.m)

df = data_stereo.query('tel_id == {}'.format(config['tel_ids']['LST-1']))
ts_type = config['coincidence']['timestamp_lst']

timestamps = Time(df[ts_type].values, format='unix', scale='utc')
horizon_frames = AltAz(location=location, obstime=timestamps)

event_coords = SkyCoord(alt=container['alt'], az=container['az'], unit='rad', frame=horizon_frames)
event_coords = event_coords.transform_to('fk5')

container['ra'] = event_coords.ra.value
container['dec'] = event_coords.dec.value

# === store the parameter in the data frame ===
for tel_id in allowed_tel_ids:

    # --- calculate the Impact parameter ---
    impact = calc_impact(container['core_x'], container['core_y'], container['az'], container['alt'],
                         tel_positions_cog[tel_id][0].value, tel_positions_cog[tel_id][1].value, tel_positions_cog[tel_id][2].value)
    
    data_stereo.loc[(slice(None), slice(None), tel_id), 'impact'] = impact
    
    for param in container.keys():
        data_stereo.loc[(slice(None), slice(None), tel_id), param] = container[param]

data_stereo.to_hdf(args.output_file, key='events/params')

end_time = time.time()
print(f'\nDone. Elapsed time = {end_time - start_time:.0f} [sec]')    

