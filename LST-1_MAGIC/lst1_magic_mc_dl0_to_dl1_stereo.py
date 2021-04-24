#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import time
import yaml
import scipy
import argparse
import warnings
import numpy as np
import pandas as pd
from traitlets.config import Config
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, Angle
from ctapipe.io import event_source
from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from ctapipe.containers import ReconstructedShowerContainer
from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageExtractor, hillas_parameters, leakage
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.timing import timing_parameters
from ctapipe.image.morphology import number_of_islands
from ctapipe.reco import HillasReconstructor
from ctapipe.reco.reco_algorithms import InvalidWidthException
from utils import MAGIC_Cleaning

warnings.simplefilter('ignore')

class InfoContainer(Container):
    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    mc_energy = Field(-1, "MC event energy", unit=u.TeV)
    mc_core_x = Field(-1, "MC Core X", unit=u.m)
    mc_core_y = Field(-1, "MC Core Y", unit=u.m)
    mc_alt = Field(-1, "MC event altitude", unit=u.rad)
    mc_az = Field(-1, "MC event azimuth", unit=u.rad)
    alt_tel = Field(-1, "Telescope altitude", unit=u.rad)
    az_tel = Field(-1, "Telescope azimuth", unit=u.rad)
    n_islands = Field(-1, "Number of image islands")

def calc_impact(core_x, core_y, az, alt, tel_pos_x, tel_pos_y, tel_pos_z): 
    t = (tel_pos_x - core_x) * np.cos(alt) * np.cos(az) - (tel_pos_y - core_y) * np.cos(alt) * np.sin(az) + tel_pos_z * np.sin(alt)    
    impact = np.sqrt((core_x - tel_pos_x + t * np.cos(alt) * np.cos(az))**2 + \
                     (core_y - tel_pos_y - t * np.cos(alt) * np.sin(az))**2 + (t * np.sin(alt) - tel_pos_z)**2)
    return impact

start_time = time.time()

# ===================================================
# === Get the argument and load the configuration ===
# ===================================================

parser = argparse.ArgumentParser()

parser.add_argument('--input-file', '-i', dest='input_file', type=str, 
    help='Path to the input simtel file that contains both LST-1 and MAGIC telescopes.')

parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./dl1_data.h5', 
    help='Path and name of the output file with HDF5 format.')

parser.add_argument('--config-file', '-c', dest='config_file', type=str, default='./config.yaml', help='Path to the config file')

args = parser.parse_args()

config = yaml.safe_load(open(args.config_file, "r"))

print('\nTelescope IDs: {}'.format(config['tel_ids']))

allowed_tel_ids = []
for tel_name in config['tel_ids']:
    allowed_tel_ids.append(config['tel_ids'][tel_name])

# ===========================
# === Define the subarray ===
# ===========================

source = event_source(args.input_file)
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

# ================================================================
# === Define the Calibrator, Cleaning and Stereo Reconstructor === 
# ================================================================

geom_camera = dict()

for tel_id in allowed_tel_ids:
    geom_camera[tel_id] = subarray.tel[tel_id].camera.geometry

print('\nIntegration configuration:\n {}'.format(config['integration']))

extractor = {}

extractor['LST'] = ImageExtractor.from_name(list(config['integration']['LST'])[0], subarray=subarray, config=Config(config['integration']['LST']))
extractor['MAGIC'] = ImageExtractor.from_name(list(config['integration']['MAGIC'])[0], subarray=subarray, config=Config(config['integration']['MAGIC']))

calibrator = {}

for tel_name in config['tel_ids']:
    tel_id = config['tel_ids'][tel_name]

    if 'LST' in tel_name:
        calibrator[tel_id] = CameraCalibrator(subarray, image_extractor=extractor['LST'])
    elif 'MAGIC' in tel_name:
        calibrator[tel_id] = CameraCalibrator(subarray, image_extractor=extractor['MAGIC'])

print('\nCleaning configuration:\n {}'.format(config['cleaning']))

geom_magic = geom_camera[config['tel_ids']['MAGIC-I']]
magic_clean = MAGIC_Cleaning.magic_clean(geom_magic, config['cleaning']['MAGIC']['MAGICCleaning'])

horizon_frame = AltAz()
hillas_reconstructor = HillasReconstructor()

# =====================================================
# === Process the data from DL0 to DL1-stereo level ===
# =====================================================

with HDF5TableWriter(filename=args.output_file, group_name='events', overwrite=True) as writer:
    
    print(f'\nProcessing: \n {args.input_file}\n')
    
    with event_source(args.input_file, max_events=None) as source:

        for i_ev, event in enumerate(source):

            if i_ev%100 == 0:
                print(f'{i_ev} events')

            if i_ev == 0:

                # === array pointing initialization ===
                alt_tel = event.pointing.array_altitude
                az_tel = event.pointing.array_azimuth

                array_pointing = SkyCoord(alt=alt_tel, az=az_tel, frame=horizon_frame)

            hillas_params = {}
            timing_params = {}
            leakage_params = {} 
            event_info = {}

            tels_with_data = set(event.r1.tels_with_data) & set(allowed_tel_ids)
            
            for tel_id in tels_with_data:

                # === Waveform integration ===
                calibrator[tel_id](event, tel_id)

                # === Image cleaning === 
                image = event.dl1.tel[tel_id].image
                peak_time = event.dl1.tel[tel_id].peak_time

                if tel_id == config['tel_ids']['LST-1']: 
                    signal_pixels = tailcuts_clean(geom_camera[tel_id], image, **config['cleaning']['LST']['tailcuts_clean'])
                    
                    # use only the main island
                    num_islands, island_labels = number_of_islands(geom_camera[tel_id], signal_pixels)
                    n_pixels_on_island = np.bincount(island_labels.astype(np.int))
                    n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
                    max_island_label = np.argmax(n_pixels_on_island)
                    signal_pixels[island_labels != max_island_label] = False

                else:
                    signal_pixels, _ , _ = magic_clean.clean_image(image, peak_time)
                    num_islands, island_labels = number_of_islands(geom_camera[tel_id], signal_pixels)

                image_cleaned = image.copy()
                image_cleaned[~signal_pixels] = 0

                peak_time_cleaned = peak_time.copy()
                peak_time_cleaned[~signal_pixels] = 0

                if np.any(image_cleaned):

                    # === Hillas parameter calculation ===
                    try:    
                        hillas_params[tel_id] = hillas_parameters(geom_camera[tel_id], image_cleaned)
                    except:
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}, tel_id = {tel_id}): Hillas parameter calculation failed. Skipping.')
                        continue
                        
                    # === Timing parameter calculation ===
                    try:    
                        timing_params[tel_id] = timing_parameters(geom_camera[tel_id], image_cleaned, peak_time_cleaned, hillas_params[tel_id], signal_pixels)
                    except:
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}, tel_id = {tel_id}): Timing parameter calculation failed. Skipping.')
                        continue
                    
                    # === Leakage parameter calculation === 
                    try:
                        leakage_params[tel_id] = leakage(geom_camera[tel_id], image, signal_pixels)
                    except: 
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}, tel_id = {tel_id}): Leakage parameter calculation failed. Skipping.')
                        continue
                    
                    event_info[tel_id] = InfoContainer(
                        obs_id=event.index.obs_id,
                        event_id=scipy.int32(event.index.event_id),
                        tel_id=tel_id,
                        mc_energy=event.mc.energy,
                        mc_core_x=event.mc.core_x,
                        mc_core_y=event.mc.core_y,
                        mc_alt=event.mc.alt,
                        mc_az=event.mc.az,
                        alt_tel=event.mc.tel[tel_id].altitude_raw*u.rad,
                        az_tel=event.mc.tel[tel_id].azimuth_raw*u.rad,
                        n_islands=num_islands)
                
            if len(hillas_params) == len(allowed_tel_ids):

                # === Stereo parameter calculation ===
                try:
                    stereo_params = hillas_reconstructor.predict(hillas_params, subarray, array_pointing)

                except InvalidWidthException:
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): Stereo parameter calculation failed. Skipping.')
                    stereo_params = ReconstructedShowerContainer()

            else:
                stereo_params = ReconstructedShowerContainer()

            # === storing the parameters in the output file === 
            for tel_id in event_info.keys():
                writer.write("params", (event_info[tel_id], hillas_params[tel_id], timing_params[tel_id], leakage_params[tel_id], stereo_params))

print(f'{i_ev+1} events')

# === open the data files === 
data = pd.read_hdf(args.output_file, key='events/params')
data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
data.sort_index(inplace=True)

# --- change the scale and units of "az" and "alt" ---
data['az'] = np.deg2rad(data['az'].values)
data['alt'] = np.deg2rad(data['alt'].values)

azimuths = data['az'].values
azimuths[azimuths < 0] += 2*np.pi

data['az'] = azimuths

# --- calculate the Impact parameter ---
for tel_id in allowed_tel_ids:

    df = data.query(f'tel_id == {tel_id}')

    impact = calc_impact(df['core_x'].values, df['core_y'].values, df['az'].values, df['alt'].values,
                         tel_positions_cog[tel_id][0].value, tel_positions_cog[tel_id][1].value, tel_positions_cog[tel_id][2].value)

    mc_impact = calc_impact(df['mc_core_x'].values, df['mc_core_y'].values, df['mc_az'].values, df['mc_alt'].values,
                         tel_positions_cog[tel_id][0].value, tel_positions_cog[tel_id][1].value, tel_positions_cog[tel_id][2].value)
    
    data.loc[(slice(None), slice(None), tel_id), 'impact'] = impact
    data.loc[(slice(None), slice(None), tel_id), 'mc_impact'] = mc_impact

# --- check the number of events ---
n_events = len(data.groupby(['obs_id', 'event_id']).mean())
print(f'\nTotal: {n_events} events')

for tel_name in config['tel_ids']:
    df = data.query('tel_id == {}'.format(config['tel_ids'][tel_name]))
    print(f'{tel_name}: {len(df)} events')

data.to_hdf(args.output_file, key='events/params')

print(f'\nOutput file:\n {args.output_file}')

end_time = time.time()
print(f'\nDone. Elapsed time = {end_time - start_time:.0f} [sec]')
