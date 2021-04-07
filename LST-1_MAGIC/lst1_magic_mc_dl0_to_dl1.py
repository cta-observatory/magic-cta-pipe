#!/usr/bin/env python
# coding: utf-8

import sys
import glob
import time
import yaml
import scipy
import argparse
import warnings
import numpy as np
import pandas as pd
from traitlets.config import Config
from astropy import units as u
from ctapipe.io import event_source
from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageExtractor, hillas_parameters, leakage
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.timing import timing_parameters
from ctapipe.image.morphology import number_of_islands
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

start_time = time.time()

# ===================================================
# === Get the argument and load the configuration ===
# ===================================================

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', '-i', dest='input_file', type=str, help='Path to the input file')
parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./dl1_data.h5', help='Path to the output file')
parser.add_argument('--config-file', '-c', dest='config_file', type=str, default='./config.yaml', help='Path to the config file')

args = parser.parse_args()

print('\nTelescope IDs: {}'.format(config['tel_ids']))

config = yaml.safe_load(open(args.config_file, "r"))

allowed_tels = []
for tel_name in config['tel_ids']:
    allowed_tels.append(config['tel_ids'][tel_name])

# =======================================================
# === Define the Camera Calibrator and Image Cleaning === 
# =======================================================

source = event_source(args.input_file)
subarray = source.subarray

geom_camera = dict()

for tel_id in allowed_tels:
    geom_camera[tel_id] = subarray.tel[tel_id].camera.geometry

print('\nIntegration configuration:\n {}'.format(config['integration']))

extractor = {}

extractor['LST'] = ImageExtractor.from_name(list(config['integration']['LST'])[0], subarray=subarray, config=Config(config['integration']['LST']))
extractor['MAGIC'] = ImageExtractor.from_name(list(config['integration']['MAGIC'])[0], subarray=subarray, config=Config(config['integration']['MAGIC']))

calibrator = {config['tel_ids']['LST-1']: CameraCalibrator(subarray, image_extractor=extractor['LST']),
              config['tel_ids']['MAGIC-I']: CameraCalibrator(subarray, image_extractor=extractor['MAGIC']), 
              config['tel_ids']['MAGIC-II']: CameraCalibrator(subarray, image_extractor=extractor['MAGIC'])}

print('\nCleaning configuration:\n {}'.format(config['cleaning']))

geom_magic = geom_camera[config['tel_ids']['MAGIC-I']]
magic_clean = MAGIC_Cleaning.magic_clean(geom_magic, config['cleaning']['MAGIC']['MAGICCleaning'])

# ==============================================
# === Process the data from DL0 to DL1 level ===
# ==============================================

with HDF5TableWriter(filename=args.output_file, group_name='events', overwrite=True) as writer:
    
    print(f'\nProcessing: \n {args.input_file}\n')
    
    with event_source(args.input_file, max_events=None) as source:

        for i_ev, event in enumerate(source):

            if i_ev%100 == 0:
                print(f'{i_ev} events')

            tels_with_data = set(event.r1.tels_with_data) & set(allowed_tels)
            
            for tel_id in tels_with_data:

                # === Integrate the r1 waveform ===
                calibrator[tel_id](event, tel_id)

                # === Image cleaning === 
                image = event.dl1.tel[tel_id].image
                peak_time = event.dl1.tel[tel_id].peak_time

                if tel_id == 1: 
                    signal_pixels = tailcuts_clean(geom_camera[tel_id], image, **config['cleaning']['LST']['tailcuts_clean'])
                else:
                    signal_pixels, _ , _ = magic_clean.clean_image(image, peak_time)

                image_cleaned = image.copy()
                image_cleaned[~signal_pixels] = 0

                peak_time_cleaned = peak_time.copy()
                peak_time_cleaned[~signal_pixels] = 0

                num_islands, island_labels = number_of_islands(geom_camera[tel_id], signal_pixels)

                if np.any(image_cleaned):

                    # === Hillas parameter calculation ===
                    try:    
                        hillas_params = hillas_parameters(geom_camera[tel_id], image_cleaned)
                    except:
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}, tel_id = {tel_id}): Hillas parameter calculation failed. Skipping.')
                        continue
                        
                    # === Timing parameter calculation ===
                    try:    
                        timing_params = timing_parameters(geom_camera[tel_id], image_cleaned, peak_time_cleaned, hillas_params, signal_pixels)
                    except:
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}, tel_id = {tel_id}): Timing parameter calculation failed. Skipping.')
                        continue
                    
                    # === Leakage parameter calculation === 
                    try:
                        leakage_params = leakage(geom_camera[tel_id], image, signal_pixels)
                    except: 
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}, tel_id = {tel_id}): Leakage parameter calculation failed. Skipping.')
                        continue
                    
                    event_info = InfoContainer(
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
                    
                    writer.write("params", (event_info, hillas_params, timing_params, leakage_params))
                
print(f'{i_ev+1} events')

print(f'\nOutput file:\n {args.output_file}')

end_time = time.time()
print(f'\nDone. Elapsed time = {end_time - start_time:.0f} [sec]')
