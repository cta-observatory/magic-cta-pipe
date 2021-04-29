#!/usr/bin/env python
# coding: utf-8

import re
import os
import sys
import yaml
import glob
import copy
import time
import scipy
import argparse
import warnings 
import pandas as pd
import numpy as np
from astropy import units as u
from ctapipe_io_magic import MAGICEventSource
from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from ctapipe.image import hillas_parameters, leakage
from ctapipe.image.timing import timing_parameters
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.morphology import number_of_islands
from ctapipe.instrument import CameraGeometry  

from utils import MAGIC_Badpixels
from utils import MAGIC_Cleaning

warnings.simplefilter('ignore') 

class InfoContainer(Container):
    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    mjd = Field(-1, "Event mjd")
    millisec = Field(-1, "Event millisec")
    nanosec = Field(-1, "Event nanosec")
    alt_tel = Field(-1, "MC telescope altitude", unit=u.rad)
    az_tel = Field(-1, "MC telescope azimuth", unit=u.rad)
    n_islands = Field(-1, "Number of image islands")

start_time = time.time()

# ========================
# === Get the argument ===
# ========================

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--input-dir', '-i', dest='input_dir', type=str, 
    help='Path to the MAGIC input directory that contains the MAGIC input files with root format.')

arg_parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./dl1_magic.h5', 
    help='Path and name of the output file with HDF5 format.')

arg_parser.add_argument('--config-file', '-c', dest='config_file', type=str, default='./config.yaml', help='Path to the config file')

args = arg_parser.parse_args()

# ============
# === Main ===
# ============

input_mask = args.input_dir + "/*_Y_*.root"
data_paths = glob.glob(input_mask)
data_paths.sort()

if data_paths == []:
    print('Error: Failed to find the input files. Please check your input directory. Exiting.')
    sys.exit()

tel_id = int(re.findall('.*_M(\d)_.*\.root', data_paths[0])[0])

previous_event_id = 0

with HDF5TableWriter(filename=args.output_file, group_name='events', overwrite=True) as writer:

    config = yaml.safe_load(open(args.config_file, "r"))

    print(f'\nLoading the input files under {args.input_dir}')
    print('\nCleaning configuration: \n{}'.format(config['cleaning']['MAGIC']['MAGICCleaning']))
    print('\nBad pixel calculator configuration: \n{}'.format(config['bad_pixels']['MAGIC']))
    
    source = MAGICEventSource(input_url=input_mask)

    geom_camera = source.subarray.tel[tel_id].camera
    magic_clean = MAGIC_Cleaning.magic_clean(geom_camera, config['cleaning']['MAGIC']['MAGICCleaning'])
    badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(config=config['bad_pixels']['MAGIC'])

    print('\nProcessing the events...')

    for i_ev, event in enumerate(source._mono_event_generator(telescope=f'M{tel_id}')):  
        
        if i_ev%1000 == 0:  
            print(f'{i_ev} events')

        #Exclude pedestal runs?? 
        if previous_event_id == event.index.event_id:
            continue

        previous_event_id = copy.copy(event.index.event_id)

        # === Image Cleaning ===
        image = event.dl1.tel[tel_id].image
        peak_time = event.dl1.tel[tel_id].peak_time

        badrmspixel_mask = badpixel_calculator.get_badrmspixel_mask(event)
        deadpixel_mask = badpixel_calculator.get_deadpixel_mask(event)
        unsuitable_mask = np.logical_or(badrmspixel_mask[tel_id-1], deadpixel_mask[tel_id-1])

        signal_pixels, image, peak_time = magic_clean.clean_image(image, peak_time, unsuitable_mask=unsuitable_mask)

        image_cleaned = image.copy()
        image_cleaned[~signal_pixels] = 0

        peak_time_cleaned = peak_time.copy()
        peak_time_cleaned[~signal_pixels] = 0

        num_islands, island_labels = number_of_islands(geom_camera, signal_pixels)

        if np.any(image_cleaned):

            # === Hillas parameter calculation ===
            try:    
                hillas_params = hillas_parameters(geom_camera, image_cleaned)
            except:
                print(f'--> {i_ev} event (event ID = {event.index.event_id}): Hillas parameter calculation failed. Skipping.')
                continue
                
            # === Timing parameter calculation ===
            try:    
                timing_params = timing_parameters(geom_camera, image_cleaned, peak_time_cleaned, hillas_params, signal_pixels)
            except:
                print(f'--> {i_ev} event (event ID = {event.index.event_id}): Timing parameter calculation failed. Skipping.')
                continue
            
            # === Leakage parameter calculation === 
            try:
                leakage_params = leakage(geom_camera, image, signal_pixels)
            except: 
                print(f'--> {i_ev} event (event ID = {event.index.event_id}): Leakage parameter calculation failed. Skipping.')
                continue

            event_info = InfoContainer(
                obs_id=event.index.obs_id,
                event_id=scipy.int32(event.index.event_id),
                tel_id=tel_id,
                mjd=event.trigger.mjd,
                millisec=event.trigger.millisec,
                nanosec=event.trigger.nanosec,
                alt_tel=event.pointing.tel[tel_id].altitude.to(u.rad),
                az_tel=event.pointing.tel[tel_id].azimuth.to(u.rad),
                n_islands=num_islands
            )

            writer.write("params", (event_info, hillas_params, leakage_params, timing_params))         

end_time = time.time()
print(f'elapsed time = {end_time - start_time} [sec]')


