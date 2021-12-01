#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import re
import yaml
import glob
import copy
import time
import argparse
import warnings 
import numpy as np
from astropy import units as u
from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from ctapipe.image import hillas_parameters, leakage
from ctapipe.image.timing import timing_parameters
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.morphology import number_of_islands 
from ctapipe_io_magic import MAGICEventSource
from utils import MAGIC_Badpixels, MAGIC_Cleaning

warnings.simplefilter('ignore') 

__all__ = ['magic_cal_to_dl1']


class InfoContainer(Container):
    
    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    mjd = Field(-1, 'Event mjd')
    millisec = Field(-1, 'Event millisec')
    nanosec = Field(-1, 'Event nanosec')
    alt_tel = Field(-1, 'Telescope altitude', unit=u.rad)
    az_tel = Field(-1, 'Telescope azimuth', unit=u.rad)
    n_islands = Field(-1, 'Number of image islands')


def magic_cal_to_dl1(input_data_path, output_data_path, config):

    config_cleaning = config['magic_clean']
    config_badpixels = config['bad_pixels']

    config_cleaning['findhotpixels'] = True   # for real data

    print(f'\nConfiguration for image cleaning:\n{config_cleaning}')
    print(f'\nConfiguration for bad pixels calculation:\n{config_badpixels}')

    # --- check the input data ---
    paths_list = glob.glob(input_data_path)
    paths_list.sort()

    print('\nProcess the following data files:')
    for path in paths_list:
        print(path)

    re_parser = re.findall('.*_M(\d)_.*\.root', paths_list[0])
    tel_id = int(re_parser[0])

    # --- process the input data ---
    previous_event_id = 0

    with HDF5TableWriter(filename=output_data_path, group_name='events', overwrite=True) as writer:
        
        source = MAGICEventSource(input_url=input_data_path)

        geom_camera = source.subarray.tel[tel_id].camera.geometry
        magic_clean = MAGIC_Cleaning.magic_clean(geom_camera, config_cleaning)
        badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(config=config_badpixels)

        print('\nProcessing the events...')

        for i_ev, event in enumerate(source._mono_event_generator(telescope=f'M{tel_id}')):  

            if i_ev%100 == 0:  
                print(f'{i_ev} events')

            if previous_event_id == event.index.event_id:   # exclude pedestal runs?? 
                continue

            previous_event_id = copy.copy(event.index.event_id)

            # --- image cleaning ---
            badrmspixel_mask = badpixel_calculator.get_badrmspixel_mask(event)
            deadpixel_mask = badpixel_calculator.get_deadpixel_mask(event)
            unsuitable_mask = np.logical_or(badrmspixel_mask[tel_id-1], deadpixel_mask[tel_id-1])

            signal_pixels, image, peak_time = magic_clean.clean_image(
                event.dl1.tel[tel_id].image, event.dl1.tel[tel_id].peak_time, unsuitable_mask=unsuitable_mask
            )

            image_cleaned = image.copy()
            image_cleaned[~signal_pixels] = 0

            peak_time_cleaned = peak_time.copy()
            peak_time_cleaned[~signal_pixels] = 0

            num_islands, island_labels = number_of_islands(geom_camera, signal_pixels)

            if np.any(image_cleaned):

                # --- Hillas parameter calculation ---
                try:    
                    hillas_params = hillas_parameters(geom_camera, image_cleaned)

                except:
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): ' \
                           'Hillas parameter calculation failed. Skipping.')
                    continue
                    
                # --- Timing parameter calculation ---
                try:    
                    timing_params = timing_parameters(
                        geom_camera, image_cleaned, peak_time_cleaned, hillas_params, signal_pixels
                    )

                except:
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): ' \
                           'Timing parameter calculation failed. Skipping.')
                    continue
                
                # --- Leakage parameter calculation --- 
                try:
                    leakage_params = leakage(geom_camera, image, signal_pixels)
                    
                except: 
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): ' \
                           'Leakage parameter calculation failed. Skipping.')
                    continue

                # --- write the event information ---
                event_info = InfoContainer(
                    obs_id=event.index.obs_id,
                    event_id=event.index.event_id,
                    tel_id=tel_id,
                    mjd=event.trigger.mjd,
                    millisec=event.trigger.millisec,
                    nanosec=event.trigger.nanosec,
                    alt_tel=event.pointing.tel[tel_id].altitude,
                    az_tel=event.pointing.tel[tel_id].azimuth,
                    n_islands=num_islands
                )

                writer.write('params', (hillas_params, leakage_params, timing_params, event_info)) 

            else: 
                print(f'--> {i_ev} event (event ID = {event.index.event_id}): ' \
                       'Could not survive the image cleaning. Skipping.')
                continue

        print(f'{i_ev+1} events')


def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str, 
        help='Path to input MAGIC Calibrated data file(s), e.g., *_Y_*.root'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str,  
        help='Path and name of an output data file with HDF5 format, e.g., dl1_magic.h5'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, 
        help='Path to a config file with yaml format, e.g., config.yaml'
    )

    args = arg_parser.parse_args()

    # --- process the MAGIC Calibrated data to DL1 --- 
    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))
    
    magic_cal_to_dl1(args.input_data, args.output_data, config_lst1_magic['MAGIC'])

    print(f'\nOutput data file: {args.output_data}')

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()