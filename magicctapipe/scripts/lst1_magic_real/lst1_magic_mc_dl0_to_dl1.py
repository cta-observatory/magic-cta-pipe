#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import time
import yaml
import argparse
import warnings
import numpy as np
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
from magicctapipe.utils import MAGIC_Cleaning, calc_impact

warnings.simplefilter('ignore')

__all__ = ['mc_dl0_to_dl1']


class InfoContainer(Container):

    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    mc_energy = Field(-1, 'MC event energy', unit=u.TeV)
    mc_alt = Field(-1, 'MC event altitude', unit=u.rad)
    mc_az = Field(-1, 'MC event azimuth', unit=u.rad)
    mc_core_x = Field(-1, 'MC Core X', unit=u.m)
    mc_core_y = Field(-1, 'MC Core Y', unit=u.m)
    mc_impact = Field(-1, 'MC Impact', unit=u.m)
    alt_tel = Field(-1, 'Telescope altitude', unit=u.rad)
    az_tel = Field(-1, 'Telescope azimuth', unit=u.rad)
    n_islands = Field(-1, 'Number of image islands')


def mc_dl0_to_dl1(input_data_path, output_data_path, config):

    print(f'\nInput data file:\n{input_data_path}')

    source = event_source(input_data_path)
    subarray = source.subarray

    positions = subarray.positions

    print(f'\nSubarray configuration:\n{subarray.tels}')

    mc_tel_ids = config['mc_tel_ids']

    print(f'\nAllowed telescopes:\n{mc_tel_ids}')

    config_lst = config['LST']
    config_magic = config['MAGIC']

    config_magic['magic_clean']['findhotpixels'] = False   # for MC data 

    print(f'\nConfiguration for LST data process:\n{config_lst}')
    print(f'\nConfiguration for MAGIC data process:\n{config_magic}')

    # --- configure the processors --- 
    extractor_lst = ImageExtractor.from_name(
        'LocalPeakWindowSum', subarray=subarray, config=Config(config_lst['LocalPeakWindowSum'])
    )

    extractor_magic = ImageExtractor.from_name(
        'LocalPeakWindowSum', subarray=subarray, config=Config(config_magic['LocalPeakWindowSum'])
    )

    calibrator_lst = CameraCalibrator(subarray, image_extractor=extractor_lst)
    calibrator_magic = CameraCalibrator(subarray, image_extractor=extractor_magic)

    geom_camera = {tel_id: subarray.tel[tel_id].camera.geometry for tel_id in mc_tel_ids.values()}

    magic_clean = MAGIC_Cleaning.magic_clean(
        geom_camera[mc_tel_ids['MAGIC-I']], config_magic['magic_clean']
    )

    # --- process the input data ---
    with HDF5TableWriter(filename=output_data_path, group_name='events', overwrite=True) as writer:
        
        for tel_type in mc_tel_ids.keys():

            print(f'\nProcessing the {tel_type} events...')

            tel_id = mc_tel_ids[tel_type]

            source = event_source(input_data_path, allowed_tels=[tel_id])

            for i_ev, event in enumerate(source):

                if i_ev%100 == 0:
                    print(f'{i_ev} events')

                if tel_type == 'LST-1':

                    # --- calibration ---
                    calibrator_lst(event)

                    # --- image cleaning (lstchain v0.6.3) ---
                    image = event.dl1.tel[tel_id].image
                    peak_time = event.dl1.tel[tel_id].peak_time

                    signal_pixels = tailcuts_clean(
                        geom_camera[tel_id], image, **config_lst['tailcuts_clean']
                    )
                    
                    num_islands, island_labels = number_of_islands(geom_camera[tel_id], signal_pixels)

                    n_pixels_on_island = np.bincount(island_labels.astype(np.int))
                    n_pixels_on_island[0] = 0  

                    max_island_label = np.argmax(n_pixels_on_island)
                    signal_pixels[island_labels != max_island_label] = False

                elif ( tel_type == 'MAGIC-I' ) or ( tel_type == 'MAGIC-II' ):

                    # --- calibration --- 
                    calibrator_magic(event)
                    
                    # --- image cleaning (sum image clean) ---
                    signal_pixels, image, peak_time = magic_clean.clean_image(
                        event.dl1.tel[tel_id].image, event.dl1.tel[tel_id].peak_time
                    )

                num_islands, island_labels = number_of_islands(geom_camera[tel_id], signal_pixels)

                image_cleaned = image.copy()
                image_cleaned[~signal_pixels] = 0

                peak_time_cleaned = peak_time.copy()
                peak_time_cleaned[~signal_pixels] = 0

                if np.any(image_cleaned):

                    # --- Hillas parameter calculation ---
                    try:    
                        hillas_params = hillas_parameters(geom_camera[tel_id], image_cleaned)
                    
                    except:
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}): ' \
                            'Hillas parameter calculation failed. Skipping.')
                        continue
                        
                    # --- Timing parameter calculation ---
                    try:    
                        timing_params = timing_parameters(
                            geom_camera[tel_id], image_cleaned, peak_time_cleaned, hillas_params, signal_pixels
                        )
                    
                    except:
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}): ' \
                            'Timing parameter calculation failed. Skipping.')
                        continue
                    
                    # --- Leakage parameter calculation --- 
                    try:
                        leakage_params = leakage(geom_camera[tel_id], image, signal_pixels)
                    
                    except: 
                        print(f'--> {i_ev} event (event ID = {event.index.event_id}): ' \
                            'Leakage parameter calculation failed. Skipping.')
                        continue

                    # --- calculate the MC impact parameter ---
                    mc_impact = calc_impact(
                        event.mc.core_x, event.mc.core_y, event.mc.az, event.mc.alt,
                        positions[tel_id][0], positions[tel_id][1], positions[tel_id][2],
                    )

                    # --- write the event information ---
                    event_info = InfoContainer(
                        obs_id=event.index.obs_id,
                        event_id=event.index.event_id,
                        mc_energy=event.mc.energy,
                        mc_alt=event.mc.alt,
                        mc_az=event.mc.az,
                        mc_core_x=event.mc.core_x,
                        mc_core_y=event.mc.core_y,
                        mc_impact = mc_impact,
                        alt_tel=event.pointing.tel[tel_id].altitude,
                        az_tel=event.pointing.tel[tel_id].azimuth,
                        n_islands=num_islands
                    )

                    if tel_type == 'LST-1':
                        event_info.tel_id = 1

                    elif tel_type == 'MAGIC-I':
                        event_info.tel_id = 2
                    
                    elif tel_type == 'MAGIC-II':
                        event_info.tel_id = 3

                    writer.write('params', (event_info, hillas_params, timing_params, leakage_params))
                    
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
        help='Path to an input MC DL0 data file, e.g., gamma_40deg_90deg_run1___*.simtel.gz'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, 
        help='Path and name of an output data file with HDF5 format, e.g., dl1_gamma_40deg_90deg.h5'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, 
        help='Path to a config file with yaml format, e.g., config.yaml'
    )

    args = arg_parser.parse_args()

    # --- process the MC DL0 data to DL1 --- 
    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    mc_dl0_to_dl1(args.input_data, args.output_data, config_lst1_magic)

    print(f'\nOutput data file: {args.output_data}')

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
