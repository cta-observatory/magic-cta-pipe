#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

Process MAGIC calibrated data (*_Y_*.root) with MARS-like image cleaning method, 
and compute the DL1 parameters (i.e., Hillas, timing, and leakage parameters).
Do NOT input both M1 and M2 data at the same time when running the script.

Usage:
$ python magic_data_cal_to_dl1.py 
--input-files "./data/calibrated/20201119_M1_05093174.*_Y_CrabNebula-W0.40+035.root"
--output-file "./data/dl1/dl1_M1_run05093174.h5"
--config-file "./config.yaml"
"""

import sys
import copy
import glob
import time
import yaml
import uproot
import argparse
import warnings
import numpy as np
from astropy import units as u
from ctapipe.io import HDF5TableWriter
from ctapipe.image import hillas_parameters, leakage
from ctapipe.image.morphology import number_of_islands
from ctapipe.core.container import Container, Field
from ctapipe_io_magic import MAGICEventSource
from utils import MAGIC_Badpixels, MAGIC_Cleaning, timing_parameters

warnings.simplefilter('ignore')

__all__ = ['magic_cal_to_dl1']


class EventInfoContainer(Container):
    
    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    mjd = Field(-1, 'Event time MJD')
    millisec = Field(-1, 'Event time millisec')
    nanosec = Field(-1, 'Event time nanosec')
    alt_tel = Field(-1, 'Telescope pointing altitude', u.deg)
    az_tel = Field(-1, 'Telescope pointing azimuth', u.deg)
    n_islands = Field(-1, 'Number of image islands')
    n_pixels = Field(-1, 'Number of pixels of cleaned images')


def magic_cal_to_dl1(input_files, output_file, config):

    config_cleaning = config['MAGIC']['magic_clean']
    config_badpixels = config['MAGIC']['bad_pixels']

    config_cleaning['findhotpixels'] = True   # True for real data, False for MC data 

    print(f'\nConfiguration for the image cleaning:\n{config_cleaning}')
    print(f'\nConfiguration for the bad pixels calculation:\n{config_badpixels}')

    # --- check the input data ---
    file_paths = glob.glob(input_files)
    file_paths.sort()

    print('\nProcess the following data:')

    telescope_ids = []

    for path in file_paths:

        print(path)
        
        with uproot.open(path) as f:
            tel_id = int(f['RunHeaders']['MRawRunHeader.fTelescopeNumber'].array()[0])
            telescope_ids.append(tel_id)

    if len(set(telescope_ids)) > 1:
        print('\nM1 and M2 data are mixed. Input only M1 or M2 data. Exiting.\n')
        sys.exit()

    # --- process the input data ---
    previous_event_id = -10
    n_events_skipped = 0

    with HDF5TableWriter(filename=output_file, group_name='events', overwrite=True) as writer:
        
        source = MAGICEventSource(input_url=input_files)

        camera_geom = source.subarray.tel[tel_id].camera.geometry
        magic_clean = MAGIC_Cleaning.magic_clean(camera_geom, config_cleaning)
        badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(config=config_badpixels)

        print('\nProcessing the events...')

        mono_event_generator = source._mono_event_generator(telescope=f'M{tel_id}')

        for i_ev, event in enumerate(mono_event_generator):

            if (i_ev % 100) == 0:
                print(f'{i_ev} events')

            if event.index.event_id == previous_event_id:   # exclude pedestal runs??
            
                print(f'--> {i_ev} event (event ID: {event.index.event_id}): ' \
                      'Pedestal event (?) found. Skipping.')
                
                n_events_skipped += 1
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

            n_islands, _ = number_of_islands(camera_geom, signal_pixels)
            n_pixels = np.count_nonzero(signal_pixels)

            if np.all(image_cleaned == 0):
                
                print(f'--> {i_ev} event (event ID: {event.index.event_id}): ' \
                      'Could not survive the image cleaning. Skipping.')
                
                n_events_skipped += 1
                continue
            
            # --- hillas parameters calculation ---
            try:
                hillas_params = hillas_parameters(camera_geom, image_cleaned)

            except:
                
                print(f'--> {i_ev} event (event ID: {event.index.event_id}): ' \
                      'Hillas parameters calculation failed. Skipping.')
                
                n_events_skipped += 1
                continue
                
            # --- timing parameters calculation ---
            try:
                timing_params = timing_parameters(
                    camera_geom, image_cleaned, peak_time_cleaned, hillas_params, signal_pixels
                )

            except:
                
                print(f'--> {i_ev} event (event ID: {event.index.event_id}): ' \
                      'Timing parameters calculation failed. Skipping.')
                
                n_events_skipped += 1
                continue
            
            # --- leakage parameters calculation ---
            try:
                leakage_params = leakage(camera_geom, image, signal_pixels)
                
            except:
                
                print(f'--> {i_ev} event (event ID: {event.index.event_id}): ' \
                      'Leakage parameters calculation failed. Skipping.')

                n_events_skipped += 1
                continue

            # --- save the parameters ---
            event_info = EventInfoContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                tel_id=tel_id,
                mjd=event.trigger.mjd,
                millisec=event.trigger.millisec,
                nanosec=event.trigger.nanosec,
                alt_tel=event.pointing.tel[tel_id].altitude,
                az_tel=event.pointing.tel[tel_id].azimuth,
                n_islands=n_islands,
                n_pixels=n_pixels
            )

            writer.write('params', (event_info, hillas_params, timing_params, leakage_params))

        print(f'{i_ev+1} events processed.')
        print(f'({n_events_skipped} events are skipped)')

    print(f'\nOutput data file: {output_file}')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-files', '-i', dest='input_files', type=str,
        help='Path to input MAGIC calibrated data files (*_Y_*.root). Do not mix M1 and M2 data.'
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./dl1_magic.h5',
        help='Path to an output DL1 data file.'
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a configuration file.'
    )

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file, 'r'))

    magic_cal_to_dl1(args.input_files, args.output_file, config)

    print('\nDone.')

    end_time = time.time()
    print(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()