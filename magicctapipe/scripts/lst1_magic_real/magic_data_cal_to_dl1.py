#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import os
import yaml
import glob
import copy
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from astropy import units as u
from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from ctapipe.image import (
    hillas_parameters,
    leakage_parameters,
    timing_parameters,
)
from ctapipe.image.morphology import number_of_islands
from ctapipe_io_magic import MAGICEventSource
from magicctapipe.utils import MAGIC_Badpixels, MAGIC_Cleaning

warnings.simplefilter('ignore')

__all__ = ['magic_cal_to_dl1']


class InfoContainer(Container):
    
    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    mjd = Field(-1, 'Event time mjd')
    millisec = Field(-1, 'Event time millisec')
    nanosec = Field(-1, 'Event time nanosec')
    alt_tel = Field(-1, 'Telescope pointing altitude', unit=u.rad)
    az_tel = Field(-1, 'Telescope pointing azimuth', unit=u.rad)
    n_islands = Field(-1, 'Number of image islands')


def magic_cal_to_dl1(input_data_mask, output_data, config):

    config_cleaning = config['magic_clean']
    config_badpixels = config['bad_pixels']

    config_cleaning['findhotpixels'] = True

    print(f'\nConfiguration for image cleaning:\n{config_cleaning}')
    print(f'\nConfiguration for bad pixels calculation:\n{config_badpixels}')

    # --- check the input data ---
    paths_list = glob.glob(input_data_mask)
    paths_list.sort()

    # --- process the input data ---
    output_dir = str(Path(output_data).parent)
    os.makedirs(output_dir, exist_ok=True)

    previous_event_id = 0
    n_events_skipped = 0

    with HDF5TableWriter(
        filename=output_data,
        group_name='events',
        overwrite=True
    ) as writer:

        for path in paths_list:
            print(f"\nProcessing {path.name}...")

            source = MAGICEventSource(input_url=input_data_mask)

            tel_id = source.telescope
            is_simulation = source.is_mc

            geom_camera = source.subarray.tel[tel_id].camera.geometry
            magic_clean = MAGIC_Cleaning.magic_clean(geom_camera, config_cleaning)
            badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(
                is_simulation=is_simulation,
                config=config_badpixels,
            )

            print('\nProcessing the events...')

            for i_ev, event in enumerate(source):

                if i_ev % 100 == 0:
                    print(f'{i_ev} events')

                if event.index.event_id == previous_event_id:   # exclude pedestal runs??
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): '
                        f'Pedestal event (?) found. Skipping.')
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

                num_islands, island_labels = number_of_islands(geom_camera, signal_pixels)

                if np.sum(image_cleaned) == 0:
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): '
                        f'Could not survive the image cleaning. Skipping.')
                    n_events_skipped += 1
                    continue

                # --- Hillas parameter calculation ---
                try:
                    hillas_params = hillas_parameters(geom_camera, image_cleaned)

                except:
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): '
                        f'Hillas parameter calculation failed. Skipping.')
                    n_events_skipped += 1
                    continue

                # --- Timing parameter calculation ---
                try:
                    timing_params = timing_parameters(
                        geom_camera, image_cleaned, peak_time_cleaned, hillas_params, signal_pixels
                    )

                except:
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): '
                        f'Timing parameter calculation failed. Skipping.')
                    n_events_skipped += 1
                    continue

                # --- Leakage parameter calculation ---
                try:
                    leakage_params = leakage_parameters(geom_camera, image, signal_pixels)

                except:
                    print(f'--> {i_ev} event (event ID = {event.index.event_id}): '
                        f'Leakage parameter calculation failed. Skipping.')
                    n_events_skipped += 1
                    continue

                # --- save the event information ---
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

        print(f'\n{i_ev+1} events processed.')
        print(f'({n_events_skipped} events are skipped)')

    print(f'\nOutput data: {output_data}')


def main():

    start_time = time.time()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str,
        help='Path to M1 or M2 calibrated data files (*_Y_*.root). Do not mix M1 and M2 data.'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, default='./dl1_magic.h5',
        help='Path to an output data file with h5 extention.'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a configuration file with yaml extention.'
    )

    args = arg_parser.parse_args()

    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    magic_cal_to_dl1(args.input_data, args.output_data, config_lst1_magic['MAGIC'])

    print('\nDone.')
    print(f'\nelapsed time = {time.time() - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
