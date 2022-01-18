#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

Process the MAGIC calibrated data (*_Y_*.root) with MARS-like cleaning method,
and compute the DL1 parameters (i.e., Hillas, timing and leakage parameters).
The events that all the DL1 parameters are computed will be saved in the output file.
The telescope IDs are automatically reset to the following values when saving to an output file:
MAGIC-I: tel_id = 2,  MAGIC-II: tel_id = 3

Please note that currently only one subrun file is allowed for input data,
and when the input data contains only one drive report, the script stops
showing an error about the interpolation of pointing direction.
This issue will be solved in the coming release of ctapipe_io_magic.

Usage:
$ python magic_data_cal_to_dl1.py
--input-file "./data/calibrated/20201216_M1_05093711.001_Y_CrabNebula-W0.40+035.root"
--output-file "./data/dl1/dl1_M1_run05093711.001.h5"
--config-file "./config.yaml"
"""

import time
import yaml
import logging
import argparse
import warnings
import numpy as np
from astropy import units as u

from ctapipe.io import HDF5TableWriter
from ctapipe.core import Container, Field
from ctapipe.image import (
    number_of_islands,
    hillas_parameters,
    timing_parameters,
    leakage_parameters
)
from ctapipe.instrument import SubarrayDescription
from magicctapipe.utils import MAGIC_Cleaning, MAGICBadPixelsCalc
from ctapipe_io_magic import MAGICEventSource

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

sec2nsec = 1e9

__all__ = ['magic_cal_to_dl1']


class EventInfoContainer(Container):

    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    time_sec = Field(-1, 'Event time second')
    time_nanosec = Field(-1, 'Event time nanosecond')
    alt_tel = Field(-1, 'Telescope pointing altitude', u.rad)
    az_tel = Field(-1, 'Telescope pointing azimuth', u.rad)
    n_islands = Field(-1, 'Number of image islands')
    n_pixels = Field(-1, 'Number of pixels of cleaned images')


def magic_cal_to_dl1(input_file, output_file, config):

    config_cleaning = config['MAGIC']['magic_clean']
    config_badpixels = config['MAGIC']['bad_pixels']

    config_cleaning['findhotpixels'] = True   # True for real data, False for MC data

    logger.info(f'\nConfiguration for the image cleaning:\n{config_cleaning}')
    logger.info(f'\nConfiguration for the bad pixels calculation:\n{config_badpixels}')

    event_source = MAGICEventSource(input_url=input_file)
    subarray = event_source.subarray

    tel_id = event_source.telescope
    is_simulation = event_source.is_mc

    camera_geom = subarray.tel[tel_id].camera.geometry
    magic_clean = MAGIC_Cleaning.magic_clean(camera_geom, config_cleaning)
    badpixel_calculator = MAGICBadPixelsCalc(
        is_simulation=is_simulation,
        camera=camera_geom,
        config=config_badpixels
    )

    # --- process the input data ---
    n_events_skipped = 0
    logger.info('\nProcessing the events...')

    with HDF5TableWriter(filename=output_file, group_name='events', overwrite=True) as writer:

        for event in event_source:

            if (event.count % 100) == 0:
                logger.info(f'{event.count} events')

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

                logger.info(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                            'Could not survive the image cleaning. Skipping.')

                n_events_skipped += 1
                continue

            # --- hillas parameters calculation ---
            try:
                hillas_params = hillas_parameters(camera_geom, image_cleaned)

            except:
                logger.info(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                            'Hillas parameters calculation failed. Skipping.')

                n_events_skipped += 1
                continue

            # --- timing parameters calculation ---
            try:
                timing_params = timing_parameters(
                    camera_geom, image_cleaned, peak_time_cleaned, hillas_params, signal_pixels
                )

            except:
                logger.info(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                            'Timing parameters calculation failed. Skipping.')

                n_events_skipped += 1
                continue

            # --- leakage parameters calculation ---
            try:
                leakage_params = leakage_parameters(camera_geom, image_cleaned, signal_pixels)

            except:
                logger.info(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                            'Leakage parameters calculation failed. Skipping.')

                n_events_skipped += 1
                continue

            # --- save the parameters ---
            timestamp = event.trigger.tel[tel_id].time.to_value(format='unix', subfmt='long')

            time_sec = np.round(np.modf(timestamp)[1])
            time_nanosec = np.round(np.modf(timestamp)[0] * sec2nsec, decimals=-2)

            event_info = EventInfoContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                time_sec=int(time_sec),
                time_nanosec=int(time_nanosec),
                alt_tel=event.pointing.tel[tel_id].altitude,
                az_tel=event.pointing.tel[tel_id].azimuth,
                n_islands=n_islands,
                n_pixels=n_pixels
            )

            if tel_id == 1:
                event_info.tel_id = 2   # MAGIC-I tel_id: 1 -> 2

            elif tel_id == 2:
                event_info.tel_id = 3   # MAGIC-II tel_id: 2 -> 3

            writer.write('params', (event_info, hillas_params, timing_params, leakage_params))

        logger.info(f'{event.count + 1} events processed.')
        logger.info(f'({n_events_skipped} events are skipped)')

    # --- save the subarray description ---
    tel_positions = {
        2: u.Quantity([35.25, -23.99, -0.58], u.m),
        3: u.Quantity([-35.25, 23.99, 0.58], u.m)
    }

    tel_descriptions = {tel_id + 1: subarray.tel[tel_id] for tel_id in subarray.tel.keys()}

    subarray = SubarrayDescription(subarray.name, tel_positions, tel_descriptions)
    subarray.to_hdf(output_file)

    logger.info(f'\nOutput data file: {output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str,
        help='Path to an input MAGIC calibrated data file (*_Y_*.root).'
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./dl1_magic.h5',
        help='Path to an output DL1 data file.'
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.'
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    magic_cal_to_dl1(args.input_file, args.output_file, config)

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
