#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

Process the MAGIC calibrated data (*_Y_*.root) with MARS-like cleaning method,
and calculate the DL1 parameters (i.e., Hillas, timing and leakage parameters).
The events that all the DL1 parameters are reconstructed will be saved in the output file.
The telescope IDs are automatically reset to the following values when saving to an output file:
MAGIC-I: tel_id = 2,  MAGIC-II: tel_id = 3

The MAGICEventSource automatically searches for all the subrun files with the same observation ID
of the input file existing in the input directoly, and reads the drive reports from them.
Then, if the "process_run" option is True, the MAGICEventSource not only reads the drive reports
but also process all the subrun files together with the input subrun file.

Usage:
$ python magic_cal_to_dl1.py
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
    leakage_parameters,
)
from ctapipe.instrument import SubarrayDescription
from magicctapipe.image import MAGICClean
from ctapipe_io_magic import MAGICEventSource

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

sec2nsec = 1e9

__all__ = ['cal_to_dl1']


class EventInfoContainer(Container):
    """
    Store general event information:
    - observation/event/telescope IDs
    - telescope pointing parameters
    - parameters of cleaned image
    """

    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    alt_tel = Field(-1, 'Telescope pointing altitude', u.rad)
    az_tel = Field(-1, 'Telescope pointing azimuth', u.rad)
    n_islands = Field(-1, 'Number of islands of cleaned image')
    n_pixels = Field(-1, 'Number of pixels of cleaned image')


class TimingInfoContainer(Container):
    """ Store the event timing information """

    time_sec = Field(-1, 'Event time second')
    time_nanosec = Field(-1, 'Event time nanosecond')


class SimInfoContainer(Container):
    """ Store the simulated event information """

    mc_energy = Field(-1, 'Event MC energy', u.TeV)
    mc_alt = Field(-1, 'Event MC altitude', u.deg)
    mc_az = Field(-1, 'Event MC azimuth', u.deg)
    mc_core_x = Field(-1, 'Event MC core X', u.m)
    mc_core_y = Field(-1, 'Event MC core Y', u.m)


def cal_to_dl1(input_file, output_file, config):
    """
    Process the MAGIC calibrated level data to DL1.

    Parameters
    ----------
    input_file: str
        Path to an input MAGIC calibrated data file
    output_file: str
        Path to an output DL1 data file
    config: dict
        Configuration of the data process
    """

    process_run = config['MAGIC']['process_run']

    event_source = MAGICEventSource(
        input_url=input_file,
        process_run=process_run,
    )

    subarray = event_source.subarray
    tel_id = event_source.telescope
    is_simulation = event_source.is_simulation

    logger.info(f'\nProcess the following subrun file(s) (process_run = {process_run}):')
    for root_file in event_source.file_list:
        logger.info(root_file)

    config_cleaning = config['MAGIC']['magic_clean']

    # Check the "find_hotpixels" option:
    if is_simulation and config_cleaning['find_hotpixels'] is not False:
        logger.warning('\nThe hot pixels do not exist in simulation. Setting the option to False...')
        config_cleaning.update({'find_hotpixels': False})

    elif config_cleaning['find_hotpixels'] == 'auto':
        config_cleaning.update({'find_hotpixels': True})

    logger.info(f'\nConfiguration for the image cleaning:\n{config_cleaning}\n')

    # Initialize the MAGIC cleaning:
    camera_geom = subarray.tel[tel_id].camera.geometry
    magic_clean = MAGICClean(camera_geom, config_cleaning)

    # Start processing the events:
    with HDF5TableWriter(
        filename=output_file,
        group_name='events',
        overwrite=True,
    ) as writer:

        n_events_skipped = 0

        for event in event_source:

            if (event.count % 100) == 0:
                logger.info(f'{event.count} events')

            # Cleaning the image:
            if is_simulation:
                signal_pixels, image, peak_time = magic_clean.clean_image(
                    event_image=event.dl1.tel[tel_id].image,
                    event_pulse_time=event.dl1.tel[tel_id].peak_time,
                )
            else:
                dead_pixels = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels
                badrms_pixels = event.mon.tel[tel_id].pixel_status.pedestal_failing_pixels[2]
                unsuitable_mask = np.logical_or(dead_pixels, badrms_pixels)

                signal_pixels, image, peak_time = magic_clean.clean_image(
                    event_image=event.dl1.tel[tel_id].image,
                    event_pulse_time=event.dl1.tel[tel_id].peak_time,
                    unsuitable_mask=unsuitable_mask,
                )

            image_cleaned = image.copy()
            image_cleaned[~signal_pixels] = 0

            peak_time_cleaned = peak_time.copy()
            peak_time_cleaned[~signal_pixels] = 0

            n_islands, _ = number_of_islands(camera_geom, signal_pixels)
            n_pixels = np.count_nonzero(signal_pixels)

            if np.all(image_cleaned == 0):
                logger.warning(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                               'Could not survive the image cleaning. Skipping.')
                n_events_skipped += 1
                continue

            # Hillas parameters calculation:
            try:
                hillas_params = hillas_parameters(camera_geom, image_cleaned)
            except:
                logger.warning(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                               'Hillas parameters calculation failed. Skipping.')
                n_events_skipped += 1
                continue

            # Timing parameters calculation:
            try:
                timing_params = timing_parameters(
                    camera_geom, image_cleaned, peak_time_cleaned, hillas_params, signal_pixels
                )
            except:
                logger.warning(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                               'Timing parameters calculation failed. Skipping.')
                n_events_skipped += 1
                continue

            # Leakage parameters calculation:
            try:
                leakage_params = leakage_parameters(camera_geom, image_cleaned, signal_pixels)
            except:
                logger.warning(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                               'Leakage parameters calculation failed. Skipping.')
                n_events_skipped += 1
                continue

            # Save the event info:
            event_info = EventInfoContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                alt_tel=event.pointing.tel[tel_id].altitude,
                az_tel=event.pointing.tel[tel_id].azimuth,
                n_islands=n_islands,
                n_pixels=n_pixels,
            )

            # The MAGIC telescope IDs are reset to the following values for the
            # convenience of the combined analysis with LST-1, which has telescope ID 1.

            if tel_id == 1:
                event_info.tel_id = 2

            elif tel_id == 2:
                event_info.tel_id = 3

            if is_simulation:

                sim_info = SimInfoContainer(
                    mc_energy=event.simulation.shower.energy,
                    mc_alt=event.simulation.shower.alt,
                    mc_az=event.simulation.shower.az,
                    mc_core_x=event.simulation.shower.core_x,
                    mc_core_y=event.simulation.shower.core_y,
                )

                writer.write('params', (event_info, sim_info, hillas_params, timing_params, leakage_params))

            else:
                # The integral and fractional part of the timestamps are separately stored
                # as "time_sec" and "time_nanosec", respectively, to keep the precision.

                timestamp = event.trigger.tel[tel_id].time.to_value(format='unix', subfmt='long')
                fractional, integral = np.modf(timestamp)

                time_sec = np.round(integral).astype(int)
                time_nanosec = np.round(fractional * sec2nsec, decimals=-2).astype(int)

                time_info = TimingInfoContainer(
                    time_sec=time_sec,
                    time_nanosec=time_nanosec,
                )

                writer.write('params', (event_info, time_info, hillas_params, timing_params, leakage_params))

        n_events_processed = event.count + 1

        logger.info(f'{n_events_processed} events processed.')
        logger.info(f'({n_events_skipped} events are skipped)')

    # Save the subarray description:
    tel_descriptions = {
        2: subarray.tel[1],   # MAGIC-I
        3: subarray.tel[2],   # MAGIC-II
    }

    # In case of real data, the updated telescope positions are set which are
    # precisely measured recently and are used also for the sim_telarray simulation.

    if is_simulation:
        tel_positions = {
            2: subarray.positions[1],   # MAGIC-I
            3: subarray.positions[2],   # MAGIC-II
        }
    else:
        tel_positions = {
            2: u.Quantity([35.25, -23.99, -0.58], u.m),   # MAGIC-I
            3: u.Quantity([-35.25, 23.99, 0.58], u.m),    # MAGIC-II
        }

    subarray = SubarrayDescription(subarray.name, tel_positions, tel_descriptions)
    subarray.to_hdf(output_file)

    logger.info(f'\nOutput data file:\n{output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str,
        help='Path to an input MAGIC calibrated data file (*_Y_*.root).',
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./dl1_magic.h5',
        help='Path to an output DL1 data file.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    cal_to_dl1(args.input_file, args.output_file, config)

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
