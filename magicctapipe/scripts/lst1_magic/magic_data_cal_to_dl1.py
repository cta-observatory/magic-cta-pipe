#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script processes MAGIC calibrated level data (*_Y_*.root) with the MARS-like
cleaning method and compute DL1 parameters (i.e., Hillas, timing, and leakage parameters).
Only the events that all the DL1 parameters are reconstructed will be saved in an output file.
Telescope IDs are automatically reset to the following values when saving to the output file:
MAGIC-I: tel_id = 2,  MAGIC-II: tel_id = 3

All sub-run files that belong to the same observation ID of an input sub-run file will be automatically
searched for by the MAGICEventSource module, and drive reports are read from all the sub-run files.
If the "process_run" option is set to True, the MAGICEventSource not only reads the drive reports
but also processes all the sub-run files together with the input sub-run file.

Usage:
$ python magic_cal_to_dl1.py
--input-file ./data/calibrated/20201216_M1_05093711.001_Y_CrabNebula-W0.40+035.root
--output-dir ./data/dl1
--config-file ./config.yaml
"""

import time
import yaml
import logging
import argparse
import warnings
import numpy as np
from pathlib import Path
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

tel_positions_simtel = {
    2: u.Quantity([35.25, -23.99, -0.58], u.m),   # MAGIC-I
    3: u.Quantity([-35.25, 23.99, 0.58], u.m),    # MAGIC-II
}

__all__ = [
    'cal_to_dl1',
]


class EventInfoContainer(Container):
    """ Container to store general event information """

    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    alt_tel = Field(-1, 'Telescope pointing altitude', u.rad)
    az_tel = Field(-1, 'Telescope pointing azimuth', u.rad)
    n_islands = Field(-1, 'Number of islands of cleaned image')
    n_pixels = Field(-1, 'Number of pixels of cleaned image')


class TimingInfoContainer(Container):
    """ Container to store event timing information """

    time_sec = Field(-1, 'Event time second')
    time_nanosec = Field(-1, 'Event time nanosecond')


class SimInfoContainer(Container):
    """ Container to store simulated event information """

    mc_energy = Field(-1, 'Event MC energy', u.TeV)
    mc_alt = Field(-1, 'Event MC altitude', u.deg)
    mc_az = Field(-1, 'Event MC azimuth', u.deg)
    mc_core_x = Field(-1, 'Event MC core X', u.m)
    mc_core_y = Field(-1, 'Event MC core Y', u.m)


def cal_to_dl1(input_file, output_dir, config):
    """
    This function processes MAGIC calibrated level data to DL1.

    Parameters
    ----------
    input_file: str
        Path to an input MAGIC calibrated data file
    output_dir: str
        Path to a directory where to save an output DL1 data file
    config: dict
        Configuration for data processes
    """

    process_run = config['MAGIC']['process_run']
    event_source = MAGICEventSource(input_url=input_file, process_run=process_run)

    subarray = event_source.subarray
    is_simulation = event_source.is_simulation

    obs_id = event_source.obs_ids[0]
    tel_id = event_source.telescope

    logger.info(f'\nProcessing the following sub-run file(s) (process_run = {process_run}):')
    for root_file in event_source.file_list:
        logger.info(root_file)

    config_cleaning = config['MAGIC']['magic_clean']

    # Check the "find_hotpixels" option:
    if is_simulation and config_cleaning['find_hotpixels'] is not False:
        logger.warning('\nHot pixels do not exist in a simulation. Setting the option to False...')
        config_cleaning.update({'find_hotpixels': False})

    elif config_cleaning['find_hotpixels'] == 'auto':
        config_cleaning.update({'find_hotpixels': True})

    logger.info(f'\nConfiguration for image cleaning:\n{config_cleaning}\n')

    # Configure the MAGIC cleaning:
    camera_geom = subarray.tel[tel_id].camera.geometry
    magic_clean = MAGICClean(camera_geom, config_cleaning)

    # Prepare for saving data to an output file.
    # The output directory will be created if it doesn't exist:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if is_simulation:
        sim_config = event_source.simulation_config[obs_id]
        zd_min = np.round(90 - sim_config.max_alt.to(u.deg).value).astype(int)
        zd_max = np.round(90 - sim_config.min_alt.to(u.deg).value).astype(int)
        # Assume MAGIC standard gamma MC:
        output_file = f'{output_dir}/dl1_m{tel_id}_gamma_za{zd_min}to{zd_max}_run{obs_id}.h5'

    else:
        metadata = event_source.metadata
        source_name = metadata['source_name'][0]

        if process_run:
            output_file = f'{output_dir}/dl1_m{tel_id}_{source_name}_run{obs_id:08}.h5'
        else:
            subrun_id = metadata['subrun_number'][0]
            output_file = f'{output_dir}/dl1_m{tel_id}_{source_name}_run{obs_id:08}.{subrun_id:03}.h5'

    # Start processing events:
    with HDF5TableWriter(filename=output_file, group_name='events') as writer:

        n_events_skipped = 0

        for event in event_source:

            if (event.count % 100) == 0:
                logger.info(f'{event.count} events')

            # Apply the image cleaning:
            if is_simulation:
                signal_pixels, image, peak_time = magic_clean.clean_image(
                    event_image=event.dl1.tel[tel_id].image,
                    event_pulse_time=event.dl1.tel[tel_id].peak_time,
                )
            else:
                dead_pixels = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[0]
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

            if n_pixels == 0:
                logger.warning(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                               'Could not survive the image cleaning. Skipping.')
                n_events_skipped += 1
                continue

            # Try to compute the Hillas parameters:
            try:
                hillas_params = hillas_parameters(camera_geom, image_cleaned)
            except:
                logger.warning(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                               'Hillas parameters computation failed. Skipping.')
                n_events_skipped += 1
                continue

            # Try to compute the timing parameters:
            try:
                timing_params = timing_parameters(
                    camera_geom, image_cleaned, peak_time_cleaned, hillas_params, signal_pixels
                )
            except:
                logger.warning(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                               'Timing parameters computation failed. Skipping.')
                n_events_skipped += 1
                continue

            # Try to compute the leakage parameters:
            try:
                leakage_params = leakage_parameters(camera_geom, image_cleaned, signal_pixels)
            except:
                logger.warning(f'--> {event.count} event (event ID: {event.index.event_id}): ' \
                               'Leakage parameters computation failed. Skipping.')
                n_events_skipped += 1
                continue

            # Set the general event information:
            event_info = EventInfoContainer(
                obs_id=event.index.obs_id,
                event_id=event.index.event_id,
                alt_tel=event.pointing.tel[tel_id].altitude,
                az_tel=event.pointing.tel[tel_id].azimuth,
                n_islands=n_islands,
                n_pixels=n_pixels,
            )

            # The telescope IDs are reset to the following values for the convenience
            # of the combined analysis with LST-1, which has the telescope ID 1:
            if tel_id == 1:
                event_info.tel_id = 2   # MAGIC-I

            elif tel_id == 2:
                event_info.tel_id = 3   # MAGIC-II

            if is_simulation:
                # Set the simulated event information:
                sim_info = SimInfoContainer(
                    mc_energy=event.simulation.shower.energy,
                    mc_alt=event.simulation.shower.alt,
                    mc_az=event.simulation.shower.az,
                    mc_core_x=event.simulation.shower.core_x,
                    mc_core_y=event.simulation.shower.core_y,
                )

                writer.write('params', (event_info, sim_info, hillas_params, timing_params, leakage_params))

            else:
                # Set the event timing information.
                # The integral and fractional part of timestamp are separately saved
                # as "time_sec" and "time_nanosec", respectively, to keep the precision:
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
        logger.info(f'({n_events_skipped} events skipped)')

    # Save the subarray description:
    tel_descriptions = {
        2: subarray.tel[1],   # MAGIC-I
        3: subarray.tel[2],   # MAGIC-II
    }

    if is_simulation:
        # Save the positions used in MAGIC standard MCs:
        tel_positions = {
            2: subarray.positions[1],   # MAGIC-I
            3: subarray.positions[2],   # MAGIC-II
        }
    else:
        # In case of real data, the updated telescope positions are saved which are
        # precisely measured recently and are also used for sim_telarray simulations:
        tel_positions = tel_positions_simtel

    subarray = SubarrayDescription('MAGIC', tel_positions, tel_descriptions)
    subarray.to_hdf(output_file)

    if is_simulation:
        # Save the simulation configuration:
        with HDF5TableWriter(filename=output_file, group_name='simulation', mode='a') as writer:
            writer.write('config', sim_config)

    logger.info(f'\nOutput data file:\n{output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str, required=True,
        help='Path to an input MAGIC calibrated data file (*_Y_*.root).',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an ouptut DL1 data file.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    cal_to_dl1(args.input_file, args.output_dir, config)

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
