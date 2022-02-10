#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script processes simtel MC DL0 data (*.simtel.gz) containing LST-1 and MAGIC events
and compute DL1 parameters (i.e., Hillas, timing, and leakage parameters).
Only the events that all the DL1 parameters are computed will be saved in an output file.
Telescope IDs are automatically reset to the following values when saving to the output file:
LST-1: tel_id = 1,  MAGIC-I: tel_id = 2,  MAGIC-II: tel_id = 3

Usage:
$ python lst1_magic_mc_dl0_to_dl1.py
--input-file ./data/dl0/gamma_40deg_90deg_run1___cta-prod5-lapalma_LST-1_MAGIC_desert-2158m_mono_off0.4.simtel.gz
--output-dir ./data/dl1
--config-file ./config.yaml
"""

import re
import time
import yaml
import logging
import argparse
import warnings
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation
from traitlets.config import Config
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.core import Container, Field
from ctapipe.calib import CameraCalibrator
from ctapipe.image import (
    ImageExtractor,
    tailcuts_clean,
    apply_time_delta_cleaning,
    number_of_islands,
    hillas_parameters,
    timing_parameters,
    leakage_parameters,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from lstchain.image.cleaning import apply_dynamic_cleaning
from lstchain.image.modifier import (
    add_noise_in_pixels,
    set_numba_seed,
    random_psf_smearer,
)
from magicctapipe.image import MAGICClean
from magicctapipe.utils import calc_impact

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

__all__ = [
    'mc_dl0_to_dl1',
]


class EventInfoContainer(Container):
    """
    Container to store general event information:
    - observation/event/telescope IDs
    - telescope pointing direction
    - simulated event parameters
    - parameters of cleaned image
    - flag to magic-stereo trigger
    """

    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    alt_tel = Field(-1, 'Telescope altitude', u.rad)
    az_tel = Field(-1, 'Telescope azimuth', u.rad)
    mc_energy = Field(-1, 'MC event energy', u.TeV)
    mc_alt = Field(-1, 'MC event altitude', u.deg)
    mc_az = Field(-1, 'MC event azimuth', u.deg)
    mc_disp = Field(-1, 'MC event disp', u.deg)
    mc_core_x = Field(-1, 'MC core x', u.m)
    mc_core_y = Field(-1, 'MC core y', u.m)
    mc_impact = Field(-1, 'MC impact', u.m)
    n_islands = Field(-1, 'Number of image islands')
    n_pixels = Field(-1, 'Number of pixels of cleaned images')
    magic_stereo = Field(-1, 'True if M1 and M2 are triggered')


def mc_dl0_to_dl1(input_file, output_dir, config):
    """
    This function processes simtel DL0 data to DL1.

    Parameters
    ----------
    input_file: str
        Path to an input simtel DL0 data file
    output_dir: str
        Path to a directory where to save an output DL1 data file
    config: dict
        Configuration for data processes
    """

    logger.info(f'\nInput data file:\n{input_file}')

    event_source = EventSource(input_file)
    obs_id = event_source.obs_ids[0]

    subarray = event_source.subarray
    logger.info('\nSubarray configuration:')

    for tel_id in subarray.tel.keys():
        logger.info(f'Telescope {tel_id}: {subarray.tel[tel_id].name}, ' \
                    f'position = {subarray.positions[tel_id]}')

    mc_tel_ids = config['mc_tel_ids']
    logger.info(f'\nThe telescope IDs that will be processed:\n{mc_tel_ids}')

    tel_id_lst = mc_tel_ids['LST-1']
    tel_id_m1 = mc_tel_ids['MAGIC-I']
    tel_id_m2 = mc_tel_ids['MAGIC-II']

    # Configure LST data processes.
    # Process events with the method used in lstchain:
    config_lst = config['LST']
    logger.info('\nConfiguration for LST data process:')

    for key, value in config_lst.items():
        logger.info(f'{key}: {value}')

    increase_nsb = config_lst['increase_nsb'].pop('use')
    increase_psf = config_lst['increase_psf'].pop('use')

    if increase_nsb:
        rng = np.random.default_rng(obs_id)

    if increase_psf:
        set_numba_seed(obs_id)

    use_time_delta_cleaning = config_lst['time_delta_cleaning'].pop('use')
    use_dynamic_cleaning = config_lst['dynamic_cleaning'].pop('use')
    use_only_main_island = config_lst['use_only_main_island']

    extractor_name_lst = config_lst['image_extractor'].pop('name')
    config_extractor_lst = Config({extractor_name_lst: config_lst['image_extractor']})

    extractor_lst = ImageExtractor.from_name(
        extractor_name_lst, subarray=subarray, config=config_extractor_lst
    )

    calibrator_lst = CameraCalibrator(
        subarray, image_extractor=extractor_lst
    )

    # Configure MAGIC data processes.
    # The cleaning method will be defined later once a camera geometry is determined:
    config_magic = config['MAGIC']

    if config_magic['magic_clean']['find_hotpixels'] is not False:
        logger.warning('\nHot pixels do not exist in a simulation. Setting the option to False...')
        config_magic['magic_clean'].update({'find_hotpixels': False})

    logger.info('\nConfiguration for MAGIC data process:')

    for key, value in config_magic.items():
        if key != 'process_run':
            logger.info(f'{key}: {value}')

    extractor_name_magic = config_magic['image_extractor'].pop('name')
    config_extractor_magic = Config({extractor_name_magic: config_magic['image_extractor']})

    extractor_magic = ImageExtractor.from_name(
        extractor_name_magic, subarray=subarray, config=config_extractor_magic
    )

    calibrator_magic = CameraCalibrator(
        subarray, image_extractor=extractor_magic
    )

    # Prepare for saving data to an output file.
    # The output directory will be created if it doesn't exist:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file).resolve().name
    regex_mc_offset = r'(\w+)_run\d+_.*off(\S+)\.simtel.gz'
    regex_mc = r'(\w+)_run\d+_.*\.simtel.gz'

    if re.fullmatch(regex_mc_offset, base_name):
        parser = re.findall(regex_mc_offset, base_name)[0]
        output_file = f'{output_dir}/dl1_lst1_magic_{parser[0]}_off{parser[1]}_run{obs_id}.h5'

    elif re.fullmatch(regex_mc, base_name):
        parser = re.findall(regex_mc, base_name)[0]
        output_file = f'{output_dir}/dl1_lst1_magic_{parser}_run{obs_id}.h5'

    else:
        logger.warning('\nCould not parse information from the input file name. '\
                       'The output file will be simply named with the observation ID.')
        output_file = f'{output_dir}/dl1_lst1_magic_run{obs_id}.h5'

    # Start processing events:
    with HDF5TableWriter(filename=output_file, group_name='events') as writer:

        for tel_name, tel_id in mc_tel_ids.items():

            logger.info(f'\nProcessing {tel_name} events...')

            tel_position = subarray.positions[tel_id]
            camera_geom = subarray.tel[tel_id].camera.geometry
            focal_length = subarray.tel[tel_id].optics.equivalent_focal_length

            camera_frame = CameraFrame(
                rotation=camera_geom.cam_rotation,
                focal_length=focal_length,
            )

            # Configure the MAGIC Cleaning:
            if np.any(tel_name == np.array(['MAGIC-I', 'MAGIC-II'])):
                magic_clean = MAGICClean(camera_geom, config_magic['magic_clean'])

            magic_stereo = None
            n_events_skipped = 0

            event_source_per_tel = EventSource(input_file, allowed_tels=[tel_id])

            for event in event_source_per_tel:

                if (event.count % 100) == 0:
                    logger.info(f'{event.count} events')

                trigger_m1 = (tel_id_m1 in event.trigger.tels_with_trigger)
                trigger_m2 = (tel_id_m2 in event.trigger.tels_with_trigger)

                magic_stereo = (trigger_m1 and trigger_m2)

                if tel_name == 'LST-1':

                    # Calibrate the event:
                    calibrator_lst(event)

                    image = event.dl1.tel[tel_id].image
                    peak_time = event.dl1.tel[tel_id].peak_time

                    if increase_nsb:
                        # Add noise in pixels:
                        image = add_noise_in_pixels(rng, image, **config_lst['increase_nsb'])

                    if increase_psf:
                        # Smear the image:
                        image = random_psf_smearer(
                            image,
                            config_lst['increase_psf']['smeared_light_fraction'],
                            camera_geom.neighbor_matrix_sparse.indices,
                            camera_geom.neighbor_matrix_sparse.indptr,
                        )

                    # Apply the image cleaning:
                    signal_pixels = tailcuts_clean(camera_geom, image, **config_lst['tailcuts_clean'])

                    if use_time_delta_cleaning:
                        signal_pixels = apply_time_delta_cleaning(
                            camera_geom, signal_pixels, peak_time, **config_lst['time_delta_cleaning']
                        )

                    if use_dynamic_cleaning:
                        signal_pixels = apply_dynamic_cleaning(
                            image, signal_pixels, **config_lst['dynamic_cleaning']
                        )

                    if use_only_main_island:
                        _, island_labels = number_of_islands(camera_geom, signal_pixels)
                        n_pixels_on_island = np.bincount(island_labels.astype(np.int64))
                        n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
                        max_island_label = np.argmax(n_pixels_on_island)
                        signal_pixels[island_labels != max_island_label] = False

                elif np.any(tel_name == np.array(['MAGIC-I', 'MAGIC-II'])):

                    # Calibrate the event:
                    calibrator_magic(event)

                    # Apply the image cleaning:
                    signal_pixels, image, peak_time = magic_clean.clean_image(
                        event_image=event.dl1.tel[tel_id].image,
                        event_pulse_time=event.dl1.tel[tel_id].peak_time,
                    )

                n_islands, _ = number_of_islands(camera_geom, signal_pixels)
                n_pixels = np.count_nonzero(signal_pixels)

                image_cleaned = image.copy()
                image_cleaned[~signal_pixels] = 0

                peak_time_cleaned = peak_time.copy()
                peak_time_cleaned[~signal_pixels] = 0

                if np.all(image_cleaned == 0):
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

                # Compute the DISP parameter:
                tel_pointing = AltAz(
                    alt=event.pointing.tel[tel_id].altitude,
                    az=event.pointing.tel[tel_id].azimuth,
                )

                telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

                event_coord = SkyCoord(hillas_params.x, hillas_params.y, frame=camera_frame)
                event_coord = event_coord.transform_to(telescope_frame)

                mc_disp = angular_separation(
                    lon1=event_coord.altaz.az,
                    lat1=event_coord.altaz.alt,
                    lon2=event.simulation.shower.az,
                    lat2=event.simulation.shower.alt,
                )

                # Compute the impact parameter:
                mc_impact = calc_impact(
                    core_x=event.simulation.shower.core_x,
                    core_y=event.simulation.shower.core_y,
                    az=event.simulation.shower.az,
                    alt=event.simulation.shower.alt,
                    tel_pos_x=tel_position[0],
                    tel_pos_y=tel_position[1],
                    tel_pos_z=tel_position[2],
                )

                # Set the general event information:
                event_info = EventInfoContainer(
                    obs_id=event.index.obs_id,
                    event_id=event.index.event_id,
                    alt_tel=event.pointing.tel[tel_id].altitude,
                    az_tel=event.pointing.tel[tel_id].azimuth,
                    mc_energy=event.simulation.shower.energy,
                    mc_alt=event.simulation.shower.alt,
                    mc_az=event.simulation.shower.az,
                    mc_disp=mc_disp,
                    mc_core_x=event.simulation.shower.core_x,
                    mc_core_y=event.simulation.shower.core_y,
                    mc_impact=mc_impact,
                    n_islands=n_islands,
                    n_pixels=n_pixels,
                    magic_stereo=magic_stereo,
                )

                # The telescope IDs are reset to the following values to match them with real data:
                if tel_name == 'LST-1':
                    event_info.tel_id = 1

                elif tel_name == 'MAGIC-I':
                    event_info.tel_id = 2

                elif tel_name == 'MAGIC-II':
                    event_info.tel_id = 3

                writer.write('params', (event_info, hillas_params, timing_params, leakage_params))

            n_events_processed = event.count + 1

            logger.info(f'{n_events_processed} events processed.')
            logger.info(f'({n_events_skipped} events skipped)')

    # Save the subarray description.
    # The telescope positions are adjusted to CoG coordinate:
    positions = np.array([subarray.positions[tel_id].value for tel_id in mc_tel_ids.values()])
    positions = np.round(positions - positions.mean(axis=0), 2)

    tel_positions_cog = {
        1: u.Quantity(positions[0], u.m),   # LST-1
        2: u.Quantity(positions[1], u.m),   # MAGIC-I
        3: u.Quantity(positions[2], u.m),   # MAGIC-II
    }

    tel_descriptions = {
        1: subarray.tel[tel_id_lst],   # LST-1
        2: subarray.tel[tel_id_m1],    # MAGIC-I
        3: subarray.tel[tel_id_m2],    # MAGIC-II
    }

    subarray_cog = SubarrayDescription(subarray.name, tel_positions_cog, tel_descriptions)

    logger.info('\nSaving the subarray description in CoG coordinate:')
    for tel_id in subarray_cog.tel.keys():
        logger.info(f'Telescope {tel_id}: {subarray_cog.tel[tel_id].name}, ' \
                    f'position = {subarray_cog.positions[tel_id]}')

    subarray_cog.to_hdf(output_file)

    # Save the simulation configuration:
    with HDF5TableWriter(filename=output_file, group_name='simulation', mode='a') as writer:
        writer.write('config', event_source.simulation_config)

    logger.info(f'\nOutput data file:\n{output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str, required=True,
        help='Path to an input simtel DL0 data file (*.simtel.gz).'
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output DL1 data file.'
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.'
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    mc_dl0_to_dl1(args.input_file, args.output_dir, config)

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
