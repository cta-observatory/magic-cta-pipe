#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

Process the simtel MC DL0 data (*.simtel.gz) containing LST-1 or MAGIC events to DL1.
The allowed telescopes specified in the configuration file will only be processed.
The events that all the DL1 parameters are computed will be saved in the output file.
The telescope IDs are automatically reset to the following values when saving to an output file:
LST-1: tel_id = 1,  MAGIC-I: tel_id = 2,  MAGIC-II: tel_id = 3

Usage:
$ python lst1_magic_mc_dl0_to_dl1.py
--input-file "./data/dl0/gamma_40deg_90deg_run1___cta-prod5-lapalma_LST-1_MAGIC_desert-2158m_mono_off0.4.simtel.gz"
--output-file "./data/dl1/dl1_lst1_magic_gamma_40deg_90deg_off0.4_run1.h5"
--config-file "./config.yaml"
"""

import time
import yaml
import logging
import argparse
import warnings
import numpy as np
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
    leakage_parameters
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from magicctapipe.utils import (
    add_noise_in_pixels,
    set_numba_seed,
    random_psf_smearer,
    apply_dynamic_cleaning,
    MAGIC_Cleaning,
    calc_impact
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

__all__ = ['mc_dl0_to_dl1']


class EventInfoContainer(Container):

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


def mc_dl0_to_dl1(input_file, output_file, config):

    logger.info(f'\nInput data file:\n{input_file}')

    event_source = EventSource(input_file)
    obs_id = event_source.obs_ids[0]

    subarray = event_source.subarray
    logger.info(f'\nSubarray configuration:')

    for tel_id in subarray.tel.keys():
        logger.info(f'Telescope {tel_id}: {subarray.tel[tel_id].name}, position = {subarray.positions[tel_id]}')

    allowed_tels = config['mc_allowed_tels']
    logger.info(f'\nAllowed telescopes:\n{allowed_tels}')

    # --- define the processors ---
    process_lst1_events = ('LST-1' in allowed_tels)
    process_m1_events = ('MAGIC-I' in allowed_tels)
    process_m2_events = ('MAGIC-II' in allowed_tels)

    tel_descriptions = {}
    tel_positions = {}

    if process_lst1_events:

        config_lst = config['LST']
        logger.info(f'\nConfiguration for LST data process:')

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

        calibrator_lst = CameraCalibrator(subarray, image_extractor=extractor_lst)

        tel_id_lst = allowed_tels['LST-1']
        tel_descriptions[1] = subarray.tel[tel_id_lst]
        tel_positions[1] = subarray.positions[tel_id_lst]

    if process_m1_events or process_m2_events:

        config_magic = config['MAGIC']
        config_magic['magic_clean']['findhotpixels'] = False   # False for MC data, True for real data

        logger.info(f'\nConfiguration for MAGIC data process:')

        for key, value in config_magic.items():
            logger.info(f'{key}: {value}')

        extractor_name_magic = config_magic['image_extractor'].pop('name')
        config_extractor_magic = Config({extractor_name_magic: config_magic['image_extractor']})

        extractor_magic = ImageExtractor.from_name(
            extractor_name_magic, subarray=subarray, config=config_extractor_magic
        )

        calibrator_magic = CameraCalibrator(subarray, image_extractor=extractor_magic)

        if process_m1_events:
            tel_id_m1 = allowed_tels['MAGIC-I']
            tel_descriptions[2] = subarray.tel[tel_id_m1]
            tel_positions[2] = subarray.positions[tel_id_m1]

        if process_m2_events:
            tel_id_m2 = allowed_tels['MAGIC-II']
            tel_descriptions[3] = subarray.tel[tel_id_m2]
            tel_positions[3] = subarray.positions[tel_id_m2]

    # --- process the events ---
    with HDF5TableWriter(filename=output_file, group_name='events', overwrite=True) as writer:

        for tel_name, tel_id in allowed_tels.items():

            logger.info(f'\nProcessing the {tel_name} events...')
            event_source = EventSource(input_file, allowed_tels=[tel_id])

            subarray = event_source.subarray
            position = subarray.positions[tel_id]

            camera_geom = subarray.tel[tel_id].camera.geometry
            focal_length = subarray.tel[tel_id].optics.equivalent_focal_length
            camera_frame = CameraFrame(rotation=camera_geom.cam_rotation, focal_length=focal_length)

            if np.any(tel_name == np.array(['MAGIC-I', 'MAGIC-II'])):
                magic_clean = MAGIC_Cleaning.magic_clean(camera_geom, config_magic['magic_clean'])

            magic_stereo = None
            n_events_skipped = 0

            for event in event_source:

                if (event.count % 100) == 0:
                    logger.info(f'{event.count} events')

                if process_m1_events and process_m2_events:

                    trigger_m1 = (tel_id_m1 in event.trigger.tels_with_trigger)
                    trigger_m2 = (tel_id_m2 in event.trigger.tels_with_trigger)

                    magic_stereo = (trigger_m1 & trigger_m2)

                if tel_name == 'LST-1':

                    # --- calibration ---
                    calibrator_lst(event)

                    image = event.dl1.tel[tel_id].image
                    peak_time = event.dl1.tel[tel_id].peak_time

                    # --- image modification ---
                    if increase_nsb:
                        image = add_noise_in_pixels(rng, image, **config_lst['increase_nsb'])

                    if increase_psf:

                        image = random_psf_smearer(
                            image, config_lst['increase_psf']['smeared_light_fraction'],
                            camera_geom.neighbor_matrix_sparse.indices,
                            camera_geom.neighbor_matrix_sparse.indptr
                        )

                    # --- image cleaning ---
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

                    # --- calibration ---
                    calibrator_magic(event)

                    # --- image cleaning ---
                    signal_pixels, image, peak_time = magic_clean.clean_image(
                        event.dl1.tel[tel_id].image, event.dl1.tel[tel_id].peak_time
                    )

                n_islands, _ = number_of_islands(camera_geom, signal_pixels)
                n_pixels = np.count_nonzero(signal_pixels)

                image_cleaned = image.copy()
                image_cleaned[~signal_pixels] = 0

                peak_time_cleaned = peak_time.copy()
                peak_time_cleaned[~signal_pixels] = 0

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

                # --- calculate additional parameters ---
                tel_pointing = AltAz(alt=event.pointing.tel[tel_id].altitude, az=event.pointing.tel[tel_id].azimuth)
                telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

                event_coord = SkyCoord(hillas_params.x, hillas_params.y, frame=camera_frame)
                event_coord = event_coord.transform_to(telescope_frame)

                mc_disp = angular_separation(
                    lon1=event_coord.altaz.az, lat1=event_coord.altaz.alt,
                    lon2=event.simulation.shower.az, lat2=event.simulation.shower.alt
                )

                mc_impact = calc_impact(
                    core_x=event.simulation.shower.core_x, core_y=event.simulation.shower.core_y,
                    az=event.simulation.shower.az, alt=event.simulation.shower.alt,
                    tel_pos_x=position[0], tel_pos_y=position[1], tel_pos_z=position[2]
                )

                # --- save the event information ---
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
                    magic_stereo=magic_stereo
                )

                if tel_name == 'LST-1':
                    event_info.tel_id = 1

                elif tel_name == 'MAGIC-I':
                    event_info.tel_id = 2

                elif tel_name == 'MAGIC-II':
                    event_info.tel_id = 3

                writer.write('params', (event_info, hillas_params, timing_params, leakage_params))

            logger.info(f'{event.count + 1} events processed.')
            logger.info(f'({n_events_skipped} events are skipped)')

    # --- save in the output file ---
    telescope_ids = tel_positions.keys()

    tel_positions = np.array([tel_positions[tel_id].value for tel_id in telescope_ids])
    tel_positions = np.round(tel_positions - tel_positions.mean(axis=0), 2)
    tel_positions = {tel_id: u.Quantity(tel_positions[i_tel], u.m) for i_tel, tel_id in enumerate(telescope_ids)}

    subarray = SubarrayDescription(subarray.name, tel_positions, tel_descriptions)

    logger.info(f'\nSaving the adjusted subarray description:')
    for tel_id in subarray.tel.keys():
        logger.info(f'Telescope {tel_id}: {subarray.tel[tel_id].name}, position = {subarray.positions[tel_id]}')

    subarray.to_hdf(output_file)

    with HDF5TableWriter(filename=output_file, group_name='simulation', mode='a') as writer:
        writer.write('config', event_source.simulation_config)

    logger.info(f'\nOutput data file: {output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str,
        help='Path to an input simtel DL0 data file (*.simtel.gz).'
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./dl1_mc.h5',
        help='Path to an output DL1 data file.'
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.'
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    mc_dl0_to_dl1(args.input_file, args.output_file, config)

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
