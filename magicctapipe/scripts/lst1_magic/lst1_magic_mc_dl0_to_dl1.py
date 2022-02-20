#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script processes simtel MC DL0 data (*.simtel.gz) containing LST-1 and MAGIC events
and computes the DL1 parameters (i.e., Hillas, timing and leakage parameters).
The script saves an event in an output file only when it reconstructs all the DL1 parameters.
The telescope IDs are reset to the following values when saving to the file:
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
from astropy.coordinates import (
    AltAz,
    SkyCoord,
    angular_separation,
)
from traitlets.config import Config
from ctapipe.io import EventSource, HDF5TableWriter
from ctapipe.core import Container, Field
from ctapipe.calib import CameraCalibrator
from ctapipe.image import (
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
    Container to store the following event information:
    - observation/event/telescope IDs
    - telescope pointing direction
    - simulated event parameters
    - parameters of cleaned image
    """

    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    alt_tel = Field(-1, 'Telescope pointing altitude', u.rad)
    az_tel = Field(-1, 'Telescope pointing azimuth', u.rad)
    mc_energy = Field(-1, 'MC event energy', u.TeV)
    mc_alt = Field(-1, 'MC event altitude', u.deg)
    mc_az = Field(-1, 'MC event azimuth', u.deg)
    mc_disp = Field(-1, 'MC event disp', u.deg)
    mc_core_x = Field(-1, 'MC event core x', u.m)
    mc_core_y = Field(-1, 'MC event core y', u.m)
    mc_impact = Field(-1, 'MC event impact', u.m)
    n_pixels = Field(-1, 'Number of pixels of a cleaned image')
    n_islands = Field(-1, 'Number of islands of a cleaned image')


def mc_dl0_to_dl1(input_file, output_dir, config):
    """
    Processes simtel DL0 data to DL1.

    Parameters
    ----------
    input_file: str
        Path to an input simtel DL0 data file
    output_dir: str
        Path to a directory where to save an output DL1 data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    logger.info('\nInput file:')
    logger.info(input_file)

    event_source = EventSource(input_file)

    obs_id = event_source.obs_ids[0]
    subarray = event_source.subarray

    logger.info('\nSubarray configuration:')
    for tel_id in subarray.tel.keys():
        logger.info(f'Telescope {tel_id}: {subarray.tel[tel_id].name}, position = {subarray.positions[tel_id]}')

    mc_tel_ids = config['mc_tel_ids']
    logger.info('\nThe telescope IDs of LST-1 and MAGIC:')
    logger.info(mc_tel_ids)

    tel_id_lst = mc_tel_ids['LST-1']
    tel_id_m1 = mc_tel_ids['MAGIC-I']
    tel_id_m2 = mc_tel_ids['MAGIC-II']

    # Configure the LST data process:
    config_lst = config['LST']

    logger.info('\nConfiguration for the LST data process:')
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
    use_only_main_island = config_lst['tailcuts_clean'].pop('use_only_main_island')

    extractor_type_lst = config_lst['image_extractor'].pop('type')
    config_extractor_lst = Config({extractor_type_lst: config_lst['image_extractor']})

    calibrator_lst = CameraCalibrator(
        image_extractor_type=extractor_type_lst,
        config=config_extractor_lst,
        subarray=subarray,
    )

    # Configure the MAGIC data process:
    config_magic = config['MAGIC']

    if config_magic['magic_clean']['find_hotpixels'] is not False:
        logger.warning('\nHot pixels do not exist in a simulation. Setting the option to false...')
        config_magic['magic_clean'].update({'find_hotpixels': False})

    logger.info('\nConfiguration for the MAGIC data process:')
    for key, value in config_magic.items():
        logger.info(f'{key}: {value}')

    extractor_type_magic = config_magic['image_extractor'].pop('type')
    config_extractor_magic = Config({extractor_type_magic: config_magic['image_extractor']})

    calibrator_magic = CameraCalibrator(
        image_extractor_type=extractor_type_magic,
        config=config_extractor_magic,
        subarray=subarray,
    )

    # Prepare for saving data to an output file.
    # Here we parse run information from the input file name:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file).resolve().name
    regex_mc = r'(\S+)_run(\d+)_.*\.simtel.gz'
    regex_mc_off = r'(\S+)_run(\d+)_.*_off(\S+)\.simtel.gz'

    if re.fullmatch(regex_mc, base_name):
        parser = re.findall(regex_mc, base_name)[0]
        output_file = f'{output_dir}/dl1_{parser[0]}_LST-1_MAGIC_run{parser[1]}.h5'

    elif re.fullmatch(regex_mc_off, base_name):
        parser = re.findall(regex_mc_off, base_name)[0]
        output_file = f'{output_dir}/dl1_{parser[0]}_off{parser[2]}deg_LST-1_MAGIC_run{parser[1]}.h5'

    # Process the events:
    with HDF5TableWriter(output_file, 'events', mode='w') as writer:

        for tel_name, tel_id in mc_tel_ids.items():

            logger.info(f'\nProcessing the {tel_name} events:')

            tel_position = subarray.positions[tel_id]
            camera_geom = subarray.tel[tel_id].camera.geometry
            focal_length = subarray.tel[tel_id].optics.equivalent_focal_length

            camera_frame = CameraFrame(
                rotation=camera_geom.cam_rotation,
                focal_length=focal_length,
            )

            # Configure the MAGIC cleaning:
            if tel_name in ['MAGIC-I', 'MAGIC-II']:
                magic_clean = MAGICClean(camera_geom, config_magic['magic_clean'])

            n_events_skipped = 0
            event_source_per_tel = EventSource(input_file, allowed_tels=[tel_id])

            for event in event_source_per_tel:

                if event.count % 100 == 0:
                    logger.info(f'{event.count} events')

                if tel_name == 'LST-1':

                    # Calibrate the event:
                    calibrator_lst(event)

                    image = event.dl1.tel[tel_id].image
                    peak_time = event.dl1.tel[tel_id].peak_time

                    if increase_nsb:
                        # Add a noise in pixels:
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
                            camera_geom, signal_pixels, peak_time, **config_lst['time_delta_cleaning'],
                        )

                    if use_dynamic_cleaning:
                        signal_pixels = apply_dynamic_cleaning(
                            image, signal_pixels, **config_lst['dynamic_cleaning'],
                        )

                    if use_only_main_island:
                        _, island_labels = number_of_islands(camera_geom, signal_pixels)
                        n_pixels_on_island = np.bincount(island_labels.astype(np.int64))
                        n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
                        max_island_label = np.argmax(n_pixels_on_island)
                        signal_pixels[island_labels != max_island_label] = False

                elif tel_name in ['MAGIC-I', 'MAGIC-II']:

                    # Calibrate the event:
                    calibrator_magic(event)

                    # Apply the image cleaning:
                    signal_pixels, image, peak_time = magic_clean.clean_image(
                        event.dl1.tel[tel_id].image, event.dl1.tel[tel_id].peak_time,
                    )

                image_cleaned = image.copy()
                image_cleaned[~signal_pixels] = 0

                peak_time_cleaned = peak_time.copy()
                peak_time_cleaned[~signal_pixels] = 0

                n_pixels = np.count_nonzero(signal_pixels)
                n_islands, _ = number_of_islands(camera_geom, signal_pixels)

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
                        camera_geom, image_cleaned, peak_time_cleaned, hillas_params, signal_pixels,
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

                # Calculate the impact parameter:
                mc_impact = calc_impact(
                    core_x=event.simulation.shower.core_x,
                    core_y=event.simulation.shower.core_y,
                    az=event.simulation.shower.az,
                    alt=event.simulation.shower.alt,
                    tel_pos_x=tel_position[0],
                    tel_pos_y=tel_position[1],
                    tel_pos_z=tel_position[2],
                )

                # Set the event information:
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
                    n_pixels=n_pixels,
                    n_islands=n_islands,
                )

                # Reset the telescope IDs:
                if tel_name == 'LST-1':
                    event_info.tel_id = 1

                elif tel_name == 'MAGIC-I':
                    event_info.tel_id = 2

                elif tel_name == 'MAGIC-II':
                    event_info.tel_id = 3

                # Save the parameters to the output file:
                writer.write('params', (event_info, hillas_params, timing_params, leakage_params))

            n_events_processed = event.count + 1

            logger.info(f'\nIn total {n_events_processed} events are processed.')
            logger.info(f'({n_events_skipped} events are skipped)')

    # Reset the telescope IDs of the telescope positions.
    # In addition, we convert the coordinate to the one relative to the center of the LST-1 + MAGIC array:
    positions = np.array([subarray.positions[tel_id].value for tel_id in mc_tel_ids.values()])
    positions_cog = positions - positions.mean(axis=0)

    tel_positions = {
        1: u.Quantity(positions_cog[0,:], u.m),    # LST-1
        2: u.Quantity(positions_cog[1,:], u.m),    # MAGIC-I
        3: u.Quantity(positions_cog[2,:], u.m),    # MAGIC-II
    }

    # Reset the telescope IDs of the telescope descriptions:
    tel_descriptions = {
        1: subarray.tel[tel_id_lst],   # LST-1
        2: subarray.tel[tel_id_m1],    # MAGIC-I
        3: subarray.tel[tel_id_m2],    # MAGIC-II
    }

    # Save the subarray description:
    subarray_lst1_magic = SubarrayDescription('LST1-MAGIC-Array', tel_positions, tel_descriptions)
    subarray_lst1_magic.to_hdf(output_file)

    # Save the simulation configuration:
    with HDF5TableWriter(output_file, 'simulation', mode='a') as writer:
        writer.write('config', event_source.simulation_config)

    logger.info('\nOutput file:')
    logger.info(output_file)

    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str, required=True,
        help='Path to an input simtel DL0 data file (*.simtel.gz).',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output DL1 data file.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    mc_dl0_to_dl1(args.input_file, args.output_dir, config)

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
