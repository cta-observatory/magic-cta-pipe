#!/usr/bin/env python
# coding: utf-8

"""
This script processes LST-1 and MAGIC events of simtel MC DL0 data (*.simtel.gz)
and computes the DL1 parameters (i.e., Hillas, timing and leakage parameters).
It saves only the events that all the DL1 parameters are successfully reconstructed.
The telescope IDs are reset to the following ones when saving to an output file:
LST-1: tel_id = 1,  MAGIC-I: tel_id = 2,  MAGIC-II: tel_id = 3

Usage:
$ python lst1_magic_mc_dl0_to_dl1.py
--input-file ./data/gamma_off0.4deg/dl0/gamma_40deg_90deg_run1___cta-prod5-lapalma_LST-1_MAGIC_desert-2158m_mono_off0.4.simtel.gz
--output-dir ./data/gamma_off0.4deg/dl1
--config-file ./config.yaml
(--muons)
"""

import os
import re
import time
import yaml
import logging
import argparse
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.table import Table
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
    ImageExtractor,
    tailcuts_clean,
    apply_time_delta_cleaning,
    number_of_islands,
    hillas_parameters,
    timing_parameters,
    leakage_parameters,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.coordinates import TelescopeFrame
from lstchain.image.cleaning import apply_dynamic_cleaning
from lstchain.image.modifier import (
    set_numba_seed,
    add_noise_in_pixels,
    random_psf_smearer,
)
from lstchain.image.muon import create_muon_table
from magicctapipe.image import MAGICClean
from magicctapipe.image.muons import perform_muon_analysis
from magicctapipe.utils import calculate_impact

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [
    'EventInfoContainer',
    'mc_dl0_to_dl1',
]


class EventInfoContainer(Container):
    """ Container to store event information """

    obs_id = Field(-1, 'Observation ID')
    event_id = Field(-1, 'Event ID')
    tel_id = Field(-1, 'Telescope ID')
    pointing_alt = Field(-1, 'Telescope pointing altitude', u.rad)
    pointing_az = Field(-1, 'Telescope pointing azimuth', u.rad)
    true_energy = Field(-1, 'MC event true energy', u.TeV)
    true_alt = Field(-1, 'MC event true altitude', u.deg)
    true_az = Field(-1, 'MC event true azimuth', u.deg)
    true_disp = Field(-1, 'MC event true disp', u.deg)
    true_core_x = Field(-1, 'MC event true core x', u.m)
    true_core_y = Field(-1, 'MC event true core y', u.m)
    true_impact = Field(-1, 'MC event true impact', u.m)
    n_pixels = Field(-1, 'Number of pixels of a cleaned image')
    n_islands = Field(-1, 'Number of islands of a cleaned image')
    magic_stereo = Field(-1, 'True if both M1 and M2 are triggered')


def mc_dl0_to_dl1(
    input_file,
    output_dir,
    config,
    muons_analysis=False,
):
    """
    Processes LST-1 and MAGIC events of simtel MC DL0 data
    and computes the DL1 parameters.

    Parameters
    ----------
    input_file: str
        Path to an input simtel MC DL0 data file
    output_dir: str
        Path to a directory where to save an output DL1 data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    muons_analysis: bool
        Perform the muon ring analysis if True
    """

    logger.info(f'\nMuons analysis: {muons_analysis}')

    allowed_tel_ids = config['mc_tel_ids']

    logger.info('\nAllowed telescope IDs:')
    logger.info(allowed_tel_ids)

    tel_id_lst1 = allowed_tel_ids['LST-1']
    tel_id_m1 = allowed_tel_ids['MAGIC-I']
    tel_id_m2 = allowed_tel_ids['MAGIC-II']

    # Load the input file:
    logger.info('\nLoading the input file:')
    logger.info(input_file)

    event_source = EventSource(
        input_file,
        allowed_tels=list(allowed_tel_ids.values()),
        focal_length_choice='effective',
    )

    obs_id = event_source.obs_ids[0]
    subarray = event_source.subarray

    camera_geoms = {}
    tel_positions = subarray.positions

    logger.info('\nSubarray configuration:')
    for tel_id in allowed_tel_ids.values():
        logger.info(f'Telescope {tel_id}: {subarray.tel[tel_id].name}, position = {tel_positions[tel_id]}')
        camera_geoms[tel_id] = subarray.tel[tel_id].camera.geometry

    # Configure the LST event processors:
    config_lst = config['LST']

    logger.info('\nLST image extractor:')
    logger.info(config_lst['image_extractor'])

    extractor_type_lst = config_lst['image_extractor'].pop('type')
    config_extractor_lst = Config({extractor_type_lst: config_lst['image_extractor']})

    calibrator_lst = CameraCalibrator(
        image_extractor_type=extractor_type_lst,
        config=config_extractor_lst,
        subarray=subarray,
    )

    logger.info('\nLST image modifier:')
    for modifier in ['increase_nsb', 'increase_psf']:
        logger.info(f'{modifier}: {config_lst[modifier]}')

    increase_nsb = config_lst['increase_nsb'].pop('use')
    increase_psf = config_lst['increase_psf'].pop('use')

    if increase_nsb:
        rng = np.random.default_rng(obs_id)

    if increase_psf:
        set_numba_seed(obs_id)

    logger.info('\nLST image cleaning:')
    for cleaning in ['tailcuts_clean', 'time_delta_cleaning', 'dynamic_cleaning']:
        logger.info(f'{cleaning}: {config_lst[cleaning]}')

    use_time_delta_cleaning = config_lst['time_delta_cleaning'].pop('use')
    use_dynamic_cleaning = config_lst['dynamic_cleaning'].pop('use')

    use_only_main_island = config_lst['use_only_main_island']
    logger.info(f'use_only_main_island: {use_only_main_island}')

    # Configure the MAGIC event processors:
    config_magic = config['MAGIC']

    logger.info('\nMAGIC image extractor:')
    logger.info(config_magic['image_extractor'])

    extractor_type_magic = config_magic['image_extractor'].pop('type')
    config_extractor_magic = Config({extractor_type_magic: config_magic['image_extractor']})

    calibrator_magic = CameraCalibrator(
        image_extractor_type=extractor_type_magic,
        config=config_extractor_magic,
        subarray=subarray,
    )

    logger.info('\nMAGIC charge correction:')
    logger.info(config_magic['charge_correction'])

    use_charge_correction = config_magic['charge_correction'].pop('use')

    logger.info('\nMAGIC image cleaning:')
    if config_magic['magic_clean']['find_hotpixels'] is not False:
        logger.warning('Hot pixels do not exist in a simulation. Setting the "find_hotpixels" option to False...')
        config_magic['magic_clean'].update({'find_hotpixels': False})

    logger.info(config_magic['magic_clean'])
    magic_clean = MAGICClean(camera_geoms[tel_id_m1], config_magic['magic_clean'])   # assume M1 and M2 cameras are identical

    # Configure the muon analysis:
    muon_parameters = create_muon_table()
    muon_parameters['telescope_type'] = []
    r1_dl1_calibrator_for_muon_rings = {}

    if muons_analysis:
        extractor_muon_name_lst = 'GlobalPeakWindowSum'
        extractor_lst_muons = ImageExtractor.from_name(
            extractor_muon_name_lst, subarray=subarray, config=config_extractor_lst
        )
        r1_dl1_calibrator_for_muon_rings[tel_id_lst1] = CameraCalibrator(subarray,
                                                                         image_extractor=extractor_lst_muons)
        extractor_muon_name_magic = 'GlobalPeakWindowSum'
        extractor_magic_muons = ImageExtractor.from_name(
            extractor_muon_name_magic, subarray=subarray, config=config_extractor_magic
        )
        r1_dl1_calibrator_for_muon_rings_magic = CameraCalibrator(subarray,
                                                                  image_extractor=extractor_magic_muons)
        r1_dl1_calibrator_for_muon_rings[tel_id_m1] = r1_dl1_calibrator_for_muon_rings_magic
        r1_dl1_calibrator_for_muon_rings[tel_id_m2] = r1_dl1_calibrator_for_muon_rings_magic
        muon_config = {tel_id_lst1: {},
                       tel_id_m1: {},
                       tel_id_m2: {}}
        if 'muon_ring' in config_lst:
            muon_config[tel_id_lst1] = config_lst['muon_ring']
        if 'muon_ring' in config_magic:
            muon_config[tel_id_m1] = config_magic['muon_ring']
            muon_config[tel_id_m2] = config_magic['muon_ring']

    # Prepare for saving data to an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    regex_off = r'(\S+)_run(\d+)_.*_off(\S+)\.simtel.gz'
    regex = r'(\S+)_run(\d+)[_\.].*simtel.gz'

    file_name = Path(input_file).name

    if re.fullmatch(regex_off, file_name):
        parser = re.findall(regex_off, file_name)[0]
        output_file = f'{output_dir}/dl1_{parser[0]}_off{parser[2]}deg_LST-1_MAGIC_run{parser[1]}.h5'

    elif re.fullmatch(regex, file_name):
        parser = re.findall(regex, file_name)[0]
        output_file = f'{output_dir}/dl1_{parser[0]}_LST-1_MAGIC_run{parser[1]}.h5'

    else:
        raise RuntimeError('Could not parse information from the input file name.')

    # Start processing the events:
    logger.info('\nProcessing the events...')

    with HDF5TableWriter(output_file, group_name='events', mode='w') as writer:

        for event in event_source:

            if event.count % 100 == 0:
                logger.info(f'{event.count} events')

            tels_with_trigger = event.trigger.tels_with_trigger

            # Check if the event triggers both M1 and M2:
            trigger_m1 = (tel_id_m1 in tels_with_trigger)
            trigger_m2 = (tel_id_m2 in tels_with_trigger)

            magic_stereo = (trigger_m1 and trigger_m2)

            for tel_id in tels_with_trigger:

                if tel_id == tel_id_lst1:

                    # Calibrate the event:
                    calibrator_lst._calibrate_dl0(event, tel_id)
                    calibrator_lst._calibrate_dl1(event, tel_id)

                    image = event.dl1.tel[tel_id].image.astype(np.float64)
                    peak_time = event.dl1.tel[tel_id].peak_time.astype(np.float64)

                    if increase_nsb:
                        # Add noises in pixels:
                        image = add_noise_in_pixels(rng, image, **config_lst['increase_nsb'])

                    if increase_psf:
                        # Smear the image:
                        image = random_psf_smearer(image, config_lst['increase_psf']['smeared_light_fraction'],
                                                   camera_geoms[tel_id].neighbor_matrix_sparse.indices,
                                                   camera_geoms[tel_id].neighbor_matrix_sparse.indptr)

                    # Apply the image cleaning:
                    signal_pixels = tailcuts_clean(camera_geoms[tel_id], image, **config_lst['tailcuts_clean'])

                    if use_time_delta_cleaning:
                        signal_pixels = apply_time_delta_cleaning(camera_geoms[tel_id], signal_pixels,
                                                                  peak_time, **config_lst['time_delta_cleaning'])

                    if use_dynamic_cleaning:
                        signal_pixels = apply_dynamic_cleaning(image, signal_pixels, **config_lst['dynamic_cleaning'])

                    if use_only_main_island:
                        _, island_labels = number_of_islands(camera_geoms[tel_id], signal_pixels)
                        n_pixels_on_island = np.bincount(island_labels.astype(np.int64))
                        n_pixels_on_island[0] = 0  # first island is no-island and should not be considered
                        max_island_label = np.argmax(n_pixels_on_island)
                        signal_pixels[island_labels != max_island_label] = False

                else:
                    # Calibrate the event:
                    calibrator_magic._calibrate_dl0(event, tel_id)
                    calibrator_magic._calibrate_dl1(event, tel_id)

                    image = event.dl1.tel[tel_id].image.astype(np.float64)
                    peak_time = event.dl1.tel[tel_id].peak_time.astype(np.float64)

                    if use_charge_correction:
                        # Scale the charges of the DL1 image by the correction factor:
                        image *= config_magic['charge_correction']['correction_factor']

                    # Apply the image cleaning:
                    signal_pixels, image, peak_time = magic_clean.clean_image(image, peak_time)

                image_cleaned = image.copy()
                image_cleaned[~signal_pixels] = 0

                peak_time_cleaned = peak_time.copy()
                peak_time_cleaned[~signal_pixels] = 0

                n_pixels = np.count_nonzero(signal_pixels)
                n_islands, _ = number_of_islands(camera_geoms[tel_id], signal_pixels)

                if n_pixels == 0:
                    logger.warning(f'--> {event.count} event (event ID {event.index.event_id}, telescope {tel_id}): ' \
                                   'Could not survive the image cleaning. Skipping.')
                    continue

                # Try to compute the Hillas parameters:
                try:
                    hillas_params = hillas_parameters(camera_geoms[tel_id], image_cleaned)
                except:
                    logger.warning(f'--> {event.count} event (event ID {event.index.event_id}, telescope {tel_id}): ' \
                                   'Hillas parameters computation failed. Skipping.')
                    continue

                # Try to compute the timing parameters:
                try:
                    timing_params = timing_parameters(camera_geoms[tel_id], image_cleaned,
                                                      peak_time_cleaned, hillas_params, signal_pixels)
                except:
                    logger.warning(f'--> {event.count} event (event ID {event.index.event_id}, telescope {tel_id}): ' \
                                   'Timing parameters computation failed. Skipping.')
                    continue

                # Try to compute the leakage parameters:
                try:
                    leakage_params = leakage_parameters(camera_geoms[tel_id], image_cleaned, signal_pixels)
                except:
                    logger.warning(f'--> {event.count} event (event ID {event.index.event_id}, telescope {tel_id}): ' \
                                   'Leakage parameters computation failed. Skipping.')
                    continue

                # Compute the DISP parameter:
                if (event.pointing.tel[tel_id].altitude > 90*u.deg) & (event.pointing.tel[tel_id].altitude < 90.01*u.deg):
                    # simu at altitude == 90 can have saved value rounded up at float precision limit
                    event.pointing.tel[tel_id].altitude = 90*u.deg

                tel_pointing = AltAz(
                    alt=event.pointing.tel[tel_id].altitude,
                    az=event.pointing.tel[tel_id].azimuth,
                )

                tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

                event_coord = SkyCoord(hillas_params.x, hillas_params.y, frame=camera_geoms[tel_id].frame)
                event_coord = event_coord.transform_to(tel_frame)

                true_disp = angular_separation(
                    lon1=event_coord.altaz.az,
                    lat1=event_coord.altaz.alt,
                    lon2=event.simulation.shower.az,
                    lat2=event.simulation.shower.alt,
                )

                # Calculate the impact parameter:
                true_impact = calculate_impact(
                    core_x=event.simulation.shower.core_x,
                    core_y=event.simulation.shower.core_y,
                    az=event.simulation.shower.az,
                    alt=event.simulation.shower.alt,
                    tel_pos_x=tel_positions[tel_id][0],
                    tel_pos_y=tel_positions[tel_id][1],
                    tel_pos_z=tel_positions[tel_id][2],
                )

                # Set the event information:
                event_info = EventInfoContainer(
                    obs_id=event.index.obs_id,
                    event_id=event.index.event_id,
                    pointing_alt=event.pointing.tel[tel_id].altitude,
                    pointing_az=event.pointing.tel[tel_id].azimuth,
                    true_energy=event.simulation.shower.energy,
                    true_alt=event.simulation.shower.alt,
                    true_az=event.simulation.shower.az,
                    true_disp=true_disp,
                    true_core_x=event.simulation.shower.core_x,
                    true_core_y=event.simulation.shower.core_y,
                    true_impact=true_impact,
                    n_pixels=n_pixels,
                    n_islands=n_islands,
                    magic_stereo=magic_stereo,
                )

                # Reset the telescope IDs:
                if tel_id == tel_id_lst1:
                    event_info.tel_id = 1

                elif tel_id == tel_id_m1:
                    event_info.tel_id = 2

                elif tel_id == tel_id_m2:
                    event_info.tel_id = 3

                # Save the parameters to the output file:
                writer.write('parameters', (event_info, hillas_params, timing_params, leakage_params))

                if muons_analysis:
                    perform_muon_analysis(muon_parameters,
                                          event=event,
                                          telescope_id=tel_id,
                                          image=image,
                                          subarray=subarray,
                                          r1_dl1_calibrator_for_muon_rings=
                                          r1_dl1_calibrator_for_muon_rings[tel_id],
                                          good_ring_config=muon_config[tel_id],
                                          telescope_type=subarray.tel[tel_id].name,
                                          data_type='mc')

        n_events_processed = event.count + 1
        logger.info(f'\nIn total {n_events_processed} events are processed.')

    # Reset the telescope IDs of the subarray description.
    # In addition, convert the telescope coordinate to the one relative to the center of the LST-1 + MAGIC array:
    positions = np.array([tel_positions[tel_id].value for tel_id in allowed_tel_ids.values()])
    positions_cog = positions - positions.mean(axis=0)

    tel_positions_cog = {
        1: u.Quantity(positions_cog[0, :], u.m),    # LST-1
        2: u.Quantity(positions_cog[1, :], u.m),    # MAGIC-I
        3: u.Quantity(positions_cog[2, :], u.m),    # MAGIC-II
    }

    tel_descriptions = {
        1: subarray.tel[tel_id_lst1],   # LST-1
        2: subarray.tel[tel_id_m1],     # MAGIC-I
        3: subarray.tel[tel_id_m2],     # MAGIC-II
    }

    subarray_lst1_magic = SubarrayDescription('LST1-MAGIC-Array', tel_positions_cog, tel_descriptions)

    # Save the subarray description:
    subarray_lst1_magic.to_hdf(output_file)

    # Save the simulation configuration:
    with HDF5TableWriter(output_file, group_name='simulation', mode='a') as writer:
        writer.write('config', event_source.simulation_config)

    logger.info('\nOutput file:')
    logger.info(output_file)

    if muons_analysis:
        dir, name = os.path.split(output_file)
        name = name.replace('dl1', 'muons')
        # Consider the possibilities of DL1 files with .fits.h5 & .h5 ending:
        name = name.replace('.h5', '.fits')
        muon_output_filename = dir + '/' + name
        table = Table(muon_parameters)
        table.write(muon_output_filename, format='fits', overwrite=True)
        logger.info(f'\nOutput muons file: {muon_output_filename}')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str, required=True,
        help='Path to an input simtel MC DL0 data file.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output DL1 data file.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.',
    )

    parser.add_argument(
        '--muons', '-m', dest='muons', action='store_true',
        help='Boolean to do or not the muon analysis',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # Process the input data:
    mc_dl0_to_dl1(args.input_file, args.output_dir, config, args.muons)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
