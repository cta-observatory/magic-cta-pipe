#!/usr/bin/env python
"""
This script uses geometrical model (see https://doi.org/10.1016/j.jheap.2024.03.003
for details) for the estimation of the vertical transmission, base height and thickness
of a cloud, the so-called "proton" LIDAR.

The script calculates an offset angle from the primary direction, corresponding to the
emission height of Cherenkov light. This is necessary to construct the longitudinal
profile of the observed Cherenkov light and estimate the cloud's vertical transmission
profile. It processes DL1 stereo files, generating output DL1 stereo files that also
contain the longitudinal  distribution of the observed Cherenkov light.

Note: DL1 stereo files must include images. This requires processing the DL1 files to the
stereo level with the "save_images: True" flag enabled in the configuration file, with
the following quality cuts:

quality_cuts: "(n_pixels > 20) & (n_islands < 1.5) & (concentration_cog > 0.001) & ((slope < -1)| (slope > 1))"

Usage:
$ python lst1_magic_cloud_profile_estimator.py
--input_file "/user/data/dl1_stereo_LST-1_MAGIC.Run03265.0040.h5"
(--output_dir "/user/data/cloud_profiles/")
(--config_file "/user/data/config.yaml")
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import astropy.units as u
import ctapipe
import h5py
import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from astropy.coordinates import AltAz
from astropy.coordinates import Angle
from astropy.coordinates import angular_separation
from astropy.coordinates import SkyCoord
from ctapipe.containers import ArrayEventContainer
from ctapipe.containers import CameraHillasParametersContainer
from ctapipe.containers import ImageParametersContainer
from ctapipe.containers import LeakageContainer
from ctapipe.containers import MorphologyContainer
from ctapipe.containers import TimingParametersContainer
from ctapipe.coordinates import TelescopeFrame
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import read_table
from ctapipe.reco import HillasReconstructor
from numpy.linalg import LinAlgError

import magicctapipe
from magicctapipe.io import check_input_list
from magicctapipe.io import format_object
from magicctapipe.io import get_stereo_events
from magicctapipe.io import save_pandas_data_in_table
from magicctapipe.utils import calculate_mean_direction
from magicctapipe.utils import NO_EVENTS_WITHIN_MAXIMUM_DISTANCE

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def model(imp, h, alt):
    """
    Geometrical model needed to estimate the vertical tranmission profile of a cloud.
    Calculates an oﬀset angle from the primary direction corresponding
    to the emission height of the Cherenkov light need to construct the longitudinal
    profile of the observed Cherenkov light
    and to estimate the vertical tranmission profile of a cloud.

    Parameters
    ----------
    imp : float
        Stereo-reconstructed impact parameter of the event in units of meter.
    h : float
        Emission height of the Cherenkov light (above the observatory) in units of meter.
    alt : float
        Altitude angle of the observations in units of radian.

    Returns
    -------
    float
        Oﬀset angle (in united of degree) from the primary direction corresponding to
        the emission height of the Cherenkov light.
    """

    d = h / np.sin(alt)
    model = np.arctan((imp / d).to('')).to_value('deg')
    correction_factor = 0.85 / np.sin(alt)
    return model * correction_factor


def make_profile(
    telescopes_images, dl1_params, config, assigned_tel_ids, magic_only_analysis=False,
):
    """
    Calculates the longitudinal distribution of the observed Cherenkov light.

    Parameters
    ----------
    telescopes_images : dict
        Dictionary containing telescope's details and image.
    dl1_params : pandas.DataFrame
        Pandas DataFrame containing DL1 and stereo-reconstruced parameters.
    config : dict
        Configuration dictionary containing various processing parameters, including those for
        the estimation of the longitudinal distribution of the deteced Cherenkov light.
    assigned_tel_ids : dict
        Dictionary mapping telescope types to their respective telescope IDs.
    magic_only_analysis : bool, optional
        If `True`, the function reconstructs the stereo parameters using only MAGIC events (default is `False`).

    Returns
    -------
    pandas.DataFrame
        Updated dl1_params DataFrame including the longitudinal profile of the observed Cherenkov light.
    """

    profiles_data = {
        'light_emission_profile_phe': [],
        'obs_id': [],
        'tel_id': [],
        'event_id': [],
    }

    cloud_estimator_params = config.get('cloud_estimator', {})
    quality_cuts = cloud_estimator_params['quality_cuts']
    max_offset = u.Quantity(cloud_estimator_params.get('max_offset'))
    end_altitude = u.Quantity(cloud_estimator_params.get('end_altitude'))
    start_altitude = u.Quantity(cloud_estimator_params.get('start_altitude'))
    n_bins_altitude = cloud_estimator_params.get('n_bins_altitude')

    hs = np.linspace(start_altitude, end_altitude, n_bins_altitude)
    hs0 = np.array(
        [hs[0].value] + list(0.5 * (hs[:-1].value + hs[1:].value)) + [hs[-1].value],
    )

    logger.info(
        '\nReconstructing the longitudinal distribution of the observed Cherenkov light ...',
    )
    logger.info(f'\nQuality cuts: {quality_cuts}')

    # Loop through each telescope's data in the dictionary
    for tel_id, telescope_data in telescopes_images.items():
        dl1_images = telescope_data['dl1_images']
        camgeom = telescope_data['camgeom']
        focal_eff = telescope_data['focal_eff']

        m2deg = np.rad2deg(1) / focal_eff * u.degree

        # Get the indices of the valid events for the current telescope

        inds = np.where(
            (dl1_params['tel_id'] == tel_id)
            & (dl1_params.index.isin(dl1_params.query(quality_cuts).index)),
        )[0]

        logger.info(f'\nProcessing images for telescope ID {tel_id}...')

        for index in inds:
            # Extract relevant event info
            event_id = dl1_params['event_id'][index]
            obs_id = dl1_params['obs_id'][index]
            event_id_lst = dl1_params['event_id_lst'][index]
            obs_id_lst = dl1_params['obs_id_lst'][index]
            event_id_magic = dl1_params['event_id_magic'][index]
            obs_id_magic = dl1_params['obs_id_magic'][index]

            if assigned_tel_ids['LST-1'] == tel_id:
                event_id_image, obs_id_image = event_id_lst, obs_id_lst
            else:
                event_id_image, obs_id_image = event_id_magic, obs_id_magic

            # Other event-specific parameters
            pointing_az = dl1_params['pointing_az'][index]
            pointing_alt = dl1_params['pointing_alt'][index]
            time_diff = dl1_params['time_diff'][index]
            n_islands = dl1_params['n_islands'][index]
            signal_pixels = dl1_params['n_pixels'][index]

            # Convert altitude and azimuth to radians
            alt_rad = np.deg2rad(dl1_params['alt'][index])
            az_rad = np.deg2rad(dl1_params['az'][index])

            impact = dl1_params['impact'][index] * u.m
            cog_x = (dl1_params['x'][index] * m2deg).value * u.deg
            cog_y = (dl1_params['y'][index] * m2deg).value * u.deg

            # Source position
            reco_pos = SkyCoord(alt=alt_rad * u.rad, az=az_rad * u.rad, frame=AltAz())
            telescope_pointing = SkyCoord(
                alt=pointing_alt * u.rad,
                az=pointing_az * u.rad,
                frame=AltAz(),
            )

            tel_frame = TelescopeFrame(telescope_pointing=telescope_pointing)
            tel = reco_pos.transform_to(tel_frame)

            src_x = tel.fov_lat
            src_y = tel.fov_lon

            # Transform to Engineering camera
            src_x, src_y = -src_y, -src_x
            cog_x, cog_y = -cog_y, -cog_x

            # Angle between the center and the source
            psi = np.arctan2(src_x - cog_x, src_y - cog_y)

            # Camera geometry transformations
            pix_x_tel = (camgeom.pix_x * m2deg).to(u.deg)
            pix_y_tel = (camgeom.pix_y * m2deg).to(u.deg)

            distance = np.abs(
                (pix_y_tel - src_y) * np.cos(psi) + (pix_x_tel - src_x) * np.sin(psi),
            )

            distance2 = (pix_x_tel - src_x) ** 2 + (pix_y_tel - src_y) ** 2

            # Perpendicular distance from the main axis of the image
            perpdist = np.sqrt(distance2 - distance**2)

            d2_cog_src = (cog_x - src_x) ** 2 + (cog_y - src_y) ** 2
            d2_cog_pix = (cog_x - pix_x_tel) ** 2 + (cog_y - pix_y_tel) ** 2
            d2_src_pix = (src_x - pix_x_tel) ** 2 + (src_y - pix_y_tel) ** 2

            distance[d2_cog_pix > d2_cog_src + d2_src_pix] = 0

            # Convert to heights
            dist_hs = model(impact, hs, pointing_alt) * u.deg
            ibins = np.digitize(distance, dist_hs)

            # Find corresponding image for the event
            inds_img = np.where(
                (dl1_images['event_id'] == event_id_image)
                & (dl1_images['tel_id'] == tel_id)
                & (dl1_images['obs_id'] == obs_id_image),
            )[0]

            if len(inds_img) == 0:
                raise ValueError("Error: 'inds_img' list is empty!")
            index_img = inds_img[0]

            image = dl1_images['image'][index_img]
            clean_mask = dl1_images['image_mask'][index_img]
            clean_mask[perpdist > max_offset] = False

            # imageclean = image * clean_mask

            profile = np.zeros(len(hs0))

            valid_pixids = np.where(clean_mask == True)[0]

            if len(valid_pixids) == 0:
                logger.warning(f'No valid pixels for event {event_id}!')
                continue

            for pixid in valid_pixids:
                profile[ibins[pixid]] += image[pixid]

            profiles_data['light_emission_profile_phe'].append(np.array(profile))
            profiles_data['obs_id'].append(obs_id)
            profiles_data['tel_id'].append(tel_id)
            profiles_data['event_id'].append(event_id)

    profile_df = pd.DataFrame(profiles_data)
    profile_df.dropna(subset=['obs_id', 'event_id', 'tel_id'], inplace=True)
    dl1_params_merged = dl1_params.merge(
        profile_df, on=['obs_id', 'event_id', 'tel_id'], how='inner',
    )

    return dl1_params_merged, hs0


def main():
    """
    Main function.
    """
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_file',
        '-i',
        dest='input_file',
        type=str,
        required=True,
        help='Path to an input .h5 DL1 data file',
    )

    parser.add_argument(
        '--output_dir',
        '-o',
        dest='output_dir',
        type=str,
        default='./cloud_profiles',
        help='Path to a directory where to save an output corrected DL1 file',
    )

    parser.add_argument(
        '--config_file',
        '-c',
        dest='config_file',
        type=str,
        default='./resources/config.yaml',
        help='Path to a configuration file',
    )

    args = parser.parse_args()

    subarray_info = SubarrayDescription.from_hdf(args.input_file)
    tel_descriptions = subarray_info.tel

    camgeom = {}

    for telid, telescope in tel_descriptions.items():
        camgeom[telid] = telescope.camera.geometry

    optics_table = read_table(
        args.input_file, '/configuration/instrument/telescope/optics',
    )

    focal_eff = {}

    for telid, telescope in tel_descriptions.items():
        optics_row = optics_table[optics_table['optics_name'] == telescope.name]
        if len(optics_row) > 0:
            focal_eff[telid] = optics_row['effective_focal_length'][0] * u.m
        else:
            raise ValueError(f'No optics data found for telescope: {telescope.name}')

    with open(args.config_file) as file:
        config = yaml.safe_load(file)

    tel_ids = config['mc_tel_ids']

    dl1_params = pd.read_hdf(args.input_file, key='events/parameters')
    dl1_params['total_uncert'] = np.sqrt(
        dl1_params['alt_uncert'] ** 2
        + (dl1_params['az_uncert'] * np.cos(dl1_params['pointing_alt'])) ** 2,
    )

    telescopes_images = {}

    for tel_name, tel_id in tel_ids.items():
        if tel_id != 0:  # Only process telescopes that have a non-zero ID
            # Read images for each telescope
            image_node_path = '/events/dl1/image_' + str(tel_id)
            try:
                dl1_images = read_table(args.input_file, image_node_path)
            except tables.NoSuchNodeError:
                raise RuntimeError(
                    f'Fatal error: No image found for telescope with ID {tel_id}.',
                )

            # Store data for the current telescope in the dictionary
            telescopes_images[tel_id] = {
                'tel_name': tel_name,
                'tel_id': tel_id,
                'dl1_images': dl1_images,
                'camgeom': camgeom[tel_id],
                'focal_eff': focal_eff[tel_id],
            }

    Cherenkov_light_emission_profile, heights = make_profile(
        telescopes_images, dl1_params, config, tel_ids,
    )
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    input_file_name = Path(args.input_file).name
    output_file_name = input_file_name.replace(
        'dl1_stereo_MAGIC_LST-1', 'dl1_stereo_Cherenkov_emission_profile',
    )
    output_file = f'{args.output_dir}/{output_file_name}'

    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('heights', data=heights)
    Cherenkov_light_emission_profile.to_hdf(
        output_file, key='events/parameters', mode='a',
    )

    logger.info(f"\nProton LIDAR parameters: {config.get('cloud_estimator', {})}")
    logger.info(f'ctapipe version: {ctapipe.__version__}')
    logger.info(f'magicctapipe version: {magicctapipe.__version__}')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')
    logger.info(f'\nOutput file: {output_file}')

    logger.info('\nDone.')


if __name__ == '__main__':
    main()
