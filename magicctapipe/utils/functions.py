#!/usr/bin/env python
# coding: utf-8

import tables
import logging
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    SkyCoord,
    EarthLocation,
)
from astropy.coordinates.builtin_frames import SkyOffsetFrame

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

tel_combinations = {
    'm1_m2': [2, 3],   # combo_type = 0
    'lst1_m1': [1, 2],   # combo_type = 1
    'lst1_m2': [1, 3],   # combo_type = 2
    'lst1_m1_m2': [1, 2, 3],   # combo_type = 3
}

__all__ = [
    'get_dl2_mean',
    'calc_mean_direction',
    'check_tel_combinations',
    'save_data_to_hdf',
    'calc_impact',
    'calc_nsim',
    'transform_to_radec',
    'calc_angular_separation',
]


def get_dl2_mean(input_data):
    """
    Calculates the mean of the DL2 parameters
    weighted by the uncertainties of RF estimations.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing the DL2 parameters

    Returns
    -------
    dl2_mean: pandas.core.frame.DataFrame
        Pandas data frame containing the mean of the DL2 parameters
    """

    logger.info('\nComputing the mean of the DL2 parameters...')

    is_simulation = ('mc_energy' in input_data.columns)
    groupby_mean = input_data.groupby(['obs_id', 'event_id']).mean()

    # Compute the mean of the gammaness/hadronness:
    gammaness_mean = groupby_mean['gammaness']
    hadronness_mean = groupby_mean['hadronness']

    # Compute the mean of the reconstructed energy:
    weights = 1 / input_data['reco_energy_err']
    weighted_energy = np.log10(input_data['reco_energy']) * weights

    weights_sum = weights.groupby(['obs_id', 'event_id']).sum()
    weighted_energy_sum = weighted_energy.groupby(['obs_id', 'event_id']).sum()

    reco_energy_mean = 10 ** (weighted_energy_sum / weights_sum)

    # Compute the mean of the reconstructed arrival direction:
    reco_az_mean, reco_alt_mean = calc_mean_direction(
        lon=np.deg2rad(input_data['reco_az']),
        lat=np.deg2rad(input_data['reco_alt']),
        weights=input_data['reco_disp_err'],
    )

    # Compute the mean of the telescope pointing direction:
    az_tel_mean, alt_tel_mean = calc_mean_direction(
        lon=input_data['az_tel'], lat=input_data['alt_tel'],
    )

    # Create a base data frame:
    dl2_mean = pd.DataFrame(
        data={'gammaness': gammaness_mean.to_numpy(),
              'hadronness': hadronness_mean.to_numpy(),
              'reco_energy': reco_energy_mean.to_numpy(),
              'reco_alt': reco_alt_mean.to(u.deg).value,
              'reco_az': reco_az_mean.to(u.deg).value,
              'alt_tel': alt_tel_mean.to(u.rad).value,
              'az_tel': az_tel_mean.to(u.rad).value},
        index=groupby_mean.index,
    )

    if is_simulation:
        # Add the MC parameters:
        mc_params = groupby_mean[['mc_energy', 'mc_alt', 'mc_az']]
        dl2_mean = dl2_mean.join(mc_params)

    else:
        # Add the mean of the Ra/Dec direction:
        reco_ra_mean, reco_dec_mean = calc_mean_direction(
            lon=np.deg2rad(input_data['reco_ra']),
            lat=np.deg2rad(input_data['reco_dec']),
            weights=input_data['reco_disp_err'],
        )

        ra_tel_mean, dec_tel_mean = calc_mean_direction(
            lon=input_data['ra_tel'], lat=input_data['dec_tel'],
        )

        radec_mean = pd.DataFrame(
            data={'reco_ra': reco_ra_mean.to(u.deg).value,
                  'reco_dec': reco_dec_mean.to(u.deg).value,
                  'ra_tel': ra_tel_mean.to(u.rad).value,
                  'dec_tel': dec_tel_mean.to(u.rad).value},
            index=groupby_mean.index,
        )

        dl2_mean = dl2_mean.join(radec_mean)

    # Add the telescope combination types:
    combo_types = check_tel_combinations(input_data)
    dl2_mean = dl2_mean.join(combo_types)

    return dl2_mean


def calc_mean_direction(lon, lat, weights=None):
    """
    Calculates mean directions in a spherical coordinate.
    The input Series should have the index of "obs_id" and "event_id".

    Parameters
    ----------
    lon: pandas.core.series.Series
        Longitude in a spherical coordinate
    lat: pandas.core.series.Series
        Latitude in a spherical coodinate
    weights: pandas.core.series.Series
        Weights applied when calculating the mean directions

    Returns
    -------
    lon_mean: 
        Longitude of the mean directions
    lat_mean: 
        Latitude of the mean directions
    """

    x_coords = np.cos(lat) * np.cos(lon)
    y_coords = np.cos(lat) * np.sin(lon)
    z_coords = np.sin(lat)

    if weights is not None:
        weighted_x_coords = x_coords * weights
        weighted_y_coords = y_coords * weights
        weighted_z_coords = z_coords * weights

        weights_sum = weights.groupby(['obs_id', 'event_id']).sum()

        weighted_x_coords_sum = weighted_x_coords.groupby(['obs_id', 'event_id']).sum()
        weighted_y_coords_sum = weighted_y_coords.groupby(['obs_id', 'event_id']).sum()
        weighted_z_coords_sum = weighted_z_coords.groupby(['obs_id', 'event_id']).sum()

        x_coords_mean = weighted_x_coords_sum / weights_sum
        y_coords_mean = weighted_y_coords_sum / weights_sum
        z_coords_mean = weighted_z_coords_sum / weights_sum

    else:
        x_coords_mean = x_coords.groupby(['obs_id', 'event_id']).sum()
        y_coords_mean = y_coords.groupby(['obs_id', 'event_id']).sum()
        z_coords_mean = z_coords.groupby(['obs_id', 'event_id']).sum()

    coord_mean = SkyCoord(
        x=x_coords_mean.values,
        y=y_coords_mean.values,
        z=z_coords_mean.values,
        representation_type='cartesian',
    )

    lon_mean = coord_mean.spherical.lon
    lat_mean = coord_mean.spherical.lat

    return lon_mean, lat_mean


def check_tel_combinations(input_data):

    combo_types = pd.DataFrame()

    n_events_total = len(input_data.groupby(['obs_id', 'event_id']).size())
    logger.info(f'\nIn total {n_events_total} stereo events are found:')

    for combo_type, (tel_combo, tel_ids) in enumerate(tel_combinations.items()):

        df = input_data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')

        groupby_size = df.groupby(['obs_id', 'event_id']).size()
        groupby_size = groupby_size[groupby_size == len(tel_ids)]

        n_events = len(groupby_size)
        ratio = n_events / n_events_total

        logger.info(f'{tel_combo} (type {combo_type}): {n_events:.0f} events ({ratio * 100:.1f}%)')

        df_combo_type = pd.DataFrame({'combo_type': combo_type}, index=groupby_size.index)
        combo_types = combo_types.append(df_combo_type)

    combo_types.sort_index(inplace=True)

    return combo_types


def save_data_to_hdf(data, output_file, group_name, table_name):

    with tables.open_file(output_file, mode='a') as f_out:

        event_values = [tuple(array) for array in data.to_numpy()]
        dtypes = np.dtype([(name, dtype) for name, dtype in zip(data.dtypes.index, data.dtypes)])

        event_table = np.array(event_values, dtype=dtypes)
        f_out.create_table(group_name, table_name, createparents=True, obj=event_table)


def calc_impact(core_x, core_y, az, alt, tel_pos_x, tel_pos_y, tel_pos_z):

    t = (tel_pos_x - core_x) * np.cos(alt) * np.cos(az) \
        - (tel_pos_y - core_y) * np.cos(alt) * np.sin(az) \
        + tel_pos_z * np.sin(alt)

    impact = np.sqrt((core_x - tel_pos_x + t * np.cos(alt) * np.cos(az)) ** 2 \
                     + (core_y - tel_pos_y - t * np.cos(alt) * np.sin(az)) ** 2 \
                     + (t * np.sin(alt) - tel_pos_z) ** 2)

    return impact


def calc_nsim(n_events_sim, eslope_sim, emin_sim, emax_sim, cscat_sim, viewcone_sim,
              emin=None, emax=None, distmin=None, distmax=None, angmin=None, angmax=None):

    norm = 1

    if (emin != None) & (emax != None):
        norm *= (emax ** (eslope_sim + 1) - emin ** (eslope_sim + 1)) \
                / (emax_sim ** (eslope_sim + 1) - emin_sim ** (eslope_sim + 1))

    if (distmin != None) & (distmax != None):
        norm *= (distmax ** 2 - distmin ** 2) / cscat_sim ** 2

    if (angmin != None) & (angmax != None):
        norm *= (np.cos(angmin) - np.cos(angmax)) / (1 - np.cos(viewcone_sim))

    nsim = norm * n_events_sim

    return nsim.value


def transform_to_radec(alt, az, timestamp):

    lat_orm = u.Quantity(28.76177, u.deg)
    lon_orm = u.Quantity(-17.89064, u.deg)
    height_orm = u.Quantity(2199.835, u.m)

    location = EarthLocation.from_geodetic(lat=lat_orm, lon=lon_orm, height=height_orm)

    horizon_frames = AltAz(location=location, obstime=timestamp)

    event_coords = SkyCoord(alt=alt, az=az, frame=horizon_frames)
    event_coords = event_coords.transform_to('icrs')

    return event_coords.ra, event_coords.dec


def calc_angular_separation(on_coord, event_coords, tel_coords, n_off_region):

    theta_on = on_coord.separation(event_coords)

    offsets = np.arccos(np.cos(on_coord.dec) * np.cos(tel_coords.dec) * np.cos(tel_coords.ra - on_coord.ra) \
                        + np.sin(on_coord.dec) * np.sin(tel_coords.dec))

    numerator = np.sin(tel_coords.dec) * np.cos(on_coord.dec) \
                - np.sin(on_coord.dec) * np.cos(tel_coords.dec) * np.cos(tel_coords.ra - on_coord.ra)

    denominator = np.cos(tel_coords.dec) * np.sin(tel_coords.ra - on_coord.ra)

    rotations = np.arctan2(numerator, denominator)
    rotations[rotations < 0] += u.Quantity(360, u.deg)

    mean_offset = offsets.to(u.deg).mean()
    mean_rot = rotations.to(u.deg).mean()

    skyoffset_frame = SkyOffsetFrame(origin=on_coord, rotation=-mean_rot)

    wobble_coord = SkyCoord(mean_offset, u.Quantity(0, u.deg), frame=skyoffset_frame)
    wobble_coord = wobble_coord.transform_to('icrs')

    rotations_off = np.arange(0, 359, 360/(n_off_region + 1))
    rotations_off = rotations_off[rotations_off != 180]
    rotations_off += mean_rot.value

    theta_off = {}
    off_coords = {}

    for i_off, rot in enumerate(rotations_off):

        skyoffset_frame = SkyOffsetFrame(origin=wobble_coord, rotation=u.Quantity(-rot, u.deg))

        off_coords[i_off+1] = SkyCoord(mean_offset, u.Quantity(0, u.deg), frame=skyoffset_frame)
        off_coords[i_off+1] = off_coords[i_off+1].transform_to('icrs')

        theta_off[i_off+1] = off_coords[i_off+1].separation(event_coords)

    return theta_on, theta_off, off_coords
