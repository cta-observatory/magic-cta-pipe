#!/usr/bin/env python
# coding: utf-8

import tables
import logging
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.coordinates.builtin_frames import SkyOffsetFrame

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [
    'calc_impact',
    'calc_mean_direction',
    'calc_angular_distance',
    'transform_altaz_to_radec',
    'check_tel_combination',
    'save_pandas_to_table',
    'get_dl2_mean',
]


def calc_impact(core_x, core_y, az, alt, tel_pos_x, tel_pos_y, tel_pos_z):
    """
    Calculates the impact distance from a given telescope.

    Parameters
    ----------
    core_x: astropy.units.quantity.Quantity
        Core position along the geographical north
    core_y: astropy.units.quantity.Quantity
        Core position along the geographical west
    az: astropy.units.quantity.Quantity
        Azimuth of the event arrival direction
    alt: astropy.units.quantity.Quantity
        Altitude of the event arrival direction
    tel_pos_x: astropy.units.quantity.Quantity
        Telescope position along the geographical north
    tel_pos_y: astropy.units.quantity.Quantity
        Telescope position along the geographical west
    tel_pos_z: astropy.units.quantity.Quantity
        Altitude of a telescope position

    Returns
    -------
    impact: astropy.units.quantity.Quantity
        Impact distance from the input telescope position
    """

    t = (tel_pos_x - core_x) * np.cos(alt) * np.cos(az) \
        - (tel_pos_y - core_y) * np.cos(alt) * np.sin(az) \
        + tel_pos_z * np.sin(alt)

    impact = np.sqrt((core_x - tel_pos_x + t * np.cos(alt) * np.cos(az)) ** 2 \
                     + (core_y - tel_pos_y - t * np.cos(alt) * np.sin(az)) ** 2 \
                     + (t * np.sin(alt) - tel_pos_z) ** 2)

    return impact


def calc_mean_direction(lon, lat, weights=None):
    """
    Calculates the mean of input directions in a spherical coordinate.
    The inputs should be the "Series" type with the indices of 'obs_id' and 'event_id'.
    The unit of the input longitude/latitude should be radian.

    Parameters
    ----------
    lon: pandas.core.series.Series
        Longitude in a spherical coordinate
    lat: pandas.core.series.Series
        Latitude in a spherical coodinate
    weights: pandas.core.series.Series
        Weights applied when calculating the mean direction

    Returns
    -------
    lon_mean: astropy.units.quantity.Quantity
        Longitude of the mean direction
    lat_mean: astropy.units.quantity.Quantity
        Latitude of the mean direction
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

        x_coord_mean = weighted_x_coords_sum / weights_sum
        y_coord_mean = weighted_y_coords_sum / weights_sum
        z_coord_mean = weighted_z_coords_sum / weights_sum

    else:
        x_coord_mean = x_coords.groupby(['obs_id', 'event_id']).sum()
        y_coord_mean = y_coords.groupby(['obs_id', 'event_id']).sum()
        z_coord_mean = z_coords.groupby(['obs_id', 'event_id']).sum()

    coord_mean = SkyCoord(
        x=x_coord_mean.values,
        y=y_coord_mean.values,
        z=z_coord_mean.values,
        representation_type='cartesian',
    )

    lon_mean = coord_mean.spherical.lon
    lat_mean = coord_mean.spherical.lat

    return lon_mean, lat_mean


def calc_angular_distance(on_coord, event_coord, tel_coord, n_off_regions):
    """
    Calculates the angular distance between the shower arrival direction
    and ON/OFF regions.

    Parameters
    ----------
    on_coord: astropy.coordinates.sky_coordinate.SkyCoord
        Coordinate of the ON region
    event_coord: astropy.coordinates.sky_coordinate.SkyCoord
        Coordinate of the shower arrival direction
    tel_coord: astropy.coordinates.sky_coordinate.SkyCoord
        Coordinate of the telescope pointing direction
    n_off_regions: int
        Number of OFF regions to be extracted

    Returns
    -------
    theta_on: astropy.units.quantity.Quantity
        Angular distance from the ON region
    theta_off: dict
        Angular distances from the OFF regions
    off_coords: dict
        Coordinates of the OFF regions
    """

    # Compute the distance from the ON region:
    theta_on = on_coord.separation(event_coord)

    # Compute the wobble offset and rotation:
    offsets = np.arccos(np.cos(on_coord.dec) * np.cos(tel_coord.dec) * np.cos(tel_coord.ra - on_coord.ra) \
                        + np.sin(on_coord.dec) * np.sin(tel_coord.dec))

    numerator = np.sin(tel_coord.dec) * np.cos(on_coord.dec) \
                - np.sin(on_coord.dec) * np.cos(tel_coord.dec) * np.cos(tel_coord.ra - on_coord.ra)

    denominator = np.cos(tel_coord.dec) * np.sin(tel_coord.ra - on_coord.ra)

    rotations = np.arctan2(numerator, denominator)
    rotations[rotations < 0] += u.Quantity(360, u.deg)

    # Define the wobble and OFF coordinates:
    mean_offset = offsets.to(u.deg).mean()
    mean_rot = rotations.to(u.deg).mean()

    skyoffset_frame = SkyOffsetFrame(origin=on_coord, rotation=-mean_rot)

    wobble_coord = SkyCoord(mean_offset, u.Quantity(0, u.deg), frame=skyoffset_frame)
    wobble_coord = wobble_coord.transform_to('icrs')

    rotations_off = np.arange(0, 359, 360/(n_off_regions + 1))
    rotations_off = rotations_off[rotations_off != 180]
    rotations_off += mean_rot.value

    theta_off = {}
    off_coords = {}

    for i_off, rot in enumerate(rotations_off):

        skyoffset_frame = SkyOffsetFrame(origin=wobble_coord, rotation=u.Quantity(-rot, u.deg))

        off_coords[i_off+1] = SkyCoord(mean_offset, u.Quantity(0, u.deg), frame=skyoffset_frame)
        off_coords[i_off+1] = off_coords[i_off+1].transform_to('icrs')

        # Compute the distance from the OFF region:
        theta_off[i_off+1] = off_coords[i_off+1].separation(event_coord)

    return theta_on, theta_off, off_coords


def transform_altaz_to_radec(alt, az, timestamp):
    """
    Transforms an AltAz direction measured from ORM
    to the RaDec coordinate by using a telescope timestamp.

    Parameters
    ----------
    alt: astropy.units.quantity.Quantity
        Altitude measured from ORM
    az: astropy.units.quantity.Quantity
        Azimuth measured from ORM
    timestamp: astropy.units.quantity.Quantity
        Timestamp when the direction was measured

    Returns
    -------
    ra: astropy.units.quantity.Quantity
        Right ascension of the input direction
    dec: astropy.units.quantity.Quantity
        Declination of the input direction
    """

    # Hardcode the longitude/latitude of ORM:
    lat_orm = u.Quantity(28.76177, u.deg)
    lon_orm = u.Quantity(-17.89064, u.deg)
    height_orm = u.Quantity(2199.835, u.m)

    location = EarthLocation.from_geodetic(lon=lon_orm, lat=lat_orm, height=height_orm)
    horizon_frames = AltAz(location=location, obstime=timestamp)

    event_coord = SkyCoord(alt=alt, az=az, frame=horizon_frames)
    event_coord = event_coord.transform_to('icrs')

    ra = event_coord.ra
    dec = event_coord.dec

    return ra, dec


def check_tel_combination(input_data):
    """
    Checks the telescope combination types of input events
    and returns a pandas data frame of the types.

    Parameters
    ----------
    input_data: pandas.core.frame.DataFrame
        Pandas data frame containing shower events

    Returns
    -------
    combo_type: pandas.core.frame.DataFrame
        Pandas data frame containing the telescope combination types
    """

    tel_combinations = {
        'm1_m2': [2, 3],   # combo_type = 0
        'lst1_m1': [1, 2],   # combo_type = 1
        'lst1_m2': [1, 3],   # combo_type = 2
        'lst1_m1_m2': [1, 2, 3],   # combo_type = 3
    }

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


def save_pandas_to_table(input_data, output_file, group_name, table_name):
    """
    Saves a pandas data frame to a table.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing shower events
    output_file: str
        Path to an output HDF file
    group_name: str
        Group name of the output table
    table_name: str
        Name of the output table
    """

    with tables.open_file(output_file, mode='a') as f_out:

        event_values = [tuple(array) for array in input_data.to_numpy()]
        dtypes = np.dtype([(name, dtype) for name, dtype in zip(input_data.dtypes.index, input_data.dtypes)])

        event_table = np.array(event_values, dtype=dtypes)
        f_out.create_table(group_name, table_name, createparents=True, obj=event_table)


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

    is_simulation = ('true_energy' in input_data.columns)
    groupby_mean = input_data.groupby(['obs_id', 'event_id']).mean()

    # Compute the mean of the gammaness:
    gammaness_mean = groupby_mean['gammaness']

    # Compute the mean of the reconstructed energies:
    weights = 1 / input_data['reco_energy_err']
    weighted_energy = np.log10(input_data['reco_energy']) * weights

    weights_sum = weights.groupby(['obs_id', 'event_id']).sum()
    weighted_energy_sum = weighted_energy.groupby(['obs_id', 'event_id']).sum()

    reco_energy_mean = 10 ** (weighted_energy_sum / weights_sum)

    # Compute the mean of the reconstructed arrival directions:
    reco_az_mean, reco_alt_mean = calc_mean_direction(
        lon=np.deg2rad(input_data['reco_az']),
        lat=np.deg2rad(input_data['reco_alt']),
        weights=input_data['reco_disp_err'],
    )

    # Compute the mean of the telescope pointing directions:
    pointing_az_mean, pointing_alt_mean = calc_mean_direction(
        lon=input_data['pointing_az'], lat=input_data['pointing_alt'],
    )

    # Create a base data frame:
    dl2_mean = pd.DataFrame(
        data={'combo_type': groupby_mean['combo_type'].to_numpy(),
              'gammaness': gammaness_mean.to_numpy(),
              'reco_energy': reco_energy_mean.to_numpy(),
              'reco_alt': reco_alt_mean.to(u.deg).value,
              'reco_az': reco_az_mean.to(u.deg).value,
              'pointing_alt': pointing_alt_mean.to(u.rad).value,
              'pointing_az': pointing_az_mean.to(u.rad).value},
        index=groupby_mean.index,
    )

    if is_simulation:
        # Add the MC parameters:
        mc_params = groupby_mean[['true_energy', 'true_alt', 'true_az']]
        dl2_mean = dl2_mean.join(mc_params)

    else:
        # Compute the mean of the Ra/Dec directions:
        reco_ra_mean, reco_dec_mean = calc_mean_direction(
            lon=np.deg2rad(input_data['reco_ra']),
            lat=np.deg2rad(input_data['reco_dec']),
            weights=input_data['reco_disp_err'],
        )

        pointing_ra_mean, pointing_dec_mean = calc_mean_direction(
            lon=np.deg2rad(input_data['pointing_ra']),
            lat=np.deg2rad(input_data['pointing_dec']),
        )

        # Add the additional parameters:
        df = pd.DataFrame(
            data={'reco_ra': reco_ra_mean.to(u.deg).value,
                  'reco_dec': reco_dec_mean.to(u.deg).value,
                  'pointing_ra': pointing_ra_mean.to(u.deg).value,
                  'pointing_dec': pointing_dec_mean.to(u.deg).value,
                  'timestamp': groupby_mean['timestamp'].to_numpy()},
            index=groupby_mean.index,
        )

        dl2_mean = dl2_mean.join(df)

    return dl2_mean