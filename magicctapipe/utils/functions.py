#!/usr/bin/env python
# coding: utf-8

import tables
import logging
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.coordinates.builtin_frames import SkyOffsetFrame

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

tel_combinations = {
    'm1_m2': [2, 3],   # event_type = 0
    'lst1_m1': [1, 2],   # event_type = 1
    'lst1_m2': [1, 3],   # event_type = 2
    'lst1_m1_m2': [1, 2, 3],   # event_type = 3
}

__all__ = [
    'set_event_types',
    'save_data_to_hdf',
    'calc_impact',
    'calc_nsim',
    'transform_to_radec',
    'calc_angular_separation',
]


def set_event_types(data):

    n_events_total = len(data.groupby(['obs_id', 'event_id']).size())
    logger.info(f'\nIn total {n_events_total} stereo events are found:')

    for event_type, (tel_combo, tel_ids) in enumerate(tel_combinations.items()):

        df = data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        df['multiplicity'] = df.groupby(['obs_id', 'event_id']).size()
        df.query(f'multiplicity == {len(tel_ids)}', inplace=True)

        n_events = len(df.groupby(['obs_id', 'event_id']).size())
        logger.info(f'{tel_combo} (type {event_type}): {n_events:.0f} events ({n_events / n_events_total * 100:.1f}%)')

        data.loc[df.index, 'event_type'] = event_type

    return data


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
