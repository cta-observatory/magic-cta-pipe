#!/usr/bin/env python
# coding: utf-8

import logging

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    Angle,
    EarthLocation,
    SkyCoord,
    SkyOffsetFrame,
    angular_separation,
)
from ctapipe.coordinates import TelescopeFrame

__all__ = [
    "calculate_disp",
    "calculate_impact",
    "calculate_mean_direction",
    "calculate_pointing_separation",
    "calculate_off_coordinates",
    "calculate_dead_time_correction",
    "transform_altaz_to_radec",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The telescope combination types
TEL_COMBINATIONS = {
    "m1_m2": [2, 3],  # combo_type = 0
    "lst1_m1": [1, 2],  # combo_type = 1
    "lst1_m2": [1, 3],  # combo_type = 2
    "lst1_m1_m2": [1, 2, 3],  # combo_type = 3
}

# The pandas index to group up shower events
GROUP_INDEX = ["obs_id", "event_id"]

# The LST/MAGIC readout dead times
DEAD_TIME_LST = 7.6 * u.us
DEAD_TIME_MAGIC = 26 * u.us

# The upper limit of event time differences used when calculating
# the dead time correction factor
TIME_DIFF_UPLIM = 0.1 * u.s

# The geographical coordinate of ORM
LON_ORM = u.Quantity(-17.89064, u.deg)
LAT_ORM = u.Quantity(28.76177, u.deg)
HEIGHT_ORM = u.Quantity(2199.835, u.m)


@u.quantity_input
def calculate_disp(
    pointing_alt: u.rad,
    pointing_az: u.rad,
    shower_alt: u.deg,
    shower_az: u.deg,
    cog_x: u.m,
    cog_y: u.m,
    camera_frame,
):
    """
    Calculates the DISP parameter, i.e., the angular distance between
    an event arrival direction and the center of gravity (CoG) of the
    shower image.

    Parameters
    ----------
    pointing_alt: astropy.units.quantity.Quantity
        Altitude of the telescope pointing direction
    pointing_az: astropy.units.quantity.Quantity
        Azimuth of the telescope pointing direction
    shower_alt: astropy.units.quantity.Quantity
        Altitude of the event arrival direction
    shower_az: astropy.units.quantity.Quantity
        Azimuth of the event arrival direction
    cog_x: astropy.units.quantity.Quantity
        Image CoG along the X coordinate of the camera geometry
    cog_y: astropy.units.quantity.Quantity
        Image CoG along the Y coordinate of the camera geometry
    camera_frame: ctapipe.coordinates.camera_frame.CameraFrame
        Camera frame of the telescope

    Returns
    -------
    disp: astropy.units.quantity.Quantity
        DISP parameter
    """

    # Transform the image CoG position to the Alt/Az direction
    tel_pointing = AltAz(alt=pointing_alt, az=pointing_az)
    tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

    cog_coord = SkyCoord(cog_x, cog_y, frame=camera_frame)
    cog_coord = cog_coord.transform_to(tel_frame).altaz

    # Calculate the DISP parameter
    disp = angular_separation(
        lon1=cog_coord.az, lat1=cog_coord.alt, lon2=shower_az, lat2=shower_alt
    )

    return disp


@u.quantity_input
def calculate_impact(
    shower_alt: u.deg,
    shower_az: u.deg,
    core_x: u.m,
    core_y: u.m,
    tel_pos_x: u.m,
    tel_pos_y: u.m,
    tel_pos_z: u.m,
):
    """
    Calculates the impact distance, i.e., the closest distance between
    a shower axis and a telescope position.

    It uses equations derived from a hand calculation, but it is
    confirmed that the result is consistent with what is done in MARS.

    In ctapipe v0.16.0 the function to calculate the impact distance is
    implemented, so we may replace it to the official one in future.

    Parameters
    ----------
    shower_alt: astropy.units.quantity.Quantity
        Altitude of the event arrival direction
    shower_az: astropy.units.quantity.Quantity
        Azimuth of the event arrival direction
    core_x: astropy.units.quantity.Quantity
        Core position along the geographical north
    core_y: astropy.units.quantity.Quantity
        Core position along the geographical west
    tel_pos_x: astropy.units.quantity.Quantity
        Telescope position along the geographical north
    tel_pos_y: astropy.units.quantity.Quantity
        Telescope position along the geographical west
    tel_pos_z: astropy.units.quantity.Quantity
        Telescope height from the reference altitude

    Returns
    -------
    impact: astropy.units.quantity.Quantity
        Impact distance
    """

    diff_x = tel_pos_x - core_x
    diff_y = tel_pos_y - core_y

    param = (
        diff_x * np.cos(shower_alt) * np.cos(shower_az)
        - diff_y * np.cos(shower_alt) * np.sin(shower_az)
        + tel_pos_z * np.sin(shower_alt)
    )

    impact = np.sqrt(
        (param * np.cos(shower_alt) * np.cos(shower_az) - diff_x) ** 2
        + (param * np.cos(shower_alt) * np.sin(shower_az) + diff_y) ** 2
        + (param * np.sin(shower_alt) - tel_pos_z) ** 2
    )

    return impact


def calculate_mean_direction(lon, lat, weights=None, unit="rad"):
    """
    Calculates the mean direction per shower event.

    The input data is supposed to be the pandas Series with the
    index (obs_id, event_id) to group up the shower events.

    Parameters
    ----------
    lon: pandas.core.series.Series
        Longitude in a spherical coordinate
    lat: pandas.core.series.Series
        Latitude in a spherical coordinate
    weights: pandas.core.series.Series
        Weights for the input directions
    unit: str
        Unit of the input (and output) angles

    Returns
    -------
    lon_mean: pandas.core.series.Series
        Longitude of the mean direction
    lat_mean: pandas.core.series.Series
        Latitude of the mean direction
    """

    if unit in ["deg", "degree"]:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)

    # Transform the input directions to the cartesian coordinate and
    # then calculate the mean position for each axis
    x_coords = np.cos(lat) * np.cos(lon)
    y_coords = np.cos(lat) * np.sin(lon)
    z_coords = np.sin(lat)

    if weights is None:
        x_coord_mean = x_coords.groupby(GROUP_INDEX).mean()
        y_coord_mean = y_coords.groupby(GROUP_INDEX).mean()
        z_coord_mean = z_coords.groupby(GROUP_INDEX).mean()

    else:
        df_cartesian = pd.DataFrame(
            data={
                "weight": weights,
                "weighted_x_coord": x_coords * weights,
                "weighted_y_coord": y_coords * weights,
                "weighted_z_coord": z_coords * weights,
            }
        )

        group_sum = df_cartesian.groupby(GROUP_INDEX).sum()

        x_coord_mean = group_sum["weighted_x_coord"] / group_sum["weight"]
        y_coord_mean = group_sum["weighted_y_coord"] / group_sum["weight"]
        z_coord_mean = group_sum["weighted_z_coord"] / group_sum["weight"]

    coord_mean = SkyCoord(
        x=x_coord_mean.values,
        y=y_coord_mean.values,
        z=z_coord_mean.values,
        representation_type="cartesian",
    )

    # Transform the cartesian to the spherical coordinate
    coord_mean = coord_mean.spherical

    lon_mean = pd.Series(data=coord_mean.lon.to_value(unit), index=x_coord_mean.index)
    lat_mean = pd.Series(data=coord_mean.lat.to_value(unit), index=x_coord_mean.index)

    return lon_mean, lat_mean


def calculate_pointing_separation(event_data):
    """
    Calculates the angular distance of the LST-1 and MAGIC pointing
    directions.

    The input data is supposed to have the index
    (obs_id, event_id, tel_id).

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of LST-1 and MAGIC events

    Returns
    -------
    theta: astropy.units.quantity.Quantity
        Angular distance of the LST-1 and MAGIC pointing directions
    """

    df_lst = event_data.query("tel_id == 1")

    obs_ids = df_lst.index.get_level_values("obs_id").tolist()
    event_ids = df_lst.index.get_level_values("event_id").tolist()

    multi_indices = pd.MultiIndex.from_arrays(
        [obs_ids, event_ids], names=["obs_id", "event_id"]
    )

    df_magic = event_data.query("tel_id == [2, 3]")
    df_magic.reset_index(level="tel_id", inplace=True)
    df_magic = df_magic.loc[multi_indices]

    # Calculate the mean of the M1 and M2 pointing directions
    pointing_az_magic, pointing_alt_magic = calculate_mean_direction(
        lon=df_magic["pointing_az"], lat=df_magic["pointing_alt"]
    )

    theta = angular_separation(
        lon1=u.Quantity(df_lst["pointing_az"].to_numpy(), u.rad),
        lat1=u.Quantity(df_lst["pointing_alt"].to_numpy(), u.rad),
        lon2=u.Quantity(pointing_az_magic.to_numpy(), u.rad),
        lat2=u.Quantity(pointing_alt_magic.to_numpy(), u.rad),
    )

    return theta


@u.quantity_input
def calculate_off_coordinates(
    pointing_ra: u.deg,
    pointing_dec: u.deg,
    on_coord_ra: u.deg,
    on_coord_dec: u.deg,
    n_off_regions,
):
    """
    Gets the coordinates of OFF regions to estimate the backgrounds of
    wobble observation data.

    It calculates the wobble offset and rotation angle with equations
    derived from a hand calculation.

    Parameters
    ----------
    pointing_ra: astropy.units.quantity.Quantity
        Right ascension of the telescope pointing direction
    pointing_dec: astropy.units.quantity.Quantity
        Declination of the telescope pointing direction
    on_coord_ra: astropy.units.quantity.Quantity
        Right ascension of the center of the ON region
    on_coord_dec: astropy.units.quantity.Quantity
        Declination of the center of the ON region
    n_off_regions: int
        Number of OFF regions to be extracted

    Returns
    -------
    off_coords: dict
        Coordinates of the center of the OFF regions
    """

    ra_diff = pointing_ra - on_coord_ra

    # Calculate the wobble offset
    wobble_offset = np.arccos(
        np.cos(on_coord_dec) * np.cos(pointing_dec) * np.cos(ra_diff)
        + np.sin(on_coord_dec) * np.sin(pointing_dec)
    )

    logger.info(f"Wobble offset: {wobble_offset.to(u.deg).round(3)}")

    # Calculate the wobble rotation angle
    numerator_1 = np.sin(pointing_dec) * np.cos(on_coord_dec)
    numerator_2 = np.cos(pointing_dec) * np.sin(on_coord_dec) * np.cos(ra_diff)
    denominator = np.cos(pointing_dec) * np.sin(ra_diff)

    wobble_rotation = np.arctan2(numerator_1 - numerator_2, denominator)
    wobble_rotation = Angle(wobble_rotation).wrap_at(360 * u.deg)

    logger.info(f"Wobble rotation angle: {wobble_rotation.to(u.deg).round(3)}")

    # Compute the OFF coordinates
    wobble_coord = SkyCoord(pointing_ra, pointing_dec, frame="icrs")

    rotations_off = np.arange(0, 359, 360 / (n_off_regions + 1)) * u.deg
    rotations_off = rotations_off[rotations_off.to_value(u.deg) != 180]
    rotations_off += wobble_rotation

    off_coords = {}

    for i_off, rotation in enumerate(rotations_off, start=1):

        skyoffset_frame = SkyOffsetFrame(origin=wobble_coord, rotation=-rotation)
        off_coord = SkyCoord(wobble_offset, u.Quantity(0, u.deg), frame=skyoffset_frame)

        off_coords[i_off] = off_coord.transform_to("icrs")

    return off_coords


def calculate_dead_time_correction(event_data):
    """
    Calculates the dead time correction factor, i.e., the factor to
    estimate the effective time from the total observation time.

    It uses the following equations to get the correction factor
    "deadc", where <time_diff> is the mean of the trigger time
    differences of consecutive events:

    rate = 1 / (<time_diff> - dead_time)
    deadc = 1 / (1 + rate * dead_time) = 1 - dead_time / <time_diff>

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events

    Returns
    -------
    deadc_total: float
        Total dead time correction factor
    """

    df_events = event_data.query(f"0 < time_diff < {TIME_DIFF_UPLIM.to_value(u.s)}")

    logger.info("\nCalculating the dead time correction factor...")

    deadc_list = []

    # Calculate the LST-1 correction factor
    time_diffs_lst = df_events.query("tel_id == 1")["time_diff"]

    if len(time_diffs_lst) > 0:
        deadc_lst = 1 - DEAD_TIME_LST.to_value(u.s) / time_diffs_lst.mean()
        logger.info(f"LST-1: {deadc_lst.round(3)}")

        deadc_list.append(deadc_lst)

    # Calculate the MAGIC correction factor with one of the telescopes
    # whose number of events is larger than the other
    time_diffs_m1 = df_events.query("tel_id == 2")["time_diff"]
    time_diffs_m2 = df_events.query("tel_id == 3")["time_diff"]

    if len(time_diffs_m1) > len(time_diffs_m2):
        deadc_magic = 1 - DEAD_TIME_MAGIC.to_value(u.s) / time_diffs_m1.mean()
        logger.info(f"MAGIC-I: {deadc_magic.round(3)}")
    else:
        deadc_magic = 1 - DEAD_TIME_MAGIC.to_value(u.s) / time_diffs_m2.mean()
        logger.info(f"MAGIC-II: {deadc_magic.round(3)}")

    deadc_list.append(deadc_magic)

    # Calculate the total correction factor as the multiplicity of the
    # telescope-wise correction factors
    deadc_total = np.prod(deadc_list)

    logger.info(f"--> Total correction factor: {deadc_total.round(3)}")

    return deadc_total


@u.quantity_input
def transform_altaz_to_radec(alt: u.deg, az: u.deg, obs_time):
    """
    Transforms the Alt/Az direction measured from ORM to the RA/Dec
    coordinate by using the observation time.

    Parameters
    ----------
    alt: astropy.units.quantity.Quantity
        Altitude measured from ORM
    az: astropy.units.quantity.Quantity
        Azimuth measured from ORM
    obs_time: astropy.time.core.Time
        Observation time when the direction was measured

    Returns
    -------
    ra: astropy.coordinates.angles.Longitude
        Right ascension of the input direction
    dec: astropy.coordinates.angles.Latitude
        Declination of the input direction
    """

    location = EarthLocation.from_geodetic(lon=LON_ORM, lat=LAT_ORM, height=HEIGHT_ORM)
    horizon_frames = AltAz(location=location, obstime=obs_time)

    event_coord = SkyCoord(alt=alt, az=az, frame=horizon_frames)
    event_coord = event_coord.transform_to("icrs")

    ra = event_coord.ra
    dec = event_coord.dec

    return ra, dec
