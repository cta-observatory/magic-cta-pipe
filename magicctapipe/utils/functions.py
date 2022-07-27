#!/usr/bin/env python
# coding: utf-8

import logging

import numpy as np
import pandas as pd
import tables
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, angular_separation
from astropy.coordinates.builtin_frames import SkyOffsetFrame
from astropy.time import Time
from ctapipe.coordinates import TelescopeFrame

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [
    "calculate_disp",
    "calculate_impact",
    "calculate_mean_direction",
    "calculate_angular_distance",
    "transform_altaz_to_radec",
    "check_tel_combination",
    "save_pandas_to_table",
    "get_dl2_mean",
]


def calculate_disp(
    pointing_alt, pointing_az, shower_alt, shower_az, cog_x, cog_y, camera_frame
):
    """
    Calculates the DISP parameter.

    Parameters
    ----------
    pointing_alt: astropy.units.quantity.Quantity
        Altitude of a telescope pointing direction
    pointing_az: astropy.units.quantity.Quantity
        Azimuth of a telescope pointing direction
    shower_alt: astropy.units.quantity.Quantity
        Altitude of a shower arrival direction
    shower_az: astropy.units.quantity.Quantity
        Azimuth of a shower arrival direction
    cog_x: astropy.units.quantity.Quantity
        Image CoG along with a camera geometry X coordinate
    cog_y: astropy.units.quantity.Quantity
        Image CoG along with a camera geometry Y coordinate
    camera_frame: ctapipe.coordinates.camera_frame.CameraFrame
        Telescope camera frame

    Returns
    -------
    disp: astropy.units.quantity.Quantity
        Angular distance between an image CoG and an event arrival direction
    """

    tel_pointing = AltAz(alt=pointing_alt, az=pointing_az)
    tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

    event_coord = SkyCoord(cog_x, cog_y, frame=camera_frame)
    event_coord = event_coord.transform_to(tel_frame)

    disp = angular_separation(
        lon1=event_coord.altaz.az,
        lat1=event_coord.altaz.alt,
        lon2=shower_az,
        lat2=shower_alt,
    )

    return disp


def calculate_impact(core_x, core_y, az, alt, tel_pos_x, tel_pos_y, tel_pos_z):
    """
    Calculates the impact distance from a given telescope.

    Parameters
    ----------
    core_x: astropy.units.quantity.Quantity
        Core position along the geographical north
    core_y: astropy.units.quantity.Quantity
        Core position along the geographical west
    az: astropy.units.quantity.Quantity
        Azimuth of an event arrival direction
    alt: astropy.units.quantity.Quantity
        Altitude of an event arrival direction
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

    t = (
        (tel_pos_x - core_x) * np.cos(alt) * np.cos(az)
        - (tel_pos_y - core_y) * np.cos(alt) * np.sin(az)
        + tel_pos_z * np.sin(alt)
    )

    impact = np.sqrt(
        (core_x - tel_pos_x + t * np.cos(alt) * np.cos(az)) ** 2
        + (core_y - tel_pos_y - t * np.cos(alt) * np.sin(az)) ** 2
        + (t * np.sin(alt) - tel_pos_z) ** 2
    )

    return impact


def calculate_mean_direction(lon, lat, weights=None):
    """
    Calculates the mean of input directions in a spherical coordinate.
    The inputs should be the "Series" type with the indices of 'obs_id' and 'event_id'.
    The unit of the input longitude and latitude should be radian.

    Parameters
    ----------
    lon: pandas.core.series.Series
        Longitude in a spherical coordinate
    lat: pandas.core.series.Series
        Latitude in a spherical coordinate
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

        weights_sum = weights.groupby(["obs_id", "event_id"]).sum()

        weighted_x_coords_sum = weighted_x_coords.groupby(["obs_id", "event_id"]).sum()
        weighted_y_coords_sum = weighted_y_coords.groupby(["obs_id", "event_id"]).sum()
        weighted_z_coords_sum = weighted_z_coords.groupby(["obs_id", "event_id"]).sum()

        x_coord_mean = weighted_x_coords_sum / weights_sum
        y_coord_mean = weighted_y_coords_sum / weights_sum
        z_coord_mean = weighted_z_coords_sum / weights_sum

    else:
        x_coord_mean = x_coords.groupby(["obs_id", "event_id"]).sum()
        y_coord_mean = y_coords.groupby(["obs_id", "event_id"]).sum()
        z_coord_mean = z_coords.groupby(["obs_id", "event_id"]).sum()

    coord_mean = SkyCoord(
        x=x_coord_mean.values,
        y=y_coord_mean.values,
        z=z_coord_mean.values,
        representation_type="cartesian",
    )

    lon_mean = coord_mean.spherical.lon
    lat_mean = coord_mean.spherical.lat

    return lon_mean, lat_mean


def calculate_angular_distance(on_coord, event_coord, tel_coord, n_off_regions):
    """
    Calculates the angular distance between
    the shower arrival direction and ON/OFF regions.

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
    offsets = np.arccos(
        np.cos(on_coord.dec)
        * np.cos(tel_coord.dec)
        * np.cos(tel_coord.ra - on_coord.ra)
        + np.sin(on_coord.dec) * np.sin(tel_coord.dec)
    )

    numerator = np.sin(tel_coord.dec) * np.cos(on_coord.dec) - np.sin(
        on_coord.dec
    ) * np.cos(tel_coord.dec) * np.cos(tel_coord.ra - on_coord.ra)

    denominator = np.cos(tel_coord.dec) * np.sin(tel_coord.ra - on_coord.ra)

    rotations = np.arctan2(numerator, denominator)
    rotations[rotations < 0] += u.Quantity(360, u.deg)

    # Define the wobble and OFF coordinates:
    mean_offset = offsets.to(u.deg).mean()
    mean_rot = rotations.to(u.deg).mean()

    skyoffset_frame = SkyOffsetFrame(origin=on_coord, rotation=-mean_rot)

    wobble_coord = SkyCoord(mean_offset, u.Quantity(0, u.deg), frame=skyoffset_frame)
    wobble_coord = wobble_coord.transform_to("icrs")

    rotations_off = np.arange(0, 359, 360 / (n_off_regions + 1))
    rotations_off = rotations_off[rotations_off != 180]
    rotations_off += mean_rot.value

    theta_off = {}
    off_coords = {}

    for i_off, rot in enumerate(rotations_off, start=1):

        skyoffset_frame = SkyOffsetFrame(
            origin=wobble_coord, rotation=u.Quantity(-rot, u.deg)
        )

        off_coords[i_off] = SkyCoord(
            mean_offset, u.Quantity(0, u.deg), frame=skyoffset_frame
        )

        off_coords[i_off] = off_coords[i_off].transform_to("icrs")

        # Compute the distance from the OFF region:
        theta_off[i_off] = off_coords[i_off].separation(event_coord)

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
    LON_ORM = u.Quantity(-17.89064, u.deg)
    LAT_ORM = u.Quantity(28.76177, u.deg)
    HEIGHT_ORM = u.Quantity(2199.835, u.m)

    location = EarthLocation.from_geodetic(lon=LON_ORM, lat=LAT_ORM, height=HEIGHT_ORM)
    horizon_frames = AltAz(location=location, obstime=timestamp)

    event_coord = SkyCoord(alt=alt, az=az, frame=horizon_frames)
    event_coord = event_coord.transform_to("icrs")

    ra = event_coord.ra
    dec = event_coord.dec

    return ra, dec


def check_tel_combination(event_data):
    """
    Checks the telescope combination types of input events
    and returns a pandas data frame of the types.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events

    Returns
    -------
    combo_type: pandas.core.frame.DataFrame
        Pandas data frame of the telescope combination types
    """

    tel_combinations = {
        "m1_m2": [2, 3],  # combo_type = 0
        "lst1_m1": [1, 2],  # combo_type = 1
        "lst1_m2": [1, 3],  # combo_type = 2
        "lst1_m1_m2": [1, 2, 3],  # combo_type = 3
    }

    combo_types = pd.DataFrame()

    n_events_total = len(event_data.groupby(["obs_id", "event_id"]).size())
    logger.info(f"\nIn total {n_events_total} stereo events are found:")

    for combo_type, (tel_combo, tel_ids) in enumerate(tel_combinations.items()):

        df_events = event_data.query(
            f"(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})"
        )

        group_size = df_events.groupby(["obs_id", "event_id"]).size()
        group_size = group_size[group_size == len(tel_ids)]

        n_events = len(group_size)
        ratio = n_events / n_events_total

        logger.info(
            f"\t{tel_combo} (type {combo_type}): "
            f"{n_events:.0f} events ({ratio * 100:.1f}%)"
        )

        df_combo_type = pd.DataFrame({"combo_type": combo_type}, index=group_size.index)
        combo_types = combo_types.append(df_combo_type)

    combo_types.sort_index(inplace=True)

    return combo_types


def save_pandas_to_table(event_data, output_file, group_name, table_name, mode="w"):
    """
    Saves a pandas data frame in a table.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame containing shower events
    output_file: str
        Path to an output HDF file
    group_name: str
        Group name of the output table
    table_name: str
        Name of the output table
    mode: str
        Mode of opening a table, 'w' for overwriting and 'a' for appending
    """

    with tables.open_file(output_file, mode=mode) as f_out:

        values = [tuple(array) for array in event_data.to_numpy()]
        dtypes = np.dtype([(dtype.index, dtype) for dtype in event_data.dtypes])

        event_table = np.array(values, dtype=dtypes)
        f_out.create_table(group_name, table_name, createparents=True, obj=event_table)


def get_dl2_mean(event_data, weight=None):
    """
    Calculates the mean of the tel-wise DL2 parameters.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of the DL2 parameters
    weight: str
        Type of the weight for the tel-wise parameters

    Returns
    -------
    dl2_mean: pandas.core.frame.DataFrame
        Pandas data frame of the mean of the DL2 parameters
    """

    is_simulation = "true_energy" in event_data.columns

    if weight is None:
        index = event_data.index
        gammaness_weights = pd.Series(np.repeat(1, len(event_data)), index=index)
        energy_weights = pd.Series(np.repeat(1, len(event_data)), index=index)
        direction_weights = pd.Series(np.repeat(1, len(event_data)), index=index)

    elif weight == "var":
        gammaness_weights = 1 / event_data["gammaness_var"]
        energy_weights = 1 / event_data["reco_energy_var"]
        direction_weights = 1 / event_data["reco_disp_var"]

    elif weight == "intensity":
        gammaness_weights = event_data["intensity"]
        energy_weights = event_data["intensity"]
        direction_weights = event_data["intensity"]

    else:
        RuntimeError(f'Unknown weight type "{weight}".')

    # Compute the mean of the gammaness:
    weighted_gammaness = event_data["gammaness"] * gammaness_weights

    gammaness_weights_sum = gammaness_weights.groupby(["obs_id", "event_id"]).sum()
    weighted_gammaness_sum = weighted_gammaness.groupby(["obs_id", "event_id"]).sum()

    gammaness_mean = weighted_gammaness_sum / gammaness_weights_sum

    # Compute the mean of the reconstructed energies:
    weighted_energy = np.log10(event_data["reco_energy"]) * energy_weights

    energy_weights_sum = energy_weights.groupby(["obs_id", "event_id"]).sum()
    weighted_energy_sum = weighted_energy.groupby(["obs_id", "event_id"]).sum()

    reco_energy_mean = 10 ** (weighted_energy_sum / energy_weights_sum)

    # Compute the mean of the reconstructed arrival directions:
    reco_az_mean, reco_alt_mean = calculate_mean_direction(
        lon=np.deg2rad(event_data["reco_az"]),
        lat=np.deg2rad(event_data["reco_alt"]),
        weights=direction_weights,
    )

    # Compute the mean of the telescope pointing directions:
    pointing_az_mean, pointing_alt_mean = calculate_mean_direction(
        lon=event_data["pointing_az"], lat=event_data["pointing_alt"]
    )

    # Create a mean data frame:
    group_mean = event_data.groupby(["obs_id", "event_id"]).mean()

    dl2_mean = pd.DataFrame(
        data={
            "combo_type": group_mean["combo_type"].to_numpy(),
            "multiplicity": group_mean["multiplicity"].to_numpy(),
            "gammaness": gammaness_mean.to_numpy(),
            "reco_energy": reco_energy_mean.to_numpy(),
            "reco_alt": reco_alt_mean.to(u.deg).value,
            "reco_az": reco_az_mean.to(u.deg).value,
            "pointing_alt": pointing_alt_mean.to(u.rad).value,
            "pointing_az": pointing_az_mean.to(u.rad).value,
        },
        index=group_mean.index,
    )

    if is_simulation:
        # Add the MC parameters:
        mc_params = group_mean[["true_energy", "true_alt", "true_az"]]
        dl2_mean = dl2_mean.join(mc_params)

    else:
        # Convert the mean Alt/Az to the RA/Dec coordinate:
        timestamps = Time(
            group_mean["timestamp"].to_numpy(), format="unix", scale="utc"
        )

        reco_ra_mean, reco_dec_mean = transform_altaz_to_radec(
            alt=reco_alt_mean, az=reco_az_mean, timestamp=timestamps
        )

        pointing_ra_mean, pointing_dec_mean = transform_altaz_to_radec(
            alt=pointing_alt_mean, az=pointing_az_mean, timestamp=timestamps
        )

        # Add the additional parameters:
        radec_time = pd.DataFrame(
            data={
                "reco_ra": reco_ra_mean.to(u.deg).value,
                "reco_dec": reco_dec_mean.to(u.deg).value,
                "pointing_ra": pointing_ra_mean.to(u.deg).value,
                "pointing_dec": pointing_dec_mean.to(u.deg).value,
                "timestamp": group_mean["timestamp"].to_numpy(),
            },
            index=group_mean.index,
        )

        dl2_mean = dl2_mean.join(radec_time)

    return dl2_mean
