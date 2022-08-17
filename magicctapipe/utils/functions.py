#!/usr/bin/env python
# coding: utf-8

import logging

import numpy as np
import pandas as pd
import tables
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    Angle,
    EarthLocation,
    SkyCoord,
    SkyOffsetFrame,
    angular_separation,
)
from astropy.time import Time
from ctapipe.coordinates import TelescopeFrame

__all__ = [
    "calculate_disp",
    "calculate_impact",
    "calculate_mean_direction",
    "transform_altaz_to_radec",
    "get_dl2_mean",
    "get_off_regions",
    "get_stereo_events",
    "save_pandas_to_table",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The geographical coordinate of ORM:
LON_ORM = u.Quantity(-17.89064, u.deg)
LAT_ORM = u.Quantity(28.76177, u.deg)
HEIGHT_ORM = u.Quantity(2199.835, u.m)

# The combinations of the telescope IDs:
TEL_COMBINATIONS = {
    "m1_m2": [2, 3],  # combo_type = 0
    "lst1_m1": [1, 2],  # combo_type = 1
    "lst1_m2": [1, 3],  # combo_type = 2
    "lst1_m1_m2": [1, 2, 3],  # combo_type = 3
}

# The pandas index to group up shower events:
GROUP_INDEX = ["obs_id", "event_id"]


def calculate_disp(
    pointing_alt, pointing_az, shower_alt, shower_az, cog_x, cog_y, camera_frame
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
        Calculated DISP parameter
    """

    tel_pointing = AltAz(alt=pointing_alt, az=pointing_az)
    tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

    cog_coord = SkyCoord(cog_x, cog_y, frame=camera_frame)
    cog_coord = cog_coord.transform_to(tel_frame).altaz

    disp = angular_separation(
        lon1=cog_coord.az, lat1=cog_coord.alt, lon2=shower_az, lat2=shower_alt
    )

    return disp


def calculate_impact(
    shower_alt, shower_az, core_x, core_y, tel_pos_x, tel_pos_y, tel_pos_z
):
    """
    Calculates the impact distance, i.e., the closest distance between
    a shower axis and a telescope position.

    It uses a formula derived from a hand calculation, but it is
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
        Core position of the shower axis along the geographical north
    core_y: astropy.units.quantity.Quantity
        Core position of the shower axis along the geographical west
    tel_pos_x: astropy.units.quantity.Quantity
        Telescope position along the geographical north
    tel_pos_y: astropy.units.quantity.Quantity
        Telescope position along the geographical west
    tel_pos_z: astropy.units.quantity.Quantity
        Telescope height from the reference altitude

    Returns
    -------
    impact: astropy.units.quantity.Quantity
        Calculated impact distance
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
    Calculates the mean of input directions per shower event.

    The input data is supposed to be the pandas Series with the
    index (obs_id, event_id) to group up the shower events.

    Parameters
    ----------
    lon: pandas.core.series.Series
        Longitude in a spherical coordinate
    lat: pandas.core.series.Series
        Latitude in a spherical coordinate
    weights: pandas.core.series.Series
        Weights applied when calculating the mean direction
    unit: str
        Unit of the input (and output) longitude and latitude -
        "rad", "radian", "deg" or "degree" are allowed

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

    x_coords = np.cos(lat) * np.cos(lon)
    y_coords = np.cos(lat) * np.sin(lon)
    z_coords = np.sin(lat)

    if weights is None:
        weights = pd.Series(data=1, index=lon.index)

    weighted_x_coords = x_coords * weights
    weighted_y_coords = y_coords * weights
    weighted_z_coords = z_coords * weights

    weighted_x_coords_sum = weighted_x_coords.groupby(GROUP_INDEX).sum()
    weighted_y_coords_sum = weighted_y_coords.groupby(GROUP_INDEX).sum()
    weighted_z_coords_sum = weighted_z_coords.groupby(GROUP_INDEX).sum()

    weights_sum = weights.groupby(GROUP_INDEX).sum()

    x_coord_mean = weighted_x_coords_sum / weights_sum
    y_coord_mean = weighted_y_coords_sum / weights_sum
    z_coord_mean = weighted_z_coords_sum / weights_sum

    coord_mean = SkyCoord(
        x=x_coord_mean.values,
        y=y_coord_mean.values,
        z=z_coord_mean.values,
        representation_type="cartesian",
    )

    coord_mean = coord_mean.spherical

    lon_mean = pd.Series(data=coord_mean.lon.to_value(unit), index=weights_sum.index)
    lat_mean = pd.Series(data=coord_mean.lat.to_value(unit), index=weights_sum.index)

    return lon_mean, lat_mean


def transform_altaz_to_radec(alt, az, obs_time):
    """
    Transforms the AltAz direction measured from ORM to the RaDec
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


def get_dl2_mean(event_data, weight_type=None):
    """
    Calculates the mean of the tel-wise DL2 parameters per shower event,
    and returns them as a data frame with some additional parameters.

    The input data is supposed to have the index (obs_id, event_id) to
    group up the shower events.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events
    weight_type: str
        Type of the weights for averaging the tel-wise parameters,
        'variance' to use the inverse of the RF variance, and
        'intensity' to use the linear-scale intensity parameter

    Returns
    -------
    event_data_mean: pandas.core.frame.DataFrame
        Pandas data frame of shower events with the mean parameters
    """

    is_simulation = "true_energy" in event_data.columns

    if weight_type is None:
        energy_weights = pd.Series(data=1, index=event_data.index)
        direction_weights = pd.Series(data=1, index=event_data.index)
        gammaness_weights = pd.Series(data=1, index=event_data.index)

    elif weight_type == "variance":
        energy_weights = 1 / event_data["reco_energy_var"]
        direction_weights = 1 / event_data["reco_disp_var"]
        gammaness_weights = 1 / event_data["gammaness_var"]

    elif weight_type == "intensity":
        energy_weights = event_data["intensity"]
        direction_weights = event_data["intensity"]
        gammaness_weights = event_data["intensity"]

    else:
        raise RuntimeError(
            f"Unknown weight type '{weight_type}'. Select 'variance' or 'intensity'."
        )

    # Calculate the mean of the reconstructed energies in log scale:
    weighted_energy = np.log10(event_data["reco_energy"]) * energy_weights

    energy_weights_sum = energy_weights.groupby(GROUP_INDEX).sum()
    weighted_energy_sum = weighted_energy.groupby(GROUP_INDEX).sum()

    reco_energy_mean = 10 ** (weighted_energy_sum / energy_weights_sum)

    # Calculate the mean of the reconstructed arrival directions:
    reco_az_mean, reco_alt_mean = calculate_mean_direction(
        lon=event_data["reco_az"],
        lat=event_data["reco_alt"],
        weights=direction_weights,
        unit="deg",
    )

    # Calculate the mean of the gammaness:
    weighted_gammaness = event_data["gammaness"] * gammaness_weights

    gammaness_weights_sum = gammaness_weights.groupby(GROUP_INDEX).sum()
    weighted_gammaness_sum = weighted_gammaness.groupby(GROUP_INDEX).sum()

    gammaness_mean = weighted_gammaness_sum / gammaness_weights_sum

    # Create a mean data frame:
    group_mean = event_data.groupby(GROUP_INDEX).mean()

    pointing_az_mean, pointing_alt_mean = calculate_mean_direction(
        lon=event_data["pointing_az"], lat=event_data["pointing_alt"]
    )

    event_data_mean = pd.DataFrame(
        data={
            "combo_type": group_mean["combo_type"].to_numpy(),
            "multiplicity": group_mean["multiplicity"].to_numpy(),
            "pointing_alt": pointing_alt_mean.to_numpy(),
            "pointing_az": pointing_az_mean.to_numpy(),
            "reco_energy": reco_energy_mean.to_numpy(),
            "reco_alt": reco_alt_mean.to_numpy(),
            "reco_az": reco_az_mean.to_numpy(),
            "gammaness": gammaness_mean.to_numpy(),
        },
        index=group_mean.index,
    )

    if is_simulation:
        # Add the MC parameters:
        df_mc_mean = group_mean[["true_energy", "true_alt", "true_az"]]
        event_data_mean = event_data_mean.join(df_mc_mean)

    else:
        timestamps_mean = Time(
            group_mean["timestamp"].to_numpy(), format="unix", scale="utc"
        )

        # Convert the mean Alt/Az to the RA/Dec coordinate:
        reco_ra_mean, reco_dec_mean = transform_altaz_to_radec(
            alt=u.Quantity(reco_alt_mean.to_numpy(), u.deg),
            az=u.Quantity(reco_az_mean.to_numpy(), u.deg),
            obs_time=timestamps_mean,
        )

        pointing_ra_mean, pointing_dec_mean = transform_altaz_to_radec(
            alt=u.Quantity(pointing_alt_mean.to_numpy(), u.rad),
            az=u.Quantity(pointing_az_mean.to_numpy(), u.rad),
            obs_time=timestamps_mean,
        )

        # Add the additional parameters:
        df_radec_time = pd.DataFrame(
            data={
                "reco_ra": reco_ra_mean.to_value(u.deg),
                "reco_dec": reco_dec_mean.to_value(u.deg),
                "pointing_ra": pointing_ra_mean.to_value(u.deg),
                "pointing_dec": pointing_dec_mean.to_value(u.deg),
                "timestamp": timestamps_mean.value,
            },
            index=group_mean.index,
        )

        event_data_mean = event_data_mean.join(df_radec_time)

    return event_data_mean


def get_off_regions(
    pointing_ra, pointing_dec, on_coord_ra, on_coord_dec, n_off_regions
):
    """
    Gets OFF region(s) where to estimate the backgrounds of wobble
    observation data.

    It calculates the wobble offset and rotation angle with formulas
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
        Coordinate(s) of the center of the OFF region(s)
    """

    ra_diff = pointing_ra - on_coord_ra

    # Calculate the wobble offset:
    wobble_offset = np.arccos(
        np.cos(on_coord_dec) * np.cos(pointing_dec) * np.cos(ra_diff)
        + np.sin(on_coord_dec) * np.sin(pointing_dec)
    )

    logger.info(f"Wobble offset: {wobble_offset.to(u.deg):.3f}")

    # Calculate the wobble rotation angle:
    numerator_1 = np.sin(pointing_dec) * np.cos(on_coord_dec)
    numerator_2 = np.cos(pointing_dec) * np.sin(on_coord_dec) * np.cos(ra_diff)
    denominator = np.cos(pointing_dec) * np.sin(ra_diff)

    wobble_rotation = np.arctan2(numerator_1 - numerator_2, denominator)
    wobble_rotation = Angle(wobble_rotation).wrap_at(360 * u.deg)

    logger.info(f"Wobble rotation angle: {wobble_rotation.to(u.deg):.3f}")

    # Compute the OFF coordinates:
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


def get_stereo_events(event_data, quality_cuts=None):
    """
    Get stereo events from input data surviving specified quality cuts.

    The input data is supposed to have the index (obs_id, event_id) to
    group up the shower events.

    It adds the multiplicity of the telescopes and the telescope
    combination types, defined as follows, to the output data frame:

    MAGIC-I + MAGIC-II:  combo_type = 0
    LST-1 + MAGIC-I:  combo_type = 1
    LST-1 + MAGIC-II:  combo_type = 2
    LST-1 + MAGIC-I + MAGIC-II:  combo_type = 3

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events
    quality_cuts: str
        Quality cuts applied before extracting stereo events

    Returns
    -------
    event_data_stereo: pandas.core.frame.DataFrame
        Pandas data frame of the stereo events surviving the cuts
    """

    event_data_stereo = event_data.copy()

    # Apply the quality cuts:
    if quality_cuts is not None:
        logger.info(f"\nApplying the quality cuts:\n{quality_cuts}")
        event_data_stereo.query(quality_cuts, inplace=True)

    # Extract stereo events:
    event_data_stereo["multiplicity"] = event_data_stereo.groupby(GROUP_INDEX).size()
    event_data_stereo.query("multiplicity == [2, 3]", inplace=True)

    n_events_total = len(event_data_stereo.groupby(GROUP_INDEX).size())
    logger.info(f"\nIn total {n_events_total} stereo events are found:")

    # Check the telescope combination types:
    for combo_type, (tel_combo, tel_ids) in enumerate(TEL_COMBINATIONS.items()):

        df_events = event_data_stereo.query(
            f"(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})"
        ).copy()

        df_events["multiplicity"] = df_events.groupby(GROUP_INDEX).size()
        df_events.query(f"multiplicity == {len(tel_ids)}", inplace=True)

        n_events = int(len(df_events.groupby(GROUP_INDEX).size()))
        percentage = np.round(100 * n_events / n_events_total, 1)

        logger.info(
            f"\t{tel_combo} (type {combo_type}): {n_events} events ({percentage}%)"
        )

        event_data_stereo.loc[df_events.index, "combo_type"] = combo_type

    return event_data_stereo


def save_pandas_to_table(event_data, output_file, group_name, table_name, mode="w"):
    """
    Saves a pandas data frame in a table.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events
    output_file: str
        Path to an output HDF file
    group_name: str
        Group name of the output table
    table_name: str
        Name of the output table
    mode: str
        Mode of saving the data if a file already exists at the path
        of the output file, 'w' for overwriting the file with the new
        table, and 'a' for appending the table to the file
    """

    values = event_data.to_numpy()
    params = event_data.dtypes.index
    dtypes = event_data.dtypes.values

    event_table = np.array(
        [tuple(array) for array in values],
        dtype=np.dtype([(param, dtype) for param, dtype in zip(params, dtypes)]),
    )

    with tables.open_file(output_file, mode=mode) as f_out:
        f_out.create_table(group_name, table_name, createparents=True, obj=event_table)
