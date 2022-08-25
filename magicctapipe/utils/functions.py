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
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from ctapipe.coordinates import TelescopeFrame
from magicctapipe import __version__
from pyirf.binning import split_bin_lo_hi

__all__ = [
    "calculate_disp",
    "calculate_impact",
    "calculate_mean_direction",
    "transform_altaz_to_radec",
    "get_dl2_mean",
    "get_off_regions",
    "get_stereo_events",
    "save_pandas_to_table",
    "create_gh_cuts_hdu",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The geographical coordinate of ORM
LON_ORM = u.Quantity(-17.89064, u.deg)
LAT_ORM = u.Quantity(28.76177, u.deg)
HEIGHT_ORM = u.Quantity(2199.835, u.m)

# The telescope combination types
TEL_COMBINATIONS = {
    "m1_m2": [2, 3],  # combo_type = 0
    "lst1_m1": [1, 2],  # combo_type = 1
    "lst1_m2": [1, 3],  # combo_type = 2
    "lst1_m1_m2": [1, 2, 3],  # combo_type = 3
}

# The pandas index to group up shower events
GROUP_INDEX = ["obs_id", "event_id"]


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


def get_dl2_mean(event_data, weight_type="simple"):
    """
    Gets the mean DL2 parameters per shower event.

    The input data is supposed to have the index (obs_id, event_id) to
    group up the shower events.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events
    weight_type: str
        Type of the weights for the telescope-wise DL2 parameters -
        "simple" does not use any weights for calculations,
        "variance" uses the inverse of the RF variance, and
        "intensity" uses the linear-scale intensity parameter

    Returns
    -------
    event_data_mean: pandas.core.frame.DataFrame
        Pandas data frame of shower events with the mean parameters
    """

    is_simulation = "true_energy" in event_data.columns

    # Create a mean data frame
    if is_simulation:
        params = ["combo_type", "multiplicity", "true_energy", "true_alt", "true_az"]
    else:
        params = ["combo_type", "multiplicity", "timestamp"]

    event_data_mean = event_data[params].groupby(GROUP_INDEX).mean()

    # Calculate the mean pointing direction
    pointing_az_mean, pointing_alt_mean = calculate_mean_direction(
        event_data["pointing_az"], event_data["pointing_alt"]
    )

    event_data_mean["pointing_alt"] = pointing_alt_mean
    event_data_mean["pointing_az"] = pointing_az_mean

    # Define the weights for the DL2 parameters
    if weight_type == "simple":
        energy_weights = 1
        direction_weights = None
        gammaness_weights = 1

    elif weight_type == "variance":
        energy_weights = 1 / event_data["reco_energy_var"]
        direction_weights = 1 / event_data["reco_disp_var"]
        gammaness_weights = 1 / event_data["gammaness_var"]

    elif weight_type == "intensity":
        energy_weights = event_data["intensity"]
        direction_weights = event_data["intensity"]
        gammaness_weights = event_data["intensity"]

    df_events = pd.DataFrame(
        data={
            "energy_weight": energy_weights,
            "gammaness_weight": gammaness_weights,
            "weighted_energy": np.log10(event_data["reco_energy"]) * energy_weights,
            "weighted_gammaness": event_data["gammaness"] * gammaness_weights,
        }
    )

    # Calculate the mean DL2 parameters
    group_sum = df_events.groupby(GROUP_INDEX).sum()

    reco_energy_mean = 10 ** (group_sum["weighted_energy"] / group_sum["energy_weight"])
    gammaness_mean = group_sum["weighted_gammaness"] / group_sum["gammaness_weight"]

    reco_az_mean, reco_alt_mean = calculate_mean_direction(
        event_data["reco_az"], event_data["reco_alt"], direction_weights, unit="deg",
    )

    event_data_mean["reco_energy"] = reco_energy_mean
    event_data_mean["reco_alt"] = reco_alt_mean
    event_data_mean["reco_az"] = reco_az_mean
    event_data_mean["gammaness"] = gammaness_mean

    # Transform the Alt/Az to the RA/Dec coordinate
    if not is_simulation:

        timestamps_mean = Time(
            event_data_mean["timestamp"].to_numpy(), format="unix", scale="utc"
        )

        pointing_ra_mean, pointing_dec_mean = transform_altaz_to_radec(
            alt=u.Quantity(pointing_alt_mean.to_numpy(), u.rad),
            az=u.Quantity(pointing_az_mean.to_numpy(), u.rad),
            obs_time=timestamps_mean,
        )

        reco_ra_mean, reco_dec_mean = transform_altaz_to_radec(
            alt=u.Quantity(reco_alt_mean.to_numpy(), u.deg),
            az=u.Quantity(reco_az_mean.to_numpy(), u.deg),
            obs_time=timestamps_mean,
        )

        event_data_mean["pointing_ra"] = pointing_ra_mean
        event_data_mean["pointing_dec"] = pointing_dec_mean
        event_data_mean["reco_ra"] = reco_ra_mean
        event_data_mean["reco_dec"] = reco_dec_mean

    return event_data_mean


@u.quantity_input
def get_off_regions(
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


def get_stereo_events(event_data, quality_cuts=None):
    """
    Gets stereo events surviving specified quality cuts.

    The input data is supposed to have the index (obs_id, event_id) to
    group up the shower events.

    It adds the telescope multiplicity and combination types to the
    output data frame.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events
    quality_cuts: str
        Quality cuts applied to the input data

    Returns
    -------
    event_data_stereo: pandas.core.frame.DataFrame
        Pandas data frame of the stereo events surviving the cuts
    """

    event_data_stereo = event_data.copy()

    # Apply the quality cuts
    if quality_cuts is not None:
        logger.info(f"\nApplying the quality cuts:\n{quality_cuts}")
        event_data_stereo.query(quality_cuts, inplace=True)

    # Extract stereo events
    event_data_stereo["multiplicity"] = event_data_stereo.groupby(GROUP_INDEX).size()
    event_data_stereo.query("multiplicity == [2, 3]", inplace=True)

    n_events_total = len(event_data_stereo.groupby(GROUP_INDEX).size())
    logger.info(f"\nIn total {n_events_total} stereo events are found:")

    # Check the telescope combination types
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


def save_pandas_to_table(data, output_file, group_name, table_name, mode="w"):
    """
    Saves a pandas data frame in a table.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Pandas data frame
    output_file: str
        Path to an output HDF file
    group_name: str
        Group name of the output table
    table_name: str
        Name of the output table
    mode: str
        Mode of saving the data if a file already exists at the output
        file path, "w" for overwriting the file with the new table, and
        "a" for appending the table to the file
    """

    params = data.dtypes.index
    dtypes = data.dtypes.values

    data_array = np.array(
        [tuple(array) for array in data.to_numpy()],
        dtype=np.dtype([(param, dtype) for param, dtype in zip(params, dtypes)]),
    )

    with tables.open_file(output_file, mode=mode) as f_out:
        f_out.create_table(group_name, table_name, createparents=True, obj=data_array)


@u.quantity_input
def create_gh_cuts_hdu(
    gh_cuts, reco_energy_bins: u.TeV, fov_offset_bins: u.deg, extname, **header_cards
):
    """
    Creates a fits binary table HDU for gammaness cuts.

    Parameters
    ----------
    gh_cuts: numpy.ndarray
        Array of the gammaness cuts, which must have the shape
        (n_reco_energy_bins, n_fov_offset_bins)
    reco_energy_bins: astropy.units.quantity.Quantity
        Bin edges in the reconstructed energy
    fov_offset_bins: astropy.units.quantity.Quantity
        Bin edges in the field of view offset
    extname: str
        Name for the output HDU
    **header_cards
        Additional metadata to add to the header

    Returns
    -------
    hdu_gh_cuts: astropy.io.fits.hdu.table.BinTableHDU
        Gammaness-cuts HDU
    """

    energy_lo, energy_hi = split_bin_lo_hi(reco_energy_bins[np.newaxis, :].to(u.TeV))
    theta_lo, theta_hi = split_bin_lo_hi(fov_offset_bins[np.newaxis, :].to(u.deg))

    # Create a table
    gh_cuts_table = QTable()
    gh_cuts_table["ENERG_LO"] = energy_lo
    gh_cuts_table["ENERG_HI"] = energy_hi
    gh_cuts_table["THETA_LO"] = theta_lo
    gh_cuts_table["THETA_HI"] = theta_hi
    gh_cuts_table["GH_CUTS"] = gh_cuts.T[np.newaxis, :]

    # Create a header
    header = fits.Header()
    header["CREATOR"] = f"magicctapipe v{__version__}"
    header["HDUCLAS1"] = "RESPONSE"
    header["HDUCLAS2"] = "GH_CUTS"
    header["HDUCLAS3"] = "POINT-LIKE"
    header["HDUCLAS4"] = "GH_CUTS_2D"
    header["DATE"] = Time.now().utc.iso

    for key, value in header_cards.items():
        header[key] = value

    # Create a HDU
    hdu_gh_cuts = fits.BinTableHDU(gh_cuts_table, header=header, name=extname)

    return hdu_gh_cuts
