"""
Miscellanea functions
"""
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
    "calculate_off_coordinates",
    "transform_altaz_to_radec",
    "LON_ORM",
    "LAT_ORM",
    "HEIGHT_ORM",
]

# The geographic coordinate of ORM
LON_ORM = -17.89064 * u.deg
LAT_ORM = 28.76177 * u.deg
HEIGHT_ORM = 2199.835 * u.m


@u.quantity_input(
    pointing_alt=u.rad,
    pointing_az=u.rad,
    shower_alt=u.deg,
    shower_az=u.deg,
    cog_x=u.m,
    cog_y=u.m,
)
def calculate_disp(
    pointing_alt,
    pointing_az,
    shower_alt,
    shower_az,
    cog_x,
    cog_y,
    camera_frame,
):
    """
    Calculates the DISP parameter, i.e., the angular distance between
    the event arrival direction and the Center of Gravity (CoG) of the
    shower image.

    Parameters
    ----------
    pointing_alt : astropy.units.quantity.Quantity
        Altitude of the telescope pointing direction
    pointing_az : astropy.units.quantity.Quantity
        Azimuth of the telescope pointing direction
    shower_alt : astropy.units.quantity.Quantity
        Altitude of the event arrival direction
    shower_az : astropy.units.quantity.Quantity
        Azimuth of the event arrival direction
    cog_x : astropy.units.quantity.Quantity
        Image CoG along the X coordinate of the camera geometry
    cog_y : astropy.units.quantity.Quantity
        Image CoG along the Y coordinate of the camera geometry
    camera_frame : ctapipe.coordinates.camera_frame.CameraFrame
        Camera frame of the telescope

    Returns
    -------
    astropy.units.quantity.Quantity
        DISP parameter
    """

    tel_pointing = AltAz(alt=pointing_alt, az=pointing_az)
    tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

    # Transform the image CoG position to the Alt/Az direction
    cog_coord = SkyCoord(cog_x, cog_y, frame=camera_frame)
    cog_coord = cog_coord.transform_to(tel_frame).altaz

    # Calculate the DISP parameter
    disp = angular_separation(
        lon1=cog_coord.az, lat1=cog_coord.alt, lon2=shower_az, lat2=shower_alt
    )

    return disp


@u.quantity_input(
    shower_alt=u.deg,
    shower_az=u.deg,
    core_x=u.m,
    core_y=u.m,
    tel_pos_x=u.m,
    tel_pos_y=u.m,
    tel_pos_z=u.m,
)
def calculate_impact(
    shower_alt,
    shower_az,
    core_x,
    core_y,
    tel_pos_x,
    tel_pos_y,
    tel_pos_z,
):
    """
    Calculates the impact parameter, i.e., the closest distance between
    a shower axis and a telescope position.

    It uses the equations derived from a hand calculation, but the
    result is consistent with what calculated by MARS.

    Since ctapipe v0.16.0 a function to calculate the impact parameter
    is implemented, so we may replace it to the official one in future.

    Parameters
    ----------
    shower_alt : astropy.units.quantity.Quantity
        Altitude of the event arrival direction
    shower_az : astropy.units.quantity.Quantity
        Azimuth of the event arrival direction
    core_x : astropy.units.quantity.Quantity
        Core position along the geographic north
    core_y : astropy.units.quantity.Quantity
        Core position along the geographic west
    tel_pos_x : astropy.units.quantity.Quantity
        Telescope position along the geographic north
    tel_pos_y : astropy.units.quantity.Quantity
        Telescope position along the geographic west
    tel_pos_z : astropy.units.quantity.Quantity
        Telescope height from the reference altitude

    Returns
    -------
    astropy.units.quantity.Quantity
        Impact parameter
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


def calculate_mean_direction(lon, lat, unit, weights=None):
    """
    Calculates the mean direction per shower event.

    Please note that the input is supposed to be the pandas Series with
    the multi index (`obs_id`, `event_id`) to group telescope events.

    Parameters
    ----------
    lon : pandas.core.series.Series
        Longitude in a spherical coordinate
    lat : pandas.core.series.Series
        Latitude in a spherical coordinate
    unit : str
        Unit of the input (and output) angles -
        "deg", "degree", "rad" or "radian" are allowed
    weights : pandas.core.series.Series
        Weights for the input directions

    Returns
    -------
    lon_mean : pandas.core.series.Series
        Longitude of the mean direction
    lat_mean : pandas.core.series.Series
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
        x_coord_mean = x_coords.groupby(["obs_id", "event_id"]).mean()
        y_coord_mean = y_coords.groupby(["obs_id", "event_id"]).mean()
        z_coord_mean = z_coords.groupby(["obs_id", "event_id"]).mean()

    else:
        df_cartesian = pd.DataFrame(
            data={
                "weight": weights,
                "weighted_x_coord": x_coords * weights,
                "weighted_y_coord": y_coords * weights,
                "weighted_z_coord": z_coords * weights,
            }
        )

        group_sum = df_cartesian.groupby(["obs_id", "event_id"]).sum()

        x_coord_mean = group_sum["weighted_x_coord"] / group_sum["weight"]
        y_coord_mean = group_sum["weighted_y_coord"] / group_sum["weight"]
        z_coord_mean = group_sum["weighted_z_coord"] / group_sum["weight"]

    coord_mean = SkyCoord(
        x=x_coord_mean, y=y_coord_mean, z=z_coord_mean, representation_type="cartesian"
    )

    # Transform to the spherical coordinate
    coord_mean = coord_mean.spherical

    lon_mean = pd.Series(data=coord_mean.lon.to_value(unit), index=x_coord_mean.index)
    lat_mean = pd.Series(data=coord_mean.lat.to_value(unit), index=x_coord_mean.index)

    return lon_mean, lat_mean


@u.quantity_input(
    pointing_ra=u.deg, pointing_dec=u.deg, on_coord_ra=u.deg, on_coord_dec=u.deg
)
def calculate_off_coordinates(
    pointing_ra,
    pointing_dec,
    on_coord_ra,
    on_coord_dec,
    n_regions,
):
    """
    Calculates the coordinates of the centers of OFF regions to estimate
    the backgrounds for wobble observation data.

    It calculates the wobble rotation angle with equations derived from
    a hand calculation.

    Parameters
    ----------
    pointing_ra : astropy.units.quantity.Quantity
        Right ascension of the telescope pointing direction
    pointing_dec : astropy.units.quantity.Quantity
        Declination of the telescope pointing direction
    on_coord_ra : astropy.units.quantity.Quantity
        Right ascension of the center of the ON region
    on_coord_dec : astropy.units.quantity.Quantity
        Declination of the center of the ON region
    n_regions : int
        Number of OFF regions to be created

    Returns
    -------
    dict
        Coordinates of the centers of the OFF regions
    """

    wobble_coord = SkyCoord(pointing_ra, pointing_dec, frame="icrs")

    # Calculate the wobble offset
    wobble_offset = angular_separation(
        lon1=pointing_ra, lat1=pointing_dec, lon2=on_coord_ra, lat2=on_coord_dec
    )

    # Calculate the wobble rotation angle
    ra_diff = pointing_ra - on_coord_ra

    numerator = np.sin(pointing_dec) * np.cos(on_coord_dec)
    numerator -= np.cos(pointing_dec) * np.sin(on_coord_dec) * np.cos(ra_diff)
    denominator = np.cos(pointing_dec) * np.sin(ra_diff)

    wobble_rotation = np.arctan2(numerator, denominator)
    wobble_rotation = Angle(wobble_rotation).wrap_at("360 deg")

    # Calculate the rotation angles for the OFF regions. Here we remove
    # the angle 180 deg with which the OFF region will be created at the
    # same coordinate as the ON region.

    rotation_step = 360 / (n_regions + 1)
    rotations_off = np.arange(0, 359, rotation_step) * u.deg

    rotations_off = rotations_off[rotations_off.to_value("deg") != 180]
    rotations_off += wobble_rotation

    off_coords = {}

    # Loop over every rotation angle
    for i_off, rotation in enumerate(rotations_off, start=1):
        skyoffset_frame = SkyOffsetFrame(origin=wobble_coord, rotation=-rotation)

        # Calculate the OFF coordinate
        off_coord = SkyCoord(wobble_offset, "0 deg", frame=skyoffset_frame)
        off_coord = off_coord.transform_to("icrs")

        off_coords[i_off] = off_coord

    return off_coords


@u.quantity_input(alt=u.deg, az=u.deg)
def transform_altaz_to_radec(alt, az, obs_time):
    """
    Transforms the Alt/Az direction measured from ORM to the RA/Dec
    coordinate.

    Parameters
    ----------
    alt : astropy.units.quantity.Quantity
        Altitude measured from ORM
    az : astropy.units.quantity.Quantity
        Azimuth measured from ORM
    obs_time : astropy.time.core.Time
        Time when the direction was measured

    Returns
    -------
    ra : astropy.coordinates.Longitude
        Right ascension of the input direction
    dec : astropy.coordinates.Latitude
        Declination of the input direction
    """

    location = EarthLocation.from_geodetic(lon=LON_ORM, lat=LAT_ORM, height=HEIGHT_ORM)
    horizon_frames = AltAz(location=location, obstime=obs_time)

    # Transform to the RA/Dec coordinate
    event_coord = SkyCoord(alt=alt, az=az, frame=horizon_frames)
    event_coord = event_coord.transform_to("icrs")

    ra = event_coord.ra
    dec = event_coord.dec

    return ra, dec
