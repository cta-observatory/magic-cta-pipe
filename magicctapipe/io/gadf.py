#!/usr/bin/env python
# coding: utf-8

import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from magicctapipe import __version__
from pyirf.binning import split_bin_lo_hi

__all__ = [
    "create_gh_cuts_hdu",
    "create_event_hdu",
    "create_gti_hdu",
    "create_pointing_hdu",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The geographical coordinate of ORM
LON_ORM = u.Quantity(-17.89064, u.deg)
LAT_ORM = u.Quantity(28.76177, u.deg)
HEIGHT_ORM = u.Quantity(2199.835, u.m)

# The MJD reference time
MJDREF = Time(0, format="unix", scale="utc")


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


@u.quantity_input(source_ra=u.deg, source_dec=u.deg)
def create_event_hdu(event_table, deadc, source_name, source_ra=None, source_dec=None):
    """
    Creates an event list and its header.

    Parameters
    ----------
    event_table: astropy.table.table.QTable
        Astropy table of the DL2 events surviving gammaness cuts
    deadc: float:
        Dead time correction factor
    source_name: str
        Name of the observed source
    source_ra: astropy.units.quantity.Quantity
        Right ascension of the observed source
        (Used only when the source name cannot be resolved)
    source_dec: astropy.units.quantity.Quantity
        Declination of the observed source
        (Used only when the source name cannot be resolved)

    Returns
    -------
    event_list: astropy.table.table.QTable
        Astropy table of the DL2 events for DL3 data
    event_header: astropy.io.fits.header.Header
        Astropy header for the event list
    """

    mjdreff, mjdrefi = np.modf(MJDREF.mjd)

    time_start = Time(event_table["timestamp"][0], format="unix", scale="utc")
    time_end = Time(event_table["timestamp"][-1], format="unix", scale="utc")
    time_diffs = np.diff(event_table["timestamp"])

    elapsed_time = time_diffs.sum()
    effective_time = elapsed_time * deadc

    event_coords = SkyCoord(
        ra=event_table["reco_ra"], dec=event_table["reco_dec"], frame="icrs"
    )

    try:
        source_coord = SkyCoord.from_name(source_name, frame="icrs")

    except Exception:
        logger.warning(
            f"WARNING: The source name '{source_name}' could not be resolved. "
            "Setting the RA/Dec coordinate defined in the configuration file..."
        )
        source_coord = SkyCoord(ra=source_ra, dec=source_dec, frame="icrs")

    # create an event list
    event_list = QTable(
        data={
            "EVENT_ID": event_table["event_id"],
            "TIME": event_table["timestamp"],
            "RA": event_table["reco_ra"],
            "DEC": event_table["reco_dec"],
            "ENERGY": event_table["reco_energy"],
            "GAMMANESS": event_table["gammaness"],
            "MULTIP": event_table["multiplicity"],
            "GLON": event_coords.galactic.l.to(u.deg),
            "GLAT": event_coords.galactic.b.to(u.deg),
            "ALT": event_table["reco_alt"],
            "AZ": event_table["reco_az"],
        }
    )

    # Create an event header
    event_header = fits.Header()
    event_header["CREATED"] = Time.now().utc.iso
    event_header["HDUCLAS1"] = "EVENTS"
    event_header["OBS_ID"] = np.unique(event_table["obs_id"])[0]
    event_header["DATE-OBS"] = time_start.to_value("iso", "date")
    event_header["TIME-OBS"] = time_start.to_value("iso", "date_hms")[11:]
    event_header["DATE-END"] = time_end.to_value("iso", "date")
    event_header["TIME-END"] = time_end.to_value("iso", "date_hms")[11:]
    event_header["TSTART"] = time_start.value
    event_header["TSTOP"] = time_end.value
    event_header["MJDREFI"] = mjdrefi
    event_header["MJDREFF"] = mjdreff
    event_header["TIMEUNIT"] = "s"
    event_header["TIMESYS"] = "UTC"
    event_header["TIMEREF"] = "TOPOCENTER"
    event_header["ONTIME"] = elapsed_time.value
    event_header["TELAPSE"] = elapsed_time.value
    event_header["DEADC"] = deadc
    event_header["LIVETIME"] = effective_time.value
    event_header["OBJECT"] = source_name
    event_header["OBS_MODE"] = "WOBBLE"
    event_header["N_TELS"] = 3
    event_header["TELLIST"] = "LST-1_MAGIC"
    event_header["INSTRUME"] = "LST-1_MAGIC"
    event_header["RA_PNT"] = event_table["pointing_ra"][0].value
    event_header["DEC_PNT"] = event_table["pointing_dec"][0].value
    event_header["ALT_PNT"] = event_table["pointing_alt"][0].to_value(u.deg)
    event_header["AZ_PNT"] = event_table["pointing_az"][0].to_value(u.deg)
    event_header["RA_OBJ"] = source_coord.ra.to_value(u.deg)
    event_header["DEC_OBJ"] = source_coord.dec.to_value(u.deg)
    event_header["FOVALIGN"] = "RADEC"

    # Create a HDU
    event_hdu = fits.BinTableHDU(event_list, header=event_header, name="EVENTS")

    return event_hdu


def create_gti_hdu(event_table):
    """
    Creates a GTI table and its header.

    Parameters
    ----------
    event_table: astropy.table.table.QTable
        Astropy table of the DL2 events surviving gammaness cuts

    Returns
    -------
    gti_table: astropy.table.table.QTable
        Astropy table of the GTI information
    gti_header: astropy.io.fits.header.Header
        Astropy header for the GTI table
    """

    mjdreff, mjdrefi = np.modf(MJDREF.mjd)

    gti_table = QTable(
        data={
            "START": u.Quantity(event_table["timestamp"][0], ndmin=1),
            "STOP": u.Quantity(event_table["timestamp"][-1], ndmin=1),
        }
    )

    gti_header = fits.Header()
    gti_header["CREATED"] = Time.now().utc.iso
    gti_header["HDUCLAS1"] = "GTI"
    gti_header["OBS_ID"] = np.unique(event_table["obs_id"])[0]
    gti_header["MJDREFI"] = mjdrefi
    gti_header["MJDREFF"] = mjdreff
    gti_header["TIMEUNIT"] = "s"
    gti_header["TIMESYS"] = "UTC"
    gti_header["TIMEREF"] = "TOPOCENTER"

    # Create a HDU
    gti_hdu = fits.BinTableHDU(gti_table, header=gti_header, name="GTI")

    return gti_hdu


def create_pointing_hdu(event_table):
    """
    Creates a pointing table and its header.

    Parameters
    ----------
    event_table: astropy.table.table.QTable
        Astropy table of the DL2 events surviving gammaness cuts

    Returns
    -------
    pnt_table: astropy.table.table.QTable
        Astropy table of the pointing information
    pnt_header: astropy.io.fits.header.Header
        Astropy header for the pointing table
    """

    mjdreff, mjdrefi = np.modf(MJDREF.mjd)

    pnt_table = QTable(
        data={
            "TIME": u.Quantity(event_table["timestamp"][0], ndmin=1),
            "RA_PNT": u.Quantity(event_table["pointing_ra"][0], ndmin=1),
            "DEC_PNT": u.Quantity(event_table["pointing_dec"][0], ndmin=1),
            "ALT_PNT": u.Quantity(event_table["pointing_alt"][0].to(u.deg), ndmin=1),
            "AZ_PNT": u.Quantity(event_table["pointing_az"][0].to(u.deg), ndmin=1),
        }
    )

    pnt_header = fits.Header()
    pnt_header["CREATED"] = Time.now().utc.iso
    pnt_header["HDUCLAS1"] = "POINTING"
    pnt_header["OBS_ID"] = np.unique(event_table["obs_id"])[0]
    pnt_header["MJDREFI"] = mjdrefi
    pnt_header["MJDREFF"] = mjdreff
    pnt_header["TIMEUNIT"] = "s"
    pnt_header["TIMESYS"] = "UTC"
    pnt_header["TIMEREF"] = "TOPOCENTER"
    pnt_header["OBSGEO-L"] = (LON_ORM.to_value(u.deg), "Geographic longitude (deg)")
    pnt_header["OBSGEO-B"] = (LAT_ORM.to_value(u.deg), "Geographic latitude (deg)")
    pnt_header["OBSGEO-H"] = (HEIGHT_ORM.to_value(u.m), "Geographic height (m)")

    # Create a HDU
    pnt_hdu = fits.BinTableHDU(pnt_table, header=pnt_header, name="GTI")

    return pnt_hdu
