"""
DL3 generation utilities
"""
import logging

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import circmean
from astropy.table import QTable
from astropy.time import Time
from ctapipe_io_lst import REFERENCE_LOCATION
from pyirf.binning import split_bin_lo_hi

from .. import __version__ as MCP_VERSION
from ..utils.functions import HEIGHT_ORM, LAT_ORM, LON_ORM

__all__ = [
    "average_run_pointing",
    "create_gh_cuts_hdu",
    "create_event_hdu",
    "create_gti_hdu",
    "create_pointing_hdu",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The MJD reference time
MJDREF = Time(0, format="unix", scale="utc")


@u.quantity_input(reco_energy_bins=u.TeV, fov_offset_bins=u.deg)
def create_gh_cuts_hdu(
    gh_cuts, reco_energy_bins, fov_offset_bins, point_like, **header_cards
):
    """
    Creates a fits binary table HDU for dynamic gammaness cuts.

    Parameters
    ----------
    gh_cuts : np.ndarray
        Array of the gammaness cuts, which must have the shape
        (n_reco_energy_bins, n_fov_offset_bins)
    reco_energy_bins : astropy.units.quantity.Quantity
        Bin edges in the reconstructed energy
    fov_offset_bins : astropy.units.quantity.Quantity
        Bin edges in the field of view offset
    point_like : bool
        Meaning: true = IRFs are point-like, false = IRFS are full-enclosure
    **header_cards : dict
        Additional metadata to add to the header

    Returns
    -------
    astropy.io.fits.hdu.table.BinTableHDU
        Gammaness-cut HDU
    """

    energy_lo, energy_hi = split_bin_lo_hi(reco_energy_bins[np.newaxis, :].to("TeV"))
    theta_lo, theta_hi = split_bin_lo_hi(fov_offset_bins[np.newaxis, :].to("deg"))

    # Create a table
    qtable = QTable(
        data={
            "ENERG_LO": energy_lo,
            "ENERG_HI": energy_hi,
            "THETA_LO": theta_lo,
            "THETA_HI": theta_hi,
            "GH_CUTS": gh_cuts.T[np.newaxis, :],
        }
    )

    # Create a header
    header = fits.Header(
        cards=[
            ("CREATOR", f"magicctapipe v{MCP_VERSION}"),
            ("HDUCLAS1", "RESPONSE"),
            ("HDUCLAS2", "GH_CUTS"),
            ("HDUCLAS3", "POINT-LIKE" if point_like else "FULL-ENCLOSURE"),
            ("HDUCLAS4", "GH_CUTS_2D"),
            ("DATE", Time.now().utc.iso),
        ]
    )

    for key, value in header_cards.items():
        header[key] = value

    # Create a HDU
    gh_cuts_hdu = fits.BinTableHDU(qtable, header=header, name="GH_CUTS")

    return gh_cuts_hdu


def create_event_hdu(
    event_table, on_time, deadc, source_name, source_ra=None, source_dec=None
):
    """
    Creates a fits binary table HDU for shower events.

    Parameters
    ----------
    event_table : astropy.table.table.QTable
        Table of the DL2 events surviving gammaness cuts
    on_time : astropy.table.table.QTable
        ON time of the input data
    deadc : float
        Dead time correction factor
    source_name : str
        Name of the observed source
    source_ra : str, optional
        Right ascension of the observed source, whose format should be
        acceptable by `astropy.coordinates.sky_coordinate.SkyCoord`
        (Used only when the source name cannot be resolved)
    source_dec : str, optional
        Declination of the observed source, whose format should be
        acceptable by `astropy.coordinates.sky_coordinate.SkyCoord`
        (Used only when the source name cannot be resolved)

    Returns
    -------
    astropy.io.fits.hdu.table.BinTableHDU
        Event HDU

    Raises
    ------
    ValueError
        If the source name cannot be resolved and also either or both of
        source RA/Dec coordinate is set to None
    """
    TEL_COMBINATIONS = {
        "LST1_M1": [1, 2],  # combo_type = 0
        "LST1_M1_M2": [1, 2, 3],  # combo_type = 1
        "LST1_M2": [1, 3],  # combo_type = 2
        "M1_M2": [2, 3],  # combo_type = 3
    }  # TODO: REMOVE WHEN SWITCHING TO THE NEW RFs IMPLEMENTTATION (1 RF PER TELESCOPE)
    mjdreff, mjdrefi = np.modf(MJDREF.mjd)

    time_start = Time(event_table["timestamp"][0], format="unix", scale="utc")
    time_start_iso = time_start.to_value("iso", "date_hms")

    time_end = Time(event_table["timestamp"][-1], format="unix", scale="utc")
    time_end_iso = time_end.to_value("iso", "date_hms")

    # Calculate the elapsed and effective time
    elapsed_time = time_end - time_start
    effective_time = on_time * deadc

    # Get the instruments used for the observation
    combo_types_unique = np.unique(event_table["combo_type"])
    tel_combos = np.array(list(TEL_COMBINATIONS.keys()))[combo_types_unique]

    tel_list = [tel_combo.split("_") for tel_combo in tel_combos]
    tel_list_unique = np.unique(sum(tel_list, []))

    instruments = "_".join(tel_list_unique)

    # Transfer the RA/Dec directions to the galactic coordinate
    event_coords = SkyCoord(
        ra=event_table["reco_ra"], dec=event_table["reco_dec"], frame="icrs"
    )

    event_coords = event_coords.galactic

    try:
        # Try to get the source coordinate from the input name
        source_coord = SkyCoord.from_name(source_name, frame="icrs")

    except Exception:
        logger.warning(
            f"WARNING: The source name '{source_name}' could not be resolved. "
            f"Setting the input RA/Dec coordinate ({source_ra}, {source_dec})..."
        )

        if (source_ra is None) or (source_dec is None):
            raise ValueError("The input RA/Dec coordinate is set to `None`.")

        source_coord = SkyCoord(ra=source_ra, dec=source_dec, frame="icrs")

    # Create a table
    qtable = QTable(
        data={
            "EVENT_ID": event_table["event_id"],
            "TIME": event_table["timestamp"],
            "RA": event_table["reco_ra"],
            "DEC": event_table["reco_dec"],
            "ENERGY": event_table["reco_energy"],
            "GAMMANESS": event_table["gammaness"],
            "MULTIP": event_table["multiplicity"],
            "GLON": event_coords.l.to("deg"),
            "GLAT": event_coords.b.to("deg"),
            "ALT": event_table["reco_alt"].to("deg"),
            "AZ": event_table["reco_az"].to("deg"),
        }
    )

    mean_ra, mean_dec, mean_alt, mean_az = average_run_pointing(event_table)

    # Create a header
    header = fits.Header(
        cards=[
            ("CREATED", Time.now().utc.iso),
            ("HDUCLAS1", "EVENTS"),
            ("OBS_ID", np.unique(event_table["obs_id"])[0]),
            ("DATE-OBS", time_start_iso[:10]),
            ("TIME-OBS", time_start_iso[11:]),
            ("DATE-END", time_end_iso[:10]),
            ("TIME-END", time_end_iso[11:]),
            ("TSTART", time_start.value),
            ("TSTOP", time_end.value),
            ("MJDREFI", mjdrefi),
            ("MJDREFF", mjdreff),
            ("TIMEUNIT", "s"),
            ("TIMESYS", "UTC"),
            ("TIMEREF", "TOPOCENTER"),
            ("ONTIME", on_time.value),
            ("TELAPSE", elapsed_time.to_value("s")),
            ("DEADC", deadc),
            ("LIVETIME", effective_time.value),
            ("OBJECT", source_name),
            ("OBS_MODE", "WOBBLE"),
            ("N_TELS", np.max(event_table["multiplicity"])),
            ("TELLIST", instruments),
            ("INSTRUME", instruments),
            ("GEOLON", REFERENCE_LOCATION.lon.to_value(u.deg), "deg"),
            ("GEOLAT", REFERENCE_LOCATION.lat.to_value(u.deg), "deg"),
            ("ALTITUDE", round(REFERENCE_LOCATION.height.to_value(u.m), 2), "m"),
            ("RA_PNT", mean_ra.to_value("deg"), "deg"),
            ("DEC_PNT", mean_dec.to_value("deg"), "deg"),
            ("ALT_PNT", mean_alt.to_value("deg"), "deg"),
            ("AZ_PNT", mean_az.to_value("deg"), "deg"),
            ("RA_OBJ", source_coord.ra.to_value("deg"), "deg"),
            ("DEC_OBJ", source_coord.dec.to_value("deg"), "deg"),
            ("FOVALIGN", "RADEC"),
        ]
    )

    # Create a HDU
    event_hdu = fits.BinTableHDU(qtable, header=header, name="EVENTS")

    return event_hdu


def create_gti_hdu(event_table):
    """
    Creates a fits binary table HDU for Good Time Interval (GTI).

    Parameters
    ----------
    event_table : astropy.table.table.QTable
        Table of the DL2 events surviving gammaness cuts

    Returns
    -------
    astropy.io.fits.hdu.table.BinTableHDU
        GTI HDU
    """

    mjdreff, mjdrefi = np.modf(MJDREF.mjd)

    # Create a table
    qtable = QTable(
        data={
            "START": u.Quantity(event_table["timestamp"][0], ndmin=1),
            "STOP": u.Quantity(event_table["timestamp"][-1], ndmin=1),
        }
    )

    # Create a header
    header = fits.Header(
        cards=[
            ("CREATED", Time.now().utc.iso),
            ("HDUCLAS1", "GTI"),
            ("OBS_ID", np.unique(event_table["obs_id"])[0]),
            ("MJDREFI", mjdrefi),
            ("MJDREFF", mjdreff),
            ("TIMEUNIT", "s"),
            ("TIMESYS", "UTC"),
            ("TIMEREF", "TOPOCENTER"),
        ]
    )

    # Create a HDU
    gti_hdu = fits.BinTableHDU(qtable, header=header, name="GTI")

    return gti_hdu


def average_run_pointing(
    event_table, exclude_fraction_first=0.15, exclude_fraction_last=0.05
):
    """
    Calculates the average pointing (in RA/Dec and Alt/Az coordinates) excluding the beginning and
    the end of the run

    Parameters
    ----------
    event_table : astropy.table.table.QTable
        Table of the DL2 events surviving gammaness cuts
    exclude_fraction_first : float
        Fraction of events from the beginning of the run excluded for calculation of average position
    exclude_fraction_last : float
        Fraction of events from the beginning of the run excluded for calculation of average position

    Returns
    -------
    astropy.units.quantity.Quantity
        Average RA
    astropy.units.quantity.Quantity
        Average Dec
    astropy.units.quantity.Quantity
        Average Alt
    astropy.units.quantity.Quantity
        Average Az
    """
    first = int(len(event_table) * exclude_fraction_first)
    last = int(len(event_table) * (1 - exclude_fraction_last))
    mean_ra = circmean(event_table["pointing_ra"][first:last])
    mean_dec = event_table["pointing_dec"][first:last].mean()
    mean_alt = event_table["pointing_alt"][first:last].mean()
    mean_az = circmean(event_table["pointing_az"][first:last])
    return mean_ra, mean_dec, mean_alt, mean_az


def create_pointing_hdu(event_table):
    """
    Creates a fits binary table HDU for the pointing direction.

    Parameters
    ----------
    event_table : astropy.table.table.QTable
        Table of the DL2 events surviving gammaness cuts

    Returns
    -------
    astropy.io.fits.hdu.table.BinTableHDU
        Pointing HDU
    """
    mjdreff, mjdrefi = np.modf(MJDREF.mjd)
    mean_ra, mean_dec, mean_alt, mean_az = average_run_pointing(event_table)

    # Create a table
    qtable = QTable(
        data={
            "TIME": u.Quantity(event_table["timestamp"][0], ndmin=1),
            "RA_PNT": u.Quantity(mean_ra, ndmin=1),
            "DEC_PNT": u.Quantity(mean_dec, ndmin=1),
            "ALT_PNT": u.Quantity(mean_alt.to("deg"), ndmin=1),
            "AZ_PNT": u.Quantity(mean_az.to("deg"), ndmin=1),
        }
    )

    # Create a header
    header = fits.Header(
        cards=[
            ("CREATED", Time.now().utc.iso),
            ("HDUCLAS1", "POINTING"),
            ("OBS_ID", np.unique(event_table["obs_id"])[0]),
            ("MJDREFI", mjdrefi),
            ("MJDREFF", mjdreff),
            ("TIMEUNIT", "s"),
            ("TIMESYS", "UTC"),
            ("TIMEREF", "TOPOCENTER"),
            ("OBSGEO-L", LON_ORM.to_value("deg"), "Geographic longitude (deg)"),
            ("OBSGEO-B", LAT_ORM.to_value("deg"), "Geographic latitude (deg)"),
            ("OBSGEO-H", HEIGHT_ORM.to_value("m"), "Geographic height (m)"),
        ]
    )

    # Create a HDU
    pointing_hdu = fits.BinTableHDU(qtable, header=header, name="POINTING")

    return pointing_hdu
