#!/usr/bin/env python
# coding: utf-8

import logging
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.time import Time
from magicctapipe import __version__
from magicctapipe.io.io import TEL_COMBINATIONS
from magicctapipe.utils.functions import HEIGHT_ORM, LAT_ORM, LON_ORM
from pyirf.binning import split_bin_lo_hi

__all__ = [
    "create_gh_cuts_hdu",
    "create_event_hdu",
    "create_gti_hdu",
    "create_pointing_hdu",
    "create_hdu_index_hdu",
    "create_obs_index_hdu",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["CREATOR"] = f"magicctapipe v{__version__}"
DEFAULT_HEADER[
    "HDUDOC"
] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
DEFAULT_HEADER["HDUVERS"] = "0.3"
DEFAULT_HEADER["HDUCLASS"] = "GADF"
DEFAULT_HEADER["ORIGIN"] = "CTA"
DEFAULT_HEADER["TELESCOP"] = "CTA-N"
# The MJD reference time
MJDREF = Time(0, format="unix", scale="utc")


@u.quantity_input
def create_gh_cuts_hdu(
    gh_cuts, reco_energy_bins: u.TeV, fov_offset_bins: u.deg, **header_cards
):
    """
    Creates a fits binary table HDU for dynamic gammaness cuts.

    Parameters
    ----------
    gh_cuts: numpy.ndarray
        Array of the gammaness cuts, which must have the shape
        (n_reco_energy_bins, n_fov_offset_bins)
    reco_energy_bins: astropy.units.quantity.Quantity
        Bin edges in the reconstructed energy
    fov_offset_bins: astropy.units.quantity.Quantity
        Bin edges in the field of view offset
    **header_cards
        Additional metadata to add to the header

    Returns
    -------
    gh_cuts_hdu: astropy.io.fits.hdu.table.BinTableHDU
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
            ("CREATOR", f"magicctapipe v{__version__}"),
            ("HDUCLAS1", "RESPONSE"),
            ("HDUCLAS2", "GH_CUTS"),
            ("HDUCLAS3", "POINT-LIKE"),
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
    event_table: astropy.table.table.QTable
        Table of the DL2 events surviving gammaness cuts
    on_time: astropy.table.table.QTable
        ON time of the input data
    deadc: float
        Dead time correction factor
    source_name: str
        Name of the observed source
    source_ra: str
        Right ascension of the observed source, whose format should be
        acceptable by `astropy.coordinates.sky_coordinate.SkyCoord`
        (Used only when the source name cannot be resolved)
    source_dec: str
        Declination of the observed source, whose format should be
        acceptable by `astropy.coordinates.sky_coordinate.SkyCoord`
        (Used only when the source name cannot be resolved)

    Returns
    -------
    event_hdu: astropy.io.fits.hdu.table.BinTableHDU
        Event HDU

    Raises
    ------
    ValueError
        If the source name cannot be resolved and also either or both of
        source RA/Dec coordinate is set to None
    """

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
            ("RA_PNT", event_table["pointing_ra"][0].value, "deg"),
            ("DEC_PNT", event_table["pointing_dec"][0].value, "deg"),
            ("ALT_PNT", event_table["pointing_alt"][0].to_value("deg"), "deg"),
            ("AZ_PNT", event_table["pointing_az"][0].to_value("deg"), "deg"),
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
    event_table: astropy.table.table.QTable
        Table of the DL2 events surviving gammaness cuts

    Returns
    -------
    gti_hdu: astropy.io.fits.hdu.table.BinTableHDU
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


def create_pointing_hdu(event_table):
    """
    Creates a fits binary table HDU for the pointing direction.

    Parameters
    ----------
    event_table: astropy.table.table.QTable
        Table of the DL2 events surviving gammaness cuts

    Returns
    -------
    pointing_hdu: astropy.io.fits.hdu.table.BinTableHDU
        Pointing HDU
    """

    mjdreff, mjdrefi = np.modf(MJDREF.mjd)

    # Create a table
    qtable = QTable(
        data={
            "TIME": u.Quantity(event_table["timestamp"][0], ndmin=1),
            "RA_PNT": u.Quantity(event_table["pointing_ra"][0], ndmin=1),
            "DEC_PNT": u.Quantity(event_table["pointing_dec"][0], ndmin=1),
            "ALT_PNT": u.Quantity(event_table["pointing_alt"][0].to("deg"), ndmin=1),
            "AZ_PNT": u.Quantity(event_table["pointing_az"][0].to("deg"), ndmin=1),
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


def create_hdu_index_hdu(filename_list, fits_dir, hdu_index_file, overwrite=False):
    """
    Create the hdu index table and write it to the given file.
    The Index table is created as per,
    http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html
    Parameters
    ----------
    filename_list : list
        list of filenames of the fits files
    fits_dir : Path
        Path of the fits files
    hdu_index_file : Path
        Path for HDU index file
    overwrite : Bool
        Boolean to overwrite existing file
    """

    hdu_index_tables = []

    base_dir = os.path.commonpath(
        [hdu_index_file.parent.absolute().resolve(), fits_dir.absolute().resolve()]
    )
    # loop through the files
    for file in filename_list:
        filepath = fits_dir / file
        if filepath.is_file():
            try:
                hdu_list = fits.open(filepath)
                evt_hdr = hdu_list["EVENTS"].header

                # just test they are here
                hdu_list["GTI"].header
                hdu_list["POINTING"].header
            except Exception:
                logger.error(f"fits corrupted for file {file}")
                continue
        else:
            logger.error(f"fits {file} doesn't exist")
            continue

        # The column names for the table follows the scheme as shown in
        # https://gamma-astro-data-formats.readthedocs.io/en/latest/general/hduclass.html
        # Event list
        t_events = {
            "OBS_ID": evt_hdr["OBS_ID"],
            "HDU_TYPE": "events",
            "HDU_CLASS": "events",
            "FILE_DIR": str(os.path.relpath(fits_dir, hdu_index_file.parent)),
            "FILE_NAME": str(file),
            "HDU_NAME": "EVENTS",
            "SIZE": filepath.stat().st_size,
        }
        hdu_index_tables.append(t_events)

        # GTI
        t_gti = t_events.copy()

        t_gti["HDU_TYPE"] = "gti"
        t_gti["HDU_CLASS"] = "gti"
        t_gti["HDU_NAME"] = "GTI"

        hdu_index_tables.append(t_gti)

        # POINTING
        t_pnt = t_events.copy()

        t_pnt["HDU_TYPE"] = "pointing"
        t_pnt["HDU_CLASS"] = "pointing"
        t_pnt["HDU_NAME"] = "POINTING"

        hdu_index_tables.append(t_pnt)
        hdu_names = [
            "EFFECTIVE AREA",
            "ENERGY DISPERSION",
            "BACKGROUND",
            "PSF",
            "RAD_MAX",
        ]

        for irf in hdu_names:
            try:
                t_irf = t_events.copy()
                irf_hdu = hdu_list[irf].header["HDUCLAS4"]

                t_irf["HDU_CLASS"] = irf_hdu.lower()
                t_irf["HDU_TYPE"] = irf_hdu.lower().strip(
                    "_" + irf_hdu.lower().split("_")[-1]
                )
                t_irf["HDU_NAME"] = irf
                hdu_index_tables.append(t_irf)
            except KeyError:
                logger.error(f"Run {t_events['OBS_ID']} does not contain HDU {irf}")

    hdu_index_table = Table(hdu_index_tables)

    hdu_index_header = DEFAULT_HEADER.copy()
    hdu_index_header["CREATED"] = Time.now().utc.iso
    hdu_index_header["HDUCLAS1"] = "INDEX"
    hdu_index_header["HDUCLAS2"] = "HDU"
    hdu_index_header["INSTRUME"] = evt_hdr["INSTRUME"]
    hdu_index_header["BASE_DIR"] = base_dir

    hdu_index = fits.BinTableHDU(
        hdu_index_table, header=hdu_index_header, name="HDU INDEX"
    )
    hdu_index_list = fits.HDUList([fits.PrimaryHDU(), hdu_index])
    hdu_index_list.writeto(hdu_index_file, overwrite=overwrite)


def create_obs_index_hdu(filename_list, fits_dir, obs_index_file, overwrite):
    """
    Create the obs index table and write it to the given file.
    The Index table is created as per,
    http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html
    Parameters
    ----------
    filename_list : list
        list of filenames of the fits files
    fits_dir : Path
        Path of the fits files
    obs_index_file : Path
        Path for the OBS index file
    overwrite : Bool
        Boolean to overwrite existing file
    """
    obs_index_tables = []

    # loop through the files
    for file in filename_list:
        filepath = fits_dir / file
        if filepath.is_file():
            try:
                hdu_list = fits.open(filepath)
                evt_hdr = hdu_list["EVENTS"].header
            except Exception:
                logger.error(f"fits corrupted for file {file}")
                continue
        else:
            logger.error(f"fits {file} doesn't exist")
            continue

        # Obs_table
        t_obs = {
            "OBS_ID": evt_hdr["OBS_ID"],
            "DATE-OBS": evt_hdr["DATE-OBS"],
            "TIME-OBS": evt_hdr["TIME-OBS"],
            "DATE-END": evt_hdr["DATE-END"],
            "TIME-END": evt_hdr["TIME-END"],
            "RA_PNT": evt_hdr["RA_PNT"] * u.deg,
            "DEC_PNT": evt_hdr["DEC_PNT"] * u.deg,
            "ZEN_PNT": (90 - float(evt_hdr["ALT_PNT"])) * u.deg,
            "ALT_PNT": evt_hdr["ALT_PNT"] * u.deg,
            "AZ_PNT": evt_hdr["AZ_PNT"] * u.deg,
            "RA_OBJ": evt_hdr["RA_OBJ"] * u.deg,
            "DEC_OBJ": evt_hdr["DEC_OBJ"] * u.deg,
            "TSTART": evt_hdr["TSTART"] * u.s,
            "TSTOP": evt_hdr["TSTOP"] * u.s,
            "ONTIME": evt_hdr["ONTIME"] * u.s,
            "TELAPSE": evt_hdr["TELAPSE"] * u.s,
            "LIVETIME": evt_hdr["LIVETIME"] * u.s,
            "DEADC": evt_hdr["DEADC"],
            "OBJECT": evt_hdr["OBJECT"],
            "OBS_MODE": evt_hdr["OBS_MODE"],
            "N_TELS": evt_hdr["N_TELS"],
            "TELLIST": evt_hdr["TELLIST"],
            "INSTRUME": evt_hdr["INSTRUME"],
        }
        obs_index_tables.append(t_obs)

    obs_index_table = QTable(obs_index_tables)

    obs_index_header = DEFAULT_HEADER.copy()
    obs_index_header["CREATED"] = Time.now().utc.iso
    obs_index_header["HDUCLAS1"] = "INDEX"
    obs_index_header["HDUCLAS2"] = "OBS"
    obs_index_header["INSTRUME"] = t_obs["INSTRUME"]
    obs_index_header["MJDREFI"] = evt_hdr["MJDREFI"]
    obs_index_header["MJDREFF"] = evt_hdr["MJDREFF"]

    obs_index = fits.BinTableHDU(
        obs_index_table, header=obs_index_header, name="OBS INDEX"
    )
    obs_index_list = fits.HDUList([fits.PrimaryHDU(), obs_index])
    obs_index_list.writeto(obs_index_file, overwrite=overwrite)
