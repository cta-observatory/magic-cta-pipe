import logging
import os

import astropy.units as u
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.time import Time

from magicctapipe import __version__

__all__ = [
    "create_hdu_index_hdu",
    "create_obs_index_hdu",
]

logger = logging.getLogger(__name__)


DEFAULT_HEADER = fits.Header()
DEFAULT_HEADER["CREATOR"] = f"magicctapipe v{__version__}"
DEFAULT_HEADER[
    "HDUDOC"
] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
DEFAULT_HEADER["HDUVERS"] = "0.3"
DEFAULT_HEADER["HDUCLASS"] = "GADF"
DEFAULT_HEADER["ORIGIN"] = "CTA"
DEFAULT_HEADER["TELESCOP"] = "CTA-N"

# Observation_mode is POINTING for all MAGIC-LST observations as per GADF v0.3
OBS_MODE = "POINTING"


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
