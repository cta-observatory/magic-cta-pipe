#!/usr/bin/env python
# coding: utf-8

"""
This script processes DL2 events and creates a DL3 data file with input
IRF file(s). It reads the configurations of the IRFs, and if they are
consistent, it applies the same condition cuts to the input DL2 events.

For the interpolation of the IRFs and the dynamic gammaness/theta cuts,
there are three methods, "nearest", "linear" or "cubic", which can be
specified in the configuration file. The "nearest" method just selects
the IRFs of the closest pointing direction in (cos(Zd), Az), which works
even if the input is only one file. The other methods work only when
there are multiple IRFs available from different pointing directions.

Usage:
$ python lst1_magic_dl2_to_dl3.py
--input-file-dl2 ./dl2_LST-1_MAGIC.Run03265.h5
--input-dir-irf ./irf
--output-dir ./dl3
--config-file ./config.yaml
"""

import argparse
import glob
import logging
import operator
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from magicctapipe.utils import create_gh_cuts_hdu, get_dl2_mean, get_stereo_events
from pyirf.binning import join_bin_lo_hi
from pyirf.cuts import evaluate_binned_cut
from pyirf.interpolation import (
    interpolate_effective_area_per_energy_and_fov,
    interpolate_energy_dispersion,
)
from pyirf.io import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
)
from scipy.interpolate import griddata

__all__ = [
    "calculate_deadc",
    "load_irf_files",
    "load_dl2_data_file",
    "create_event_list",
    "create_gti_table",
    "create_pointing_table",
    "dl2_to_dl3",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The geographical coordinate of ORM
ORM_LAT = 28.76177 * u.deg
ORM_LON = -17.89064 * u.deg
ORM_HEIGHT = 2199.835 * u.m

# The LST/MAGIC readout dead times
DEAD_TIME_LST = 7.6 * u.us
DEAD_TIME_MAGIC = 26 * u.us

# The upper limit of event time differences used when calculating
# the dead time correction factor
TIME_DIFF_UPLIM = 0.1 * u.s

# The MJD reference time
MJDREF = Time(0, format="unix", scale="utc")


def calculate_deadc(event_data):
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


def load_irf_files(input_dir_irf):
    """
    Loads input IRF files and checks the consistency.

    Parameters
    ----------
    input_dir_irf: str
        Path to a directory where input IRF files are stored

    Returns
    -------
    irf_data: dict
        Combined IRF data
    extra_header: dict
        Extra header of input IRF files
    """

    irf_file_mask = f"{input_dir_irf}/irf_*.fits.gz"

    input_files_irf = glob.glob(irf_file_mask)
    input_files_irf.sort()

    n_input_files = len(input_files_irf)

    if n_input_files == 0:
        raise FileNotFoundError("Could not find IRF files in the input directory.")

    extra_header = {
        "TELESCOP": [],
        "INSTRUME": [],
        "FOVALIGN": [],
        "QUAL_CUT": [],
        "IRF_TYPE": [],
        "DL2_WEIG": [],
        "GH_CUT": [],
        "GH_EFF": [],
        "GH_MIN": [],
        "GH_MAX": [],
        "RAD_MAX": [],
        "TH_EFF": [],
        "TH_MIN": [],
        "TH_MAX": [],
    }

    irf_data = {
        "grid_point": [],
        "effective_area": [],
        "energy_dispersion": [],
        "background": [],
        "gh_cuts": [],
        "rad_max": [],
        "energy_bins": [],
        "fov_offset_bins": [],
        "migration_bins": [],
        "bkg_fov_offset_bins": [],
    }

    logger.info("\nThe following files are found:")

    for input_file in input_files_irf:

        logger.info(input_file)
        hdus_irf = fits.open(input_file)

        header = hdus_irf["EFFECTIVE AREA"].header

        for key in extra_header.keys():
            if key in header:
                extra_header[key].append(header[key])

        # Read the grid point
        coszd = np.cos(np.deg2rad(header["PNT_ZD"]))
        azimuth = np.deg2rad(header["PNT_AZ"])
        grid_point = [coszd, azimuth]

        # Read the IRF data
        aeff_data = hdus_irf["EFFECTIVE AREA"].data[0]
        edisp_data = hdus_irf["ENERGY DISPERSION"].data[0]

        energy_bins = join_bin_lo_hi(aeff_data["ENERG_LO"], aeff_data["ENERG_HI"])
        fov_offset_bins = join_bin_lo_hi(aeff_data["THETA_LO"], aeff_data["THETA_HI"])
        migration_bins = join_bin_lo_hi(edisp_data["MIGRA_LO"], edisp_data["MIGRA_HI"])

        irf_data["grid_point"].append(grid_point)
        irf_data["effective_area"].append(aeff_data["EFFAREA"])
        irf_data["energy_dispersion"].append(np.swapaxes(edisp_data["MATRIX"], 0, 2))
        irf_data["energy_bins"].append(energy_bins)
        irf_data["fov_offset_bins"].append(fov_offset_bins)
        irf_data["migration_bins"].append(migration_bins)

        if "BACKGROUND" in hdus_irf:
            bkg_data = hdus_irf["BACKGROUND"].data[0]
            bkg_fov_offset_bins = join_bin_lo_hi(
                bkg_data["THETA_LO"], bkg_data["THETA_HI"]
            )

            irf_data["background"].append(bkg_data["BKG"])
            irf_data["bkg_fov_offset_bins"].append(bkg_fov_offset_bins)

        if "GH_CUTS" in hdus_irf:
            ghcuts_data = hdus_irf["GH_CUTS"].data[0]
            irf_data["gh_cuts"].append(ghcuts_data["GH_CUTS"])

        if "RAD_MAX" in hdus_irf:
            radmax_data = hdus_irf["RAD_MAX"].data[0]
            irf_data["rad_max"].append(radmax_data["RAD_MAX"])

    # Check the IRF data consistency
    for key in irf_data.keys():

        irf_data[key] = np.array(irf_data[key])
        n_data = len(irf_data[key])

        if (n_data != 0) and (n_data != n_input_files):
            raise ValueError(
                f"The number of '{key}' data (= {n_data}) does not match "
                f"with that of the input IRF files (= {n_input_files})."
            )

        if "bins" in key:
            unique_bins = np.unique(irf_data[key], axis=0)
            n_unique_bins = len(unique_bins)

            if n_unique_bins == 1:
                irf_data[key] = unique_bins[0]

            elif n_unique_bins > 1:
                raise ValueError(f"The '{key}' of the input IRF files does not match.")

    # Check the header consistency
    for key in list(extra_header.keys()):

        n_data = len(extra_header[key])
        unique_values = np.unique(extra_header[key])

        if n_data == 0:
            extra_header.pop(key)

        elif (n_data != n_input_files) or len(unique_values) > 1:
            raise ValueError(
                "The configurations of the input IRF files do not match, "
                "at least the setting '{key}'."
            )
        else:
            extra_header[key] = unique_values[0]

    # Set the units to the IRF data
    irf_data["effective_area"] *= u.m**2
    irf_data["background"] *= u.Unit("MeV-1 s-1 sr-1")
    irf_data["rad_max"] *= u.deg
    irf_data["energy_bins"] *= u.TeV
    irf_data["fov_offset_bins"] *= u.deg
    irf_data["bkg_fov_offset_bins"] *= u.deg

    return irf_data, extra_header


def load_dl2_data_file(input_file, quality_cuts, irf_type, dl2_weight):
    """
    Loads a DL2 data file.

    Parameters
    ----------
    input_file: str
        Path to an input DL2 data file
    quality_cuts: str
        Quality cuts applied to the input events
    irf_type: str
        Type of the IRFs which will be created -
        "software(_only_3tel)", "magic_only" or "hardware" are allowed
    dl2_weight: str
        Type of the weight for averaging telescope-wise DL2 parameters -
        "simple", "variance" or "intensity" are allowed


    Returns
    -------
    event_table: astropy.table.table.QTable
        Astropy table of DL2 events
    deadc: float
        Dead time correction factor
    """

    df_events = pd.read_hdf(input_file, key="events/parameters")
    df_events.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    df_events.sort_index(inplace=True)

    df_events = get_stereo_events(df_events, quality_cuts)

    # Select the events of the specified IRF type
    logger.info(f'\nExtracting the events of the "{irf_type}" type...')

    if irf_type == "software":
        df_events.query("combo_type > 0", inplace=True)

    elif irf_type == "software_only_3tel":
        df_events.query("combo_type == 3", inplace=True)

    elif irf_type == "magic_only":
        df_events.query("combo_type == 0", inplace=True)

    elif irf_type == "hardware":
        logger.warning(
            "WARNING: Please confirm that this IRF type is correct for the input data, "
            "since the hardware trigger between LST-1 and MAGIC may NOT be used."
        )

    n_events = len(df_events.groupby(["obs_id", "event_id"]).size())
    logger.info(f"--> {n_events} stereo events")

    # Calculate the dead time correction factor
    deadc = calculate_deadc(df_events)

    # Compute the mean of the DL2 parameters
    df_dl2_mean = get_dl2_mean(df_events, dl2_weight)
    df_dl2_mean.reset_index(inplace=True)

    # Convert the pandas data frame to the astropy QTable
    event_table = QTable.from_pandas(df_dl2_mean)

    event_table["pointing_alt"] *= u.rad
    event_table["pointing_az"] *= u.rad
    event_table["pointing_ra"] *= u.deg
    event_table["pointing_dec"] *= u.deg
    event_table["reco_alt"] *= u.deg
    event_table["reco_az"] *= u.deg
    event_table["reco_ra"] *= u.deg
    event_table["reco_dec"] *= u.deg
    event_table["reco_energy"] *= u.TeV
    event_table["timestamp"] *= u.s

    return event_table, deadc


@u.quantity_input(source_ra=u.deg, source_dec=u.deg)
def create_event_list(event_table, deadc, source_name, source_ra=None, source_dec=None):
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

    return event_list, event_header


def create_gti_table(event_table):
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

    return gti_table, gti_header


def create_pointing_table(event_table):
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
    pnt_header["OBSGEO-L"] = (ORM_LON.to_value(u.deg), "Geographic longitude (deg)")
    pnt_header["OBSGEO-B"] = (ORM_LAT.to_value(u.deg), "Geographic latitude (deg)")
    pnt_header["OBSGEO-H"] = (ORM_HEIGHT.to_value(u.m), "Geographic height (m)")

    return pnt_table, pnt_header


def dl2_to_dl3(input_file_dl2, input_dir_irf, output_dir, config):
    """
    Creates a DL3 data file with input DL2 data and IRF files.

    Parameters
    ----------
    input_file_dl2: str
        Path to an input DL2 data file
    input_dir_irf: str
        Path to a directory where input IRF files are stored
    output_dir: str
        Path to a directory where to save an output DL3 data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_dl3 = config["dl2_to_dl3"]

    if config_dl3["source_ra"] is not None:
        config_dl3["source_ra"] *= u.deg

    if config_dl3["source_dec"] is not None:
        config_dl3["source_dec"] *= u.deg

    # Load the input IRF files
    logger.info(f"\nInput IRF directory:{input_dir_irf}")

    irf_data, extra_header = load_irf_files(input_dir_irf)

    logger.info(f"\nGrid points:\n{irf_data['grid_point'].round(5).tolist()}")

    logger.info("\nExtra header:")
    for key, value in extra_header.items():
        logger.info(f"\t{key}: {value}")

    # Load the input DL2 data file
    logger.info(f"\n\nInput DL2 data file:\n{input_file_dl2}")

    quality_cuts = extra_header.get("QUAL_CUT")
    irf_type = extra_header["IRF_TYPE"]
    dl2_weight = extra_header.get("DL2_WEIG")

    event_table, deadc = load_dl2_data_file(
        input_file_dl2, quality_cuts, irf_type, dl2_weight
    )

    # Calculate the mean pointing direction for the target point of the
    # IRF interpolation. Please note that the azimuth could make a full
    # 2 pi turn, whose mean angle could indicate an opposite direction.
    # Thus, here we calculate the STDs of the azimuth angles with two
    # ranges, i.e., 0 <= az < 360 deg and -180 <= az < 180 deg, and then
    # calculate the mean with the range of smaller STD.

    pointing_coszd_mean = np.sin(event_table["pointing_alt"].to_value(u.rad)).mean()

    pointing_az_wrap_360deg = Angle(event_table["pointing_az"]).wrap_at(360 * u.deg)
    pointing_az_wrap_180deg = Angle(event_table["pointing_az"]).wrap_at(180 * u.deg)

    if pointing_az_wrap_360deg.std() <= pointing_az_wrap_180deg.std():
        pointing_az_mean = pointing_az_wrap_360deg.mean().value
    else:
        pointing_az_mean = pointing_az_wrap_180deg.mean().wrap_at(360 * u.deg).value

    target_point = np.array([pointing_coszd_mean, pointing_az_mean])
    logger.info(f"\nTarget point: {target_point.round(5).tolist()}")

    # Prepare for the IRF interpolations
    interpolation_method = config_dl3.pop("interpolation_method")
    logger.info(f"\n\nInterpolation method: {interpolation_method}")

    extra_header["IRF_INTP"] = interpolation_method

    hdus = fits.HDUList([fits.PrimaryHDU()])

    # Interpolate the effective area and create the HDU
    logger.info("\nInterpolating the effective area...")

    aeff_interp = interpolate_effective_area_per_energy_and_fov(
        effective_area=irf_data["effective_area"],
        grid_points=irf_data["grid_point"],
        target_point=target_point,
        method=interpolation_method,
    )

    hdu_aeff = create_aeff2d_hdu(
        effective_area=aeff_interp[:, 0],
        true_energy_bins=irf_data["energy_bins"],
        fov_offset_bins=irf_data["fov_offset_bins"],
        point_like=True,
        extname="EFFECTIVE AREA",
        **extra_header,
    )

    hdus.append(hdu_aeff)

    # Interpolate the energy dispersion and create the HDU
    logger.info("Interpolating the energy dispersion...")

    edisp_interp = interpolate_energy_dispersion(
        energy_dispersions=irf_data["energy_dispersion"],
        grid_points=irf_data["grid_point"],
        target_point=target_point,
        method=interpolation_method,
    )

    hdu_edisp = create_energy_dispersion_hdu(
        energy_dispersion=edisp_interp[0],
        true_energy_bins=irf_data["energy_bins"],
        migration_bins=irf_data["migration_bins"],
        fov_offset_bins=irf_data["fov_offset_bins"],
        point_like=True,
        extname="ENERGY DISPERSION",
    )

    hdus.append(hdu_edisp)

    # Check the existence of the background IRF
    if len(irf_data["background"]) > 1:
        logger.warning(
            "WARNING: More than one background models are found, but the "
            "interpolation method for them is not implemented. Skipping."
        )

    elif len(irf_data["background"]) == 1:
        hdu_bkg = create_background_2d_hdu(
            background_2d=irf_data["background"].T,
            reco_energy_bins=irf_data["energy_bins"],
            fov_offset_bins=irf_data["fov_offset_bins"],
            extname="BACKGROUND",
        )

        hdus.append(hdu_bkg)

    # Interpolate the gammaness cuts and create the HDU
    if len(irf_data["gh_cuts"]) > 0:
        logger.info("Interpolating the dynamic gammaness cuts...")

        gh_cuts_interp = griddata(
            points=irf_data["grid_point"],
            values=irf_data["gh_cuts"],
            xi=target_point,
            method=interpolation_method,
        )

        hdu_gh_cuts = create_gh_cuts_hdu(
            gh_cuts=gh_cuts_interp.T[:, 0],
            reco_energy_bins=irf_data["energy_bins"],
            fov_offset_bins=irf_data["fov_offset_bins"],
            extname="GH_CUTS",
            **extra_header,
        )

        hdus.append(hdu_gh_cuts)

    # Interpolate the theta cuts and create the HDU
    if len(irf_data["rad_max"]) > 0:
        logger.info("Interpolating the dynamic theta cuts...")

        rad_max_interp = griddata(
            points=irf_data["grid_point"],
            values=irf_data["rad_max"].to_value(u.deg),
            xi=target_point,
            method=interpolation_method,
        )

        hdu_rad_max = create_rad_max_hdu(
            rad_max=u.Quantity(rad_max_interp.T[:, 0], u.deg),
            reco_energy_bins=irf_data["energy_bins"],
            fov_offset_bins=irf_data["fov_offset_bins"],
            point_like=True,
            extname="RAD_MAX",
            **extra_header,
        )

        hdus.append(hdu_rad_max)

    # Apply the interpolated gammaness cuts
    if "GH_CUT" in extra_header:
        logger.info("\nApplying the global gammaness cut:")

        global_gam_cut = extra_header["GH_CUT"]
        logger.info(f"\tGlobal cut value: {global_gam_cut}")

        event_table = event_table[event_table["gammaness"] > global_gam_cut]

    else:
        logger.info("\nApplying the dynamic gammaness cuts...")

        gh_cuts = hdus["GH_CUTS"].data[0]

        cut_table_gh = QTable()
        cut_table_gh["low"] = gh_cuts["ENERG_LO"] * u.TeV
        cut_table_gh["high"] = gh_cuts["ENERG_HI"] * u.TeV
        cut_table_gh["cut"] = gh_cuts["GH_CUTS"][0]

        logger.info(f"\nGammaness cut table:\n{cut_table_gh}")

        mask_gh_gamma = evaluate_binned_cut(
            values=event_table["gammaness"],
            bin_values=event_table["reco_energy"],
            cut_table=cut_table_gh,
            op=operator.ge,
        )

        event_table = event_table[mask_gh_gamma]

    # Create an event list HDU
    logger.info("\n\nCreating an event list HDU...")

    event_list, event_header = create_event_list(event_table, deadc, **config_dl3)

    hdu_event = fits.BinTableHDU(event_list, header=event_header, name="EVENTS")
    hdus.append(hdu_event)

    # Create a GTI table
    logger.info("Creating a GTI HDU...")

    gti_table, gti_header = create_gti_table(event_table)

    hdu_gti = fits.BinTableHDU(gti_table, header=gti_header, name="GTI")
    hdus.append(hdu_gti)

    # Create a pointing table
    logger.info("Creating a pointing HDU...")

    pnt_table, pnt_header = create_pointing_table(event_table)

    hdu_pnt = fits.BinTableHDU(pnt_table, header=pnt_header, name="POINTING")
    hdus.append(hdu_pnt)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    input_file_name = Path(input_file_dl2).name

    output_file_name = input_file_name.replace("dl2", "dl3").replace(".h5", ".fits.gz")
    output_file = f"{output_dir}/{output_file_name}"

    hdus.writeto(output_file, overwrite=True)

    logger.info(f"\nOutput file:\n{output_file}")


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file-dl2",
        "-d",
        dest="input_file_dl2",
        type=str,
        required=True,
        help="Path to an input DL2 data file.",
    )

    parser.add_argument(
        "--input-dir-irf",
        "-i",
        dest="input_dir_irf",
        type=str,
        required=True,
        help="Path to a directory where input IRF file(s) are stored.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL3 data file.",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config.yaml",
        help="Path to a yaml configuration file.",
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Process the input data
    dl2_to_dl3(args.input_file_dl2, args.input_dir_irf, args.output_dir, config)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
