#!/usr/bin/env python
# coding: utf-8

"""
This script creates a DL3 data file with input DL2 data and IRF files.
The settings used for creating the IRFs are automatically applied to the DL2 events.

Usage:
$ python lst1_magic_dl2_to_dl3.py
--input-file-dl2 ./data/dl2_LST-1_MAGIC.Run03265.h5
--input-dir-irf ./data/irf
--output-dir ./data
--config-file ./config.yaml
"""

import argparse
import glob
import logging
import operator
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from magicctapipe.utils import check_tel_combination, create_gh_cuts_hdu, get_dl2_mean
from pyirf.binning import join_bin_lo_hi
from pyirf.cuts import evaluate_binned_cut
from pyirf.interpolation import interpolate_effective_area_per_energy_and_fov
from pyirf.io import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
)
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

ORM_LAT = 28.76177  # unit: [deg]
ORM_LON = -17.89064  # unit: [deg]
ORM_HEIGHT = 2199.835  # unit: [m]

dead_time_lst = 7.6e-6  # unit: [sec]
dead_time_magic = 26e-6  # unit: [sec]

MJDREF = Time(0, format="unix", scale="utc").mjd

__all__ = [
    "calculate_deadc",
    "load_irf_files",
    "load_dl2_data_file",
    "create_event_list",
    "create_gti_table",
    "create_pointing_table",
    "dl2_to_dl3",
]


def calculate_deadc(time_diffs, dead_time):
    """
    Calculates the dead time correction factor.

    Parameters
    ----------
    time_diffs: np.ndarray
        Time differences of event arrival times
    dead_time: float
        Dead time due to the read out

    Returns
    -------
    deadc: float
        Dead time correction factor
    """

    rate = 1 / (time_diffs.mean() - dead_time)
    deadc = 1 / (1 + rate * dead_time)

    return deadc


def load_irf_files(input_dir_irf):
    """
    Loads input IRF files.

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
        raise RuntimeError(f"No IRF files are found under {input_dir_irf}.")

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

    logger.info("\nInput IRF files:")

    for input_file in input_files_irf:

        logger.info(input_file)
        hdus_irf = fits.open(input_file)

        header = hdus_irf["EFFECTIVE AREA"].header

        for key in extra_header.keys():
            if key in header:
                extra_header[key].append(header[key])

        # Read the grid point:
        coszd = np.cos(np.deg2rad(header["PNT_ZD"]))
        azimuth = np.deg2rad(header["PNT_AZ"])
        grid_point = [coszd, azimuth]

        # Read the IRF data:
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

    # Check the IRF data consistency:
    for key in irf_data.keys():

        irf_data[key] = np.array(irf_data[key])
        n_data = len(irf_data[key])

        if (n_data != 0) and (n_data != n_input_files):
            raise RuntimeError(
                f"The number of '{key}' data (= {n_data}) does not match "
                f"with that of the input IRF files (= {n_input_files})."
            )

        if "bins" in key:
            unique_bins = np.unique(irf_data[key], axis=0)
            n_unique_bins = len(unique_bins)

            if n_unique_bins == 1:
                irf_data[key] = unique_bins[0]

            elif n_unique_bins > 1:
                raise RuntimeError(
                    f"The '{key}' of the input IRF files does not match."
                )

    # Check the header consistency:
    for key in list(extra_header.keys()):

        n_data = len(extra_header[key])
        unique_values = np.unique(extra_header[key])

        if n_data == 0:
            extra_header.pop(key)

        elif (n_data != n_input_files) or len(unique_values) > 1:
            raise RuntimeError(
                "The configrations of the input IRF files do not match, "
                "at least the setting '{key}'."
            )
        else:
            extra_header[key] = unique_values[0]

    # Set the units to the IRF data:
    irf_data["effective_area"] *= u.m**2
    irf_data["background"] *= u.Unit("MeV-1 s-1 sr-1")
    irf_data["rad_max"] *= u.deg
    irf_data["energy_bins"] *= u.TeV
    irf_data["fov_offset_bins"] *= u.deg
    irf_data["bkg_fov_offset_bins"] *= u.deg

    return irf_data, extra_header


def load_dl2_data_file(
    input_file, quality_cuts=None, irf_type="software", dl2_weight=None
):
    """
    Loads an input DL2 data file.

    Parameters
    ----------
    input_file: str
        Path to an input DL2 data file
    quality_cuts: str
        Quality cuts applied to the input events
    irf_type: str
        Type of the LST-1 + MAGIC IRFs
    dl2_weight: str
        Type of the weight for averaging tel-wise DL2 parameters

    Returns
    -------
    event_table: astropy.table.table.QTable
        Astropy table of DL2 events
    deadc: float
        Dead time correction factor for the input data
    """

    logger.info(f"Input DL2 data file:\n{input_file}")

    df_events = pd.read_hdf(input_file, key="events/parameters")
    df_events.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    df_events.sort_index(inplace=True)

    # Apply the quality cuts:
    if quality_cuts is not None:
        logger.info("\nApplying the quality cuts...")

        df_events.query(quality_cuts, inplace=True)
        df_events["multiplicity"] = df_events.groupby(["obs_id", "event_id"]).size()
        df_events.query("multiplicity > 1", inplace=True)

    combo_types = check_tel_combination(df_events)
    df_events.update(combo_types)

    # Select the events of the specified IRF type:
    logger.info(f'\nExtracting the events of the "{irf_type}" type...')

    if irf_type == "software":
        df_events.query("combo_type == 3", inplace=True)

    elif irf_type == "software_with_any2":
        df_events.query("combo_type > 0", inplace=True)

    elif irf_type == "magic_stereo":
        df_events.query("combo_type == 0", inplace=True)

    elif irf_type == "hardware":
        logger.info(
            "\nThe hardware trigger has not yet been used for observations. Exiting."
        )
        sys.exit()

    n_events = len(df_events.groupby(["obs_id", "event_id"]).size())
    logger.info(f"--> {n_events} stereo events")

    # Calculate the dead time correction factor.
    # For MAGIC we select one telescope which has more number of events than the other:
    logger.info("\nCalculating the dead time correction factor...")

    deadc = 1
    condition = "(time_diff > 0) & (time_diff < 0.1)"

    df_lst = df_events.query(f"(tel_id == 1) & {condition}")
    time_diffs_lst = df_lst["time_diff"].to_numpy()

    if len(time_diffs_lst) > 0:
        deadc_lst = calculate_deadc(time_diffs_lst, dead_time_lst)
        logger.info(f"LST-1: {deadc_lst}")
        deadc *= deadc_lst

    df_m1 = df_events.query(f"(tel_id == 2) & {condition}")
    df_m2 = df_events.query(f"(tel_id == 3) & {condition}")

    time_diffs_m1 = df_m1["time_diff"].to_numpy()
    time_diffs_m2 = df_m2["time_diff"].to_numpy()

    if len(time_diffs_m1) >= len(time_diffs_m2):
        deadc_magic = calculate_deadc(time_diffs_m1, dead_time_magic)
        logger.info(f"MAGIC-I: {deadc_magic}")
    else:
        deadc_magic = calculate_deadc(time_diffs_m2, dead_time_magic)
        logger.info(f"MAGIC-II: {deadc_magic}")

    deadc *= deadc_magic

    dead_time_fraction = 100 * (1 - deadc)
    logger.info(f"--> Total dead time fraction: {dead_time_fraction:.2f}%")

    # Compute the mean of the DL2 parameters:
    df_dl2_mean = get_dl2_mean(df_events, dl2_weight)
    df_dl2_mean.reset_index(inplace=True)

    # Convert the pandas data frame to the astropy QTable:
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

    return event_table, deadc


def create_event_list(
    event_table, deadc, source_name=None, source_ra=None, source_dec=None
):
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
    source_ra:
        Right ascension of the observed source
    source_dec:
        Declination of the observed source

    Returns
    -------
    event_list: astropy.table.table.QTable
        Astropy table of the DL2 events for DL3 data
    event_header: astropy.io.fits.header.Header
        Astropy header for the event list
    """

    time_start = Time(event_table["timestamp"][0], format="unix", scale="utc")
    time_end = Time(event_table["timestamp"][-1], format="unix", scale="utc")
    time_diffs = np.diff(event_table["timestamp"])

    elapsed_time = np.sum(time_diffs)
    effective_time = elapsed_time * deadc

    event_coords = SkyCoord(
        ra=event_table["reco_ra"], dec=event_table["reco_dec"], frame="icrs"
    )

    if source_name is not None:
        source_coord = SkyCoord.from_name(source_name)
        source_coord = source_coord.transform_to("icrs")
    else:
        source_coord = SkyCoord(ra=source_ra, dec=source_dec, frame="icrs")

    # Create an event list:
    event_list = QTable(
        {
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

    # Create an event header:
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
    event_header["MJDREFI"] = np.modf(MJDREF)[1]
    event_header["MJDREFF"] = np.modf(MJDREF)[0]
    event_header["TIMEUNIT"] = "s"
    event_header["TIMESYS"] = "UTC"
    event_header["TIMEREF"] = "TOPOCENTER"
    event_header["ONTIME"] = elapsed_time
    event_header["TELAPSE"] = elapsed_time
    event_header["DEADC"] = deadc
    event_header["LIVETIME"] = effective_time
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

    gti_table = QTable(
        {
            "START": u.Quantity(event_table["timestamp"][0], unit=u.s, ndmin=1),
            "STOP": u.Quantity(event_table["timestamp"][-1], unit=u.s, ndmin=1),
        }
    )

    gti_header = fits.Header()
    gti_header["CREATED"] = Time.now().utc.iso
    gti_header["HDUCLAS1"] = "GTI"
    gti_header["OBS_ID"] = np.unique(event_table["obs_id"])[0]
    gti_header["MJDREFI"] = np.modf(MJDREF)[1]
    gti_header["MJDREFF"] = np.modf(MJDREF)[0]
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

    pnt_table = QTable(
        {
            "TIME": u.Quantity(event_table["timestamp"][0], unit=u.s, ndmin=1),
            "RA_PNT": u.Quantity(event_table["pointing_ra"][0].value, ndmin=1),
            "DEC_PNT": u.Quantity(event_table["pointing_dec"][0].value, ndmin=1),
            "ALT_PNT": u.Quantity(
                event_table["pointing_alt"][0].to_value(u.deg), ndmin=1
            ),
            "AZ_PNT": u.Quantity(
                event_table["pointing_az"][0].to_value(u.deg), ndmin=1
            ),
        }
    )

    pnt_header = fits.Header()
    pnt_header["CREATED"] = Time.now().utc.iso
    pnt_header["HDUCLAS1"] = "POINTING"
    pnt_header["OBS_ID"] = np.unique(event_table["obs_id"])[0]
    pnt_header["MJDREFI"] = np.modf(MJDREF)[1]
    pnt_header["MJDREFF"] = np.modf(MJDREF)[0]
    pnt_header["TIMEUNIT"] = "s"
    pnt_header["TIMESYS"] = "UTC"
    pnt_header["TIMEREF"] = "TOPOCENTER"
    pnt_header["OBSGEO-L"] = (ORM_LON, "Geographic longitude of the telescopes (deg)")
    pnt_header["OBSGEO-B"] = (ORM_LAT, "Geographic latitude of the telescopes (deg)")
    pnt_header["OBSGEO-H"] = (ORM_HEIGHT, "Geographic height of the telescopes (m)")

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

    # Load the input IRF files:
    irf_data, extra_header = load_irf_files(input_dir_irf)

    quality_cuts = extra_header.get("QUAL_CUT")
    irf_type = extra_header.get("IRF_TYPE")
    dl2_weight = extra_header.get("DL2_WEIG")

    # Load the input DL2 data file:
    event_table, deadc = load_dl2_data_file(
        input_file_dl2, quality_cuts, irf_type, dl2_weight
    )

    pointing_coszd = np.mean(np.sin(event_table["pointing_alt"]))
    # FIX ME: how to compute the mean if the azimuth makes a full 2pi turn:
    pointing_az = np.mean(event_table["pointing_az"])
    target_point = [pointing_coszd, pointing_az]

    # Interpolate the IRFs:
    config_dl3 = config["dl2_to_dl3"]

    interpolation_method = config_dl3.pop("irf_interpolation_method")
    extra_header["IRF_INTP"] = interpolation_method

    hdus = fits.HDUList([fits.PrimaryHDU()])

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

    # Interpolate the energy dispersion with a custom way, since there
    # is a bug in the function of pyirf v0.6.0 about the renormalization
    logger.info("Interpolating the energy dispersion...")

    edisp_interp = griddata(
        points=irf_data["grid_point"],
        values=irf_data["energy_dispersion"],
        xi=target_point,
        method=interpolation_method,
    )

    norm = np.sum(edisp_interp, axis=2, keepdims=True)  # Along the migration axis
    mask_zeros = norm != 0

    edisp_interp = np.divide(
        edisp_interp, norm, out=np.zeros_like(edisp_interp), where=mask_zeros
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

    if len(irf_data["background"]) > 1:
        logger.info(
            "Warning: more than one background models are found, but the "
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

    # Apply gammaness cuts:
    if "GH_CUT" in extra_header:
        logger.info("\nApplying the global gammaness cut:")

        global_gam_cut = extra_header["GH_CUT"]
        logger.info(f"\tGlobal cut value: {global_gam_cut}")

        event_table = event_table[event_table["gammaness"] > global_gam_cut]

    else:
        logger.info("\nApplying the dynamic gammaness cuts...")

        gh_cuts = hdus["GH_CUTS"].data

        cut_table_gh = QTable()
        cut_table_gh["low"] = gh_cuts["ENERG_LO"] * u.TeV
        cut_table_gh["high"] = gh_cuts["ENERG_HI"] * u.TeV
        cut_table_gh["cut"] = gh_cuts["GH_CUT"][0, 0]

        logger.info(f"\nGammaness cut table:\n{cut_table_gh}")

        mask_gh_gamma = evaluate_binned_cut(
            values=event_table["gammaness"],
            bin_values=event_table["reco_energy"],
            cut_table=cut_table_gh,
            op=operator.ge,
        )

        event_table = event_table[mask_gh_gamma]

    # Create an event list HDU:
    logger.info("\nCreating an event list HDU...")

    event_list, event_header = create_event_list(event_table, deadc, **config_dl3)

    hdu_event = fits.BinTableHDU(event_list, header=event_header, name="EVENTS")
    hdus.append(hdu_event)

    # Create a GTI table:
    logger.info("Creating a GTI HDU...")

    gti_table, gti_header = create_gti_table(event_table)

    hdu_gti = fits.BinTableHDU(gti_table, header=gti_header, name="GTI")
    hdus.append(hdu_gti)

    # Create a pointing table:
    logger.info("Creating a pointing HDU...")

    pnt_table, pnt_header = create_pointing_table(event_table)

    hdu_pnt = fits.BinTableHDU(pnt_table, header=pnt_header, name="POINTING")
    hdus.append(hdu_pnt)

    # Save the data in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    regex = r"dl2_(\S+)\.h5"
    file_name = Path(input_file_dl2).name

    if re.fullmatch(regex, file_name):
        parser = re.findall(regex, file_name)[0]
        output_file = f"{output_dir}/dl3_{parser}.fits.gz"
    else:
        raise RuntimeError("Could not parse information from the input file name.")

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
        help="Path to an input IRF directory (interpolation will be applied).",
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

    # Process the input data:
    dl2_to_dl3(args.input_file_dl2, args.input_dir_irf, args.output_dir, config)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
