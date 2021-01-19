import os
import glob
import time
import logging
import operator
import argparse

import numpy as np
from astropy import table
import astropy.units as u
from astropy.io import fits
from astropy.table import QTable

from pyirf.simulations import SimulatedEventsInfo

from magicctapipe.utils.filedir import *
from magicctapipe.utils.plot import *
from magicctapipe.utils.utils import *

import matplotlib.pylab as plt
from lstchain.mc import plot_utils


def read_simu_info_mcp_sum_num_showers(file_list, mc_header_key="dl2/mc_header"):
    """Function to read simulation information from DL2 files and sum the simultated
    showers. Assumes that all the mc_headers are equal

    Parameters
    ----------
    file_list : list
        magic-cta-pipe DL2 file list
    mc_header_key : str, optional
        mc_header key, by default "dl2/mc_header"

    Returns
    -------
    pd.DataFrame
        mc_header with sum num showers
    """
    d = read_mc_header(file_list[0], mc_header_key)
    num_showers = 0
    for i, file in enumerate(file_list):
        num_showers += int(read_mc_header(file, mc_header_key)["num_showers"])
    d["num_showers"] = num_showers
    return d


def convert_simu_info_mcp_to_pyirf(file_list, mc_header_key="dl2/mc_header"):
    """Function to convert simulation information from magic-cta-pipe DL2 files to 
    pyirf format

    Parameters
    ----------
    file_list : file list
        magic-cta-pipe DL2 file list
    mc_header_key : str, optional
        mc_header key, by default "dl2/mc_header"

    Returns
    -------
    SimulatedEventsInfo
        pyirf_simu_info
    """
    simu_info = read_simu_info_mcp_sum_num_showers(file_list, mc_header_key)
    pyirf_simu_info = SimulatedEventsInfo(
        n_showers=int(simu_info.num_showers) * int(simu_info.shower_reuse),
        energy_min=float(simu_info.energy_range_min) * u.TeV,
        energy_max=float(simu_info.energy_range_max) * u.TeV,
        max_impact=float(simu_info.max_scatter_range) * u.m,
        spectral_index=float(simu_info.spectral_index),
        viewcone=float(simu_info.max_viewcone_radius) * u.deg,
    )
    return pyirf_simu_info


def read_dl2_mcp_to_pyirf_MAGIC_LST_list(
    file_mask,
    reco_key="dl2/reco",
    mc_header_key="dl2/mc_header",
    useless_cols=[],
    max_files=0,
    eval_mean_events=False,
    verbose=False,
):
    """Function to

    Parameters
    ----------
    file_mask : str
        file mask for magic-cta-pipe DL2 files
    reco_key : str, optional
        key for DL2 reco files, by default "dl2/reco"
    mc_header_key : str, optional
        mc_header key, by default "dl2/mc_header"
    useless_cols : list, optional
        columns not used, by default []
    max_files : int, optional
        max number of files to be processed, 0 to process all of them, by default 0
    eval_mean_events : bool, optional
        evaluate mean of event, to get single row per obs_id, by default False
    verbose : bool, optional
        verbose mode, by default False

    Returns
    -------
    tuple
        events, pyirf_simu_info
    """
    # Map column names (DL2 -> pyirf)
    name_mapping = {
        "tel_alt": "pointing_alt",
        "tel_az": "pointing_az",
        "energy_reco": "reco_energy",
        "alt_reco": "reco_alt",
        "az_reco": "reco_az",
        # "intensity_width_1": "leakage_intensity_width_1",
        # "intensity_width_2": "leakage_intensity_width_2",
        "event_class_0": "gh_score",
        # "pos_angle_shift_reco": "reco_source_fov_offset",  # ???
    }

    # Map units
    unit_mapping = {
        "true_energy": u.TeV,
        "reco_energy": u.TeV,
        "pointing_alt": u.rad,
        "pointing_az": u.rad,
        "true_alt": u.rad,
        "true_az": u.rad,
        "reco_alt": u.rad,
        "reco_az": u.rad,
    }

    file_list = glob.glob(file_mask)

    if (max_files > 0) and (max_files < len(file_list)):
        file_list = file_list[:max_files]

    pyirf_simu_info = convert_simu_info_mcp_to_pyirf(file_list, mc_header_key)

    first_time = True
    for i, file in enumerate(file_list):
        if verbose:
            print(f"Analizing file: {file}")
        try:
            events_ = pd.read_hdf(file, key=reco_key).rename(columns=name_mapping)
            if useless_cols != []:
                events_ = events_.drop(useless_cols, axis=1, errors="ignore")
            if eval_mean_events:
                events_ = events_.mean(level=1)
            if first_time:
                events = events_
                first_time = False
            else:
                events = events.append(events_)
        except Exception as e:
            print(f"ERROR: skipping file {file}\n{e}")

    events = table.QTable.from_pandas(events)
    for k, v in unit_mapping.items():
        events[k] *= v

    return events, pyirf_simu_info


def plot_irfs_MAGIC_LST(config_file):
    """Plot IRFs for MAGIC and/or LST array using pyirf

    Parameters
    ----------
    config_file : str
        configuration file
    """
    print_title("Plot IRFs")

    cfg = load_cfg_file(config_file)

    load_default_plot_settings()

    # --- Open file ---
    # Open fits
    hdu_open = fits.open(
        os.path.join(cfg["irfs"]["save_dir"], "pyirf_eventdisplay.fits.gz")
    )

    # --- Plot Sensitivity ---
    sensitivity = QTable.read(hdu_open, hdu="SENSITIVITY")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(r"Minimal Flux Needed for 5$\mathrm{\sigma}$ Detection in 50 hours")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reconstructed energy (GeV)")
    unit = u.Unit("TeV cm-2 s-1")
    ax.set_ylabel(
        rf"$(E^2 \cdot \mathrm{{Flux Sensitivity}}) /$ ({unit.to_string('latex')})"
    )
    ax.grid(which="both")

    e = sensitivity["reco_energy_center"]
    s_mc = e ** 2 * sensitivity["flux_sensitivity"]
    e_low, e_high = sensitivity["reco_energy_low"], sensitivity["reco_energy_high"]
    plt.errorbar(
        e.to_value(u.GeV),
        s_mc.to_value(unit),
        xerr=[(e - e_low).to_value(u.GeV), (e_high - e).to_value(u.GeV)],
        label=f"MC gammas/protons",
    )
    # Plot magic sensitivity
    s = np.loadtxt(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../data/magic_sensitivity.txt",
        ),
        skiprows=1,
    )
    ax.loglog(
        s[:, 0],
        s[:, 3] * np.power(s[:, 0] / 1e3, 2),
        color="black",
        label="MAGIC (Aleksic et al. 2014)",
    )

    # Plot Crab SED
    plot_utils.plot_Crab_SED(
        ax, 100, 5 * u.GeV, 1e4 * u.GeV, label="100% Crab"
    )  # Energy in GeV
    plot_utils.plot_Crab_SED(
        ax, 10, 5 * u.GeV, 1e4 * u.GeV, linestyle="--", label="10% Crab"
    )  # Energy in GeV
    plot_utils.plot_Crab_SED(
        ax, 1, 5 * u.GeV, 1e4 * u.GeV, linestyle=":", label="1% Crab"
    )  # Energy in GeV

    plt.legend()

    save_plt(
        n=f"Sensitivity", rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    # --- Plot Angular Resolution ---
    ang_res = QTable.read(hdu_open, hdu="ANGULAR_RESOLUTION")
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlabel("Reconstructed energy (GeV)")
    ax.set_ylabel("Angular resolution (deg)")
    e = ang_res["reco_energy_center"]
    e_low, e_high = ang_res["reco_energy_low"], ang_res["reco_energy_high"]
    plt.errorbar(
        e.to_value(u.GeV),
        ang_res["angular_resolution"].to_value(u.deg),
        xerr=[(e - e_low).to_value(u.GeV), (e_high - e).to_value(u.GeV)],
    )
    save_plt(
        n=f"Angular_Resolution", rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    # --- Effective Area ---
    effective_area = QTable.read(hdu_open, hdu="EFFECTIVE_AREA")
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlabel("Reconstructed energy (GeV)")
    ax.set_ylabel(r"Effective Area ($\mathrm{m^2}$)")
    e_low, e_high = effective_area["ENERG_LO"][0], effective_area["ENERG_HI"][0]
    e = (e_low + e_high) / 2
    a = effective_area["EFFAREA"][0, 0]
    e = e_high
    m2 = u.m * u.m
    plt.errorbar(
        e.to_value(u.GeV), a.to_value(m2), xerr=(e - e_low).to_value(u.GeV),
    )
    save_plt(
        n=f"Effective_Area", rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

