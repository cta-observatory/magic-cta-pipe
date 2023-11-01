import glob
import os

import astropy.units as u
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from astropy import table
from astropy.io import fits
from astropy.table import QTable
from pyirf.simulations import SimulatedEventsInfo

from ..utils import load_default_plot_settings, print_title, read_mc_header, save_plt

__all__ = [
    "read_simu_info_mcp_sum_num_showers",
    "convert_simu_info_mcp_to_pyirf",
    "read_dl2_mcp_to_pyirf_MAGIC_LST_list",
    "plot_sensitivity",
    "plot_en_res_bias",
    "plot_en_res_resolution",
    "plot_ang_res",
    "plot_effective_area",
    "plot_gamma_eff_gh",
    "plot_irfs_MAGIC_LST",
    "plot_MARS_sensitivity",
    "plot_MAGIC_reference_sensitivity",
]


def read_simu_info_mcp_sum_num_showers(file_list, mc_header_key="dl2/mc_header"):
    """Function to read simulation information from DL2 files and sum the simultated
    showers. Assumes that all the mc_headers are equal

    Parameters
    ----------
    file_list : list
        magic-cta-pipe DL2 file list
    mc_header_key : str
        mc_header key, by default "dl2/mc_header"

    Returns
    -------
    pandas.DataFrame
        mc_header with sum num showers
    """
    d = read_mc_header(file_list[0], mc_header_key)
    num_showers = 0
    if len(file_list) > 1:
        for i, file in enumerate(file_list):
            num_showers += int(read_mc_header(file, mc_header_key)["num_showers"])
    else:
        num_showers = int(d["num_showers"].sum())
    d["num_showers"] = num_showers
    # In the case of merged DL2 now num_showers is a list where each value is the sum
    # of the showers... not very smart
    return d


def convert_simu_info_mcp_to_pyirf(file_list, mc_header_key="dl2/mc_header"):
    """Function to convert simulation information from magic-cta-pipe DL2 files to
    pyirf format

    Parameters
    ----------
    file_list : list
        magic-cta-pipe DL2 file list
    mc_header_key : str
        mc_header key, by default "dl2/mc_header"

    Returns
    -------
    pyirf.simulations.SimulatedEventsInfo
        pyirf_simu_info
    """
    simu_info = read_simu_info_mcp_sum_num_showers(file_list, mc_header_key)
    # very bad way way to separate file_list and merged file
    if len(file_list) > 1:
        pyirf_simu_info = SimulatedEventsInfo(
            n_showers=int(simu_info.num_showers) * int(simu_info.shower_reuse),
            energy_min=float(simu_info.energy_range_min) * u.TeV,
            energy_max=float(simu_info.energy_range_max) * u.TeV,
            max_impact=float(simu_info.max_scatter_range) * u.m,
            spectral_index=float(simu_info.spectral_index),
            viewcone=float(simu_info.max_viewcone_radius) * u.deg,
        )
    else:
        pyirf_simu_info = SimulatedEventsInfo(
            n_showers=int(simu_info.num_showers.iloc[0])
            * int(simu_info.shower_reuse.iloc[0]),
            energy_min=float(simu_info.energy_range_min.iloc[0]) * u.TeV,
            energy_max=float(simu_info.energy_range_max.iloc[0]) * u.TeV,
            max_impact=float(simu_info.max_scatter_range.iloc[0]) * u.m,
            spectral_index=float(simu_info.spectral_index.iloc[0]),
            viewcone=float(simu_info.max_viewcone_radius.iloc[0]) * u.deg,
        )
    # Regarding the max_impact, in pyirf is used for:
    # A = np.pi * simulated_event_info.max_impact ** 2
    return pyirf_simu_info


def read_dl2_mcp_to_pyirf_MAGIC_LST_list(
    file_mask,
    reco_key="dl2/reco",
    mc_header_key="dl2/mc_header",
    useless_cols=[],
    cuts="",
    max_files=0,
    eval_mean_events=False,
    verbose=False,
):
    """Function to read dl2 file and convert to pyirf format

    Parameters
    ----------
    file_mask : str
        file mask for magic-cta-pipe DL2 files
    reco_key : str
        key for DL2 reco files, by default "dl2/reco"
    mc_header_key : str
        mc_header key, by default "dl2/mc_header"
    useless_cols : list
        columns not used, by default []
    cuts : str
        cuts on dl2 events, by default ""
    max_files : int
        max number of files to be processed, 0 to process all of them, by default 0
    eval_mean_events : bool
        evaluate mean of event, to get single row per obs_id, by default False
    verbose : bool
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
            events_ = pd.read_hdf(file, key=reco_key)
            if cuts != "":
                print(f"Applying cuts: {cuts}")
                print(len(events_))
                events_ = events_.query(cuts)
                # l_ = ["obs_id", "event_id"]
                # events_["multiplicity"] = events_["intensity"].groupby(level=l_).count()
                # events_ = events_.query(cuts)
                print(len(events_))
            events_ = events_.rename(columns=name_mapping)
            if useless_cols != []:
                events_ = events_.drop(useless_cols, axis=1, errors="ignore")
            if eval_mean_events:
                events_ = events_.groupby(["obs_id", "event_id"]).mean()
                # events_ = events_.mean(level=1)
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


def plot_sensitivity(data, unit, label, ax=None, **kwargs):
    """Plot sensitivity

    Parameters
    ----------
    data : astropy.table.QTable
        sensitivity data
    unit : str
        sensitivity unit
    label : str
        label for plot
    ax : matplotlib.axes, optional
        give it if you want to specify the axis, by default None
    **kwargs : dict, optional
    """
    e = data["reco_energy_center"]
    s_mc = e**2 * data["flux_sensitivity"]
    e_low, e_high = data["reco_energy_low"], data["reco_energy_high"]
    ax_ = ax if ax is not None else plt
    plt_ = ax_.errorbar(
        e.to_value(u.GeV),
        s_mc.to_value(unit),
        xerr=[(e - e_low).to_value(u.GeV), (e_high - e).to_value(u.GeV)],
        label=label,
        **kwargs,
    )
    return plt_


def plot_en_res_bias(data, label, **kwargs):
    """Plot energy resolution bias

    Parameters
    ----------
    data : astropy.table.QTable
        angular resolution data
    label : str
        label for plot
    **kwargs : dict
    """
    e = data["reco_energy_center"]
    e_low, e_high = data["reco_energy_low"], data["reco_energy_high"]
    plt.errorbar(
        e.to_value(u.GeV),
        data["bias"],
        xerr=[(e - e_low).to_value(u.GeV), (e_high - e).to_value(u.GeV)],
        label=label,
        **kwargs,
    )


def plot_en_res_resolution(data, label, **kwargs):
    """Plot energy resolution resolution

    Parameters
    ----------
    data : astropy.table.QTable
        angular resolution data
    label : str
        label for plot
    **kwargs : dict
    """
    e = data["reco_energy_center"]
    e_low, e_high = data["reco_energy_low"], data["reco_energy_high"]
    plt.errorbar(
        e.to_value(u.GeV),
        data["resolution"],
        xerr=[(e - e_low).to_value(u.GeV), (e_high - e).to_value(u.GeV)],
        label=label,
        **kwargs,
    )


def plot_ang_res(data, label, **kwargs):
    """Plot angular resolution

    Parameters
    ----------
    data : astropy.table.QTable
        angular resolution data
    label : str
        label for plot
    **kwargs : dict
    """
    e = data["reco_energy_center"]
    e_low, e_high = data["reco_energy_low"], data["reco_energy_high"]
    plt.errorbar(
        e.to_value(u.GeV),
        data["angular_resolution"].to_value(u.deg),
        xerr=[(e - e_low).to_value(u.GeV), (e_high - e).to_value(u.GeV)],
        label=label,
        **kwargs,
    )


def plot_effective_area(data, label, **kwargs):
    """Plot effective area

    Parameters
    ----------
    data : astropy.table.QTable
        effective area data
    label : str
        label for plot
    **kwargs : dict
    """
    e_low, e_high = data["ENERG_LO"][0], data["ENERG_HI"][0]
    e = (e_low + e_high) / 2
    a = data["EFFAREA"][0, 0]
    e = e_high
    m2 = u.m * u.m
    plt.errorbar(
        e.to_value(u.GeV),
        a.to_value(m2),
        xerr=(e - e_low).to_value(u.GeV),
        label=label,
        **kwargs,
    )


def plot_gamma_eff_gh(gamma_efficiency, gh_cuts, sensitivity):
    """Plot gamma efficiency and gh cuts

    Parameters
    ----------
    gamma_efficiency : astropy.table.QTable
        gamma efficiency
    gh_cuts : astropy.table.QTable
        gh cuts
    sensitivity : astropy.table.QTable
        sensitivity
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Energy (GeV)")
    ax.set_xscale("log")
    # Consider only energy where the sensitivity is estimated
    m_ = sensitivity["n_signal"] > 0
    plt.plot(gh_cuts["center"][m_].to(u.GeV), gh_cuts["cut"][m_], "-o", label="GH cut")
    plt.plot(
        gamma_efficiency["center"][m_].to(u.GeV),
        gamma_efficiency["eff_gh"][m_],
        "-o",
        label="Gamma Efficiency GH",
    )
    plt.plot(
        gamma_efficiency["center"][m_].to(u.GeV),
        gamma_efficiency["eff"][m_],
        "-o",
        label="Gamma Efficiency",
    )
    plt.legend()


def plot_irfs_MAGIC_LST(config_file, irfs_dir):
    """Plot IRFs for MAGIC and/or LST array using pyirf

    Parameters
    ----------
    config_file : str
        configuration file
    """
    print_title("Plot IRFs")

    load_default_plot_settings()

    # --- Open file ---
    # Open fits
    hdu_open = fits.open(os.path.join(irfs_dir, "pyirf.fits.gz"))

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

    plot_sensitivity(data=sensitivity, unit=unit, label="MC")

    # Plot magic sensitivity
    plot_MAGIC_reference_sensitivity(ax)

    # Plot Crab SED
    # plot_utils.plot_Crab_SED(
    #    ax, 100, 5 * u.GeV, 1e4 * u.GeV, label="100% Crab"
    # )  # Energy in GeV
    # plot_utils.plot_Crab_SED(
    #    ax, 10, 5 * u.GeV, 1e4 * u.GeV, linestyle="--", label="10% Crab"
    # )  # Energy in GeV
    # plot_utils.plot_Crab_SED(
    #    ax, 1, 5 * u.GeV, 1e4 * u.GeV, linestyle=":", label="1% Crab"
    # )  # Energy in GeV

    plt.legend()

    save_plt(
        n="Sensitivity",
        rdir=irfs_dir,
        vect="pdf",
    )

    # --- Plot Angular Resolution ---
    ang_res = QTable.read(hdu_open, hdu="ANGULAR_RESOLUTION")
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlabel("Reconstructed energy (GeV)")
    ax.set_ylabel("Angular resolution (deg)")

    plot_ang_res(data=ang_res, label="Angular Resolution")
    save_plt(
        n="Angular_Resolution",
        rdir=irfs_dir,
        vect="pdf",
    )

    # --- Effective Area ---
    effective_area = QTable.read(hdu_open, hdu="EFFECTIVE_AREA")
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True energy (GeV)")
    ax.set_ylabel(r"Effective Area ($\mathrm{m^2}$)")
    plot_effective_area(data=effective_area, label="Effective Area")
    save_plt(
        n="Effective_Area",
        rdir=irfs_dir,
        vect="pdf",
    )

    # --- GH cuts and Gamma Efficiency ---
    gh_cuts = QTable.read(hdu_open, hdu="GH_CUTS")
    gamma_efficiency = QTable.read(hdu_open, hdu="GAMMA_EFFICIENCY")
    plot_gamma_eff_gh(gamma_efficiency, gh_cuts, sensitivity)
    save_plt(
        n="Gamma_Efficiency",
        rdir=irfs_dir,
        vect="pdf",
    )


def plot_MARS_sensitivity(array="4LST", label="", print_data=False, **kwargs):
    """Plot Sensitivity from MARS

    Parameters
    ----------
    array : str
        telescope array, by default "4LST"

        Possibilities:

        * "4LST": file = "magic-cta-pipe/data/MARS_4LST.txt"
        * "MAGIC": file = "magic-cta-pipe/data/MARS_MAGIC.txt"
        * "MAGIC_LST1": file = "magic-cta-pipe/data/MARS_MAGIC_LST1.txt"

    label : str
        custom plot label, by default ""
    print_data : bool
        print data, by default False
    """
    available_arrays = ["4LST", "MAGIC", "MAGIC_LST1"]

    if array in available_arrays:
        f_ = f"MARS_{array}.txt"
    else:
        print("Invalid array")
        return
    if label == "":
        label = f"{array} Di Pierro et al. ICRC2019"

    # Load data
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"../../data/{f_}")
    d = np.loadtxt(file, unpack=True)
    e_mars_, s_mars_, err_e_mars_, err_s_mars_ = [d_[:-1] for d_ in d]

    # Units
    unit_file = u.Unit("erg cm-2 s-1")
    unit = u.Unit("TeV cm-2 s-1")

    # Convert sensitivity from erg/(s cm**2) to TeV/(s cm**2)
    s_mars = (s_mars_ * unit_file).to(unit).value
    err_s_mars = (err_s_mars_ * unit_file).to(unit).value

    # Convert energy from log(E/TeV) to GeV
    e_mars = ((10 ** (e_mars_)) * u.TeV).to(u.GeV).value
    e_low = ((10 ** (e_mars_ - err_e_mars_)) * u.TeV).to(u.GeV).value
    e_high = ((10 ** (e_mars_ + err_e_mars_)) * u.TeV).to(u.GeV).value
    err_e_mars = [(e_mars - e_low), (e_high - e_mars)]

    # Set default values in kwargs
    if "linestyle" not in kwargs.keys():
        kwargs["linestyle"] = "--"

    # Plot
    plt.errorbar(
        e_mars, s_mars, xerr=err_e_mars, yerr=err_s_mars, label=label, **kwargs
    )
    if print_data:
        print("Energy\t\tDirection")
        [print(f"{l_[0]}\t{l_[1]}") for l_ in list(map(list, zip(*[e_mars, s_mars])))]


def plot_MAGIC_reference_sensitivity(ax, **kwargs):
    """Plot MAGIC reference sensitivity

    Parameters
    ----------
    ax : matplotlib.axes
        ax where you want to plot the sensitivity
    **kwargs : dict
    """
    d = np.loadtxt(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../data/MAGIC_Sensitivity_magicmpp.txt",
        ),
        unpack=True,
    )
    e, e_low, e_high = d[0], d[1], d[2]
    s, err_s = d[5], d[6]
    if "label" not in kwargs.keys():
        kwargs["label"] = "MAGIC Reference magic.mpp.mpg.de"
    if "color" not in kwargs.keys():
        kwargs["color"] = "k"
    ax.errorbar(e, s, xerr=[(e - e_low), (e_high - e)], yerr=err_s, **kwargs)
