import os
import sys
import time
import glob
import operator
import argparse
import numpy as np
import pandas as pd
from astropy import table
import astropy.units as u
from astropy.io import fits
from tables import open_file
import matplotlib.pyplot as plt

from ctapipe.io import HDF5TableReader
from ctapipe.containers import MCHeaderContainer
from ctapipe.core.container import Container, Field

from lstchain.io.io import read_dl2_to_pyirf, dl2_params_lstcam_key
from lstchain.mc import plot_utils

from pyirf.io.eventdisplay import read_eventdisplay_fits
from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution
from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
)
from pyirf.cut_optimization import optimize_gh_cut
from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
    background_2d,
)
from pyirf.io import (
    create_aeff2d_hdu,
    create_psf_table_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
    create_background_2d_hdu,
)

from magicctapipe.utils.filedir import *
from magicctapipe.utils.utils import *
from magicctapipe.irfs.utils import *
from magicctapipe.utils.plot import *

PARSER = argparse.ArgumentParser(
    description="Apply random forests. For stereo data.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
PARSER.add_argument(
    "-cfg",
    "--config_file",
    type=str,
    required=True,
    help="Configuration file, yaml format",
)


def make_irfs_MAGIC_LST(config_file):
    print_title("Make IRFs")

    cfg = load_cfg_file(config_file)

    # --- Initial variables ---
    # Observation time for sensitivity
    T_OBS = cfg["irfs"]["T_OBS"] * u.hour

    # scaling between on and off region.
    # Make off region 5 times larger than on region for better background statistics
    ALPHA = cfg["irfs"]["ALPHA"]

    # Radius to use for calculating bg rate
    MAX_BG_RADIUS = cfg["irfs"]["MAX_BG_RADIUS"] * u.deg

    # Gamma efficiency used for first calculation of the binned theta cuts
    # initial theta cuts are calculated using a fixed g/h cut corresponding to this
    # efficiency then g/h cuts are optimized after applying these initial theta cuts.
    INITIAL_GH_CUT_EFFICENCY = cfg["irfs"]["INITIAL_GH_CUT_EFFICENCY"]

    # gamma efficiency used for gh cuts calculation
    MAX_GH_CUT_EFFICIENCY = cfg["irfs"]["MAX_GH_CUT_EFFICIENCY"]
    GH_CUT_EFFICIENCY_STEP = cfg["irfs"]["GH_CUT_EFFICIENCY_STEP"]

    # Number of energy bins
    N_EBINS = cfg["irfs"]["N_EBINS"]

    # Energy range
    EMIN = cfg["irfs"]["EMIN"]
    EMAX = cfg["irfs"]["EMAX"]

    # Fixed cuts
    INTENSITY_CUT = cfg["irfs"]["INTENSITY_CUT"]
    LEAKAGE2_CUT = cfg["irfs"]["LEAKAGE2_CUT"]

    # Read hdf5 files into pyirf format
    if "useless_cols" in cfg["irfs"].keys():
        useless_cols = cfg["irfs"]["useless_cols"]
    else:
        useless_cols = []
    events_g, simu_info_g = read_dl2_mcp_to_pyirf_MAGIC_LST_list(
        file_mask=cfg["data_files"]["mc"]["test_sample"]["reco_h5"],
        useless_cols=useless_cols,
        max_files=cfg["irfs"]["max_files_gamma"],
    )
    events_p, simu_info_p = read_dl2_mcp_to_pyirf_MAGIC_LST_list(
        file_mask=cfg["data_files"]["data"]["test_sample"]["reco_h5"],
        useless_cols=useless_cols,
        max_files=cfg["irfs"]["max_files_proton"],
    )

    # --- Apply quality cuts ---
    for events in (events_g, events_p):
        events["good_events"] = (events["intensity"] >= INTENSITY_CUT) & (
            events["leakage_intensity_width_2"] <= LEAKAGE2_CUT
        )

    particles = {
        "gamma": {
            "events": events_g[events_g["good_events"]],
            "simulation_info": simu_info_g,
            "target_spectrum": CRAB_HEGRA,
        },
        "proton": {
            "events": events_p[events_p["good_events"]],
            "simulation_info": simu_info_p,
            "target_spectrum": IRFDOC_PROTON_SPECTRUM,
        },
    }

    # --- Manage MC gammas ---
    # Get simulated spectrum
    particles["gamma"]["simulated_spectrum"] = PowerLaw.from_simulation(
        particles["gamma"]["simulation_info"], T_OBS
    )
    # Reweight to target spectrum (Crab Hegra)
    particles["gamma"]["events"]["weight"] = calculate_event_weights(
        particles["gamma"]["events"]["true_energy"],
        particles["gamma"]["target_spectrum"],
        particles["gamma"]["simulated_spectrum"],
    )
    for prefix in ("true", "reco"):
        k = f"{prefix}_source_fov_offset"
        particles["gamma"]["events"][k] = calculate_source_fov_offset(
            particles["gamma"]["events"], prefix=prefix
        )
    particles["gamma"]["events"]["source_fov_offset"] = calculate_source_fov_offset(
        particles["gamma"]["events"]
    )
    # calculate theta / distance between reco and assumed source position
    # we handle only ON observations here, so the assumed source position
    # is the pointing position
    particles["gamma"]["events"]["theta"] = calculate_theta(
        particles["gamma"]["events"],
        assumed_source_az=particles["gamma"]["events"]["true_az"],
        assumed_source_alt=particles["gamma"]["events"]["true_alt"],
    )

    # --- Manage MC protons ---
    # Get simulated spectrum
    particles["proton"]["simulated_spectrum"] = PowerLaw.from_simulation(
        particles["proton"]["simulation_info"], T_OBS
    )
    # Reweight to target spectrum:
    particles["proton"]["events"]["weight"] = calculate_event_weights(
        particles["proton"]["events"]["true_energy"],
        particles["proton"]["target_spectrum"],
        particles["proton"]["simulated_spectrum"],
    )
    for prefix in ("true", "reco"):
        k = f"{prefix}_source_fov_offset"
        particles["proton"]["events"][k] = calculate_source_fov_offset(
            particles["proton"]["events"], prefix=prefix
        )
    # calculate theta / distance between reco and assumed source position
    # we handle only ON observations here, so the assumed source position
    # is the pointing position
    particles["proton"]["events"]["theta"] = calculate_theta(
        particles["proton"]["events"],
        assumed_source_az=particles["proton"]["events"]["pointing_az"],
        assumed_source_alt=particles["proton"]["events"]["pointing_alt"],
    )

    # Get events
    gammas = particles["gamma"]["events"]
    protons = particles["proton"]["events"]

    # --- Calculate the best cuts for sensitivity ---
    # Define bins
    # Sensitivity energy bins
    sensitivity_bins = np.logspace(np.log10(EMIN), np.log10(EMAX), N_EBINS + 1) * u.TeV

    # Data to optimize best cuts
    signal = gammas
    background = protons

    # Calculate an initial GH cut for calculating initial theta cuts, based on
    # INITIAL_GH_CUT_EFFICIENCY
    INITIAL_GH_CUT = np.quantile(signal["gh_score"], (1 - INITIAL_GH_CUT_EFFICENCY))

    # Initial $\theta$ cut
    # theta cut is 68 percent containmente of the gammas
    # for now with a fixed global, unoptimized score cut
    mask_theta_cuts = signal["gh_score"] >= INITIAL_GH_CUT
    theta_cuts = calculate_percentile_cut(
        signal["theta"][mask_theta_cuts],
        signal["reco_energy"][mask_theta_cuts],
        bins=sensitivity_bins,
        fill_value=np.nan * u.deg,
        percentile=68,
    )

    # evaluate the initial theta cut
    signal["selected_theta"] = evaluate_binned_cut(
        signal["theta"], signal["reco_energy"], theta_cuts, operator.le
    )

    # G/H cut optimization based on best sensitivity
    print("Optimizing G/H separation cut for best sensitivity")
    gh_cut_efficiencies = np.arange(
        GH_CUT_EFFICIENCY_STEP,
        MAX_GH_CUT_EFFICIENCY + GH_CUT_EFFICIENCY_STEP / 2,
        GH_CUT_EFFICIENCY_STEP,
    )
    sensitivity_step_2, gh_cuts = optimize_gh_cut(
        signal[signal["selected_theta"]],
        background,
        reco_energy_bins=sensitivity_bins,
        gh_cut_efficiencies=gh_cut_efficiencies,
        theta_cuts=theta_cuts,
        op=operator.ge,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )

    # Evaluate gh cut
    for tab in (gammas, protons):
        tab["selected_gh"] = evaluate_binned_cut(
            tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
        )

    # Setting of $\theta$ cut as 68% containment of events surviving the cuts
    theta_cuts_opt = calculate_percentile_cut(
        gammas[gammas["selected_gh"]]["theta"],
        gammas[gammas["selected_gh"]]["reco_energy"],
        sensitivity_bins,
        percentile=68,
        fill_value=0.32 * u.deg,
    )

    # Evaluate optimized cuts
    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts_opt, operator.le
    )
    gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]

    protons["selected"] = protons["selected_gh"]

    print(f"Selected gammas:  {gammas['selected'].sum()}")
    print(f"Selected protons: {protons['selected'].sum()}")

    # Crate event histograms
    gamma_hist = create_histogram_table(
        gammas[gammas["selected"]], bins=sensitivity_bins
    )
    proton_hist = estimate_background(
        protons[protons["selected"]],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cuts_opt,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )

    # Results dictionary
    res = {}

    # --- Sensitivity with MC gammas and protons ---
    sensitivity_mc = calculate_sensitivity(gamma_hist, proton_hist, alpha=ALPHA)
    # Plot Sensitivity curves
    # scale relative sensitivity by Crab flux to get the flux sensitivity
    spectrum = particles["gamma"]["target_spectrum"]
    sensitivity_mc["flux_sensitivity"] = sensitivity_mc[
        "relative_sensitivity"
    ] * spectrum(sensitivity_mc["reco_energy_center"])

    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    unit = u.Unit("TeV cm-2 s-1")

    e = sensitivity_mc["reco_energy_center"]
    s_mc = e ** 2 * sensitivity_mc["flux_sensitivity"]

    res["sensitivity"] = {
        "energy": e.to_value(u.GeV),
        "value": s_mc.to_value(unit),
        "energy_err": (
            sensitivity_mc["reco_energy_high"] - sensitivity_mc["reco_energy_low"]
        ).to_value(u.GeV)
        / 2,
    }
    plt.errorbar(
        res["sensitivity"]["energy"],
        res["sensitivity"]["value"],
        xerr=res["sensitivity"]["energy_err"],
        label=f"MC gammas/protons",
    )

    # Plot magic sensitivity
    s = np.loadtxt(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "./data/magic_sensitivity.txt",
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

    # Style settings
    plt.title("Minimal Flux Needed for 5$\mathrm{\sigma}$ Detection in 50 hours")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Reconstructed energy [GeV]")
    plt.ylabel(
        rf"$(E^2 \cdot \mathrm{{Flux Sensitivity}}) /$ ({unit.to_string('latex')})"
    )
    plt.grid(which="both")
    plt.legend()
    rdir = "IRFs"
    if not os.path.exists(rdir):
        os.mkdir(rdir)
    fig_name = f"Sensitivity"
    save_plt(
        n=fig_name, rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    # --- Rates ---
    fix, ax = plt.subplots(figsize=(10, 8))
    rate_gammas = gamma_hist["n_weighted"] / T_OBS.to(u.min)
    area_ratio_p = (1 - np.cos(theta_cuts_opt["cut"])) / (1 - np.cos(MAX_BG_RADIUS))
    rate_proton = proton_hist["n_weighted"] * area_ratio_p / T_OBS.to(u.min)

    plt.errorbar(
        0.5
        * (gamma_hist["reco_energy_low"] + gamma_hist["reco_energy_high"]).to_value(
            u.TeV
        ),
        rate_gammas.to_value(1 / u.min),
        xerr=0.5
        * (gamma_hist["reco_energy_high"] - gamma_hist["reco_energy_low"]).to_value(
            u.TeV
        ),
        label="Gammas MC",
    )

    plt.errorbar(
        0.5
        * (proton_hist["reco_energy_low"] + proton_hist["reco_energy_high"]).to_value(
            u.TeV
        ),
        rate_proton.to_value(1 / u.min),
        xerr=0.5
        * (proton_hist["reco_energy_high"] - proton_hist["reco_energy_low"]).to_value(
            u.TeV
        ),
        label="Protons MC",
    )
    plt.legend()
    plt.ylabel("Rate events/min")
    plt.xlabel(r"$E_\mathrm{reco} / \mathrm{TeV}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which="both")
    fig_name = f"Rates"
    save_plt(
        n=fig_name, rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    # --- Cuts ---
    fix, ax = plt.subplots(figsize=(10, 8))
    plt.errorbar(
        0.5 * (theta_cuts["low"] + theta_cuts["high"]).to_value(u.TeV),
        (theta_cuts["cut"] ** 2).to_value(u.deg ** 2),
        xerr=0.5 * (theta_cuts["high"] - theta_cuts["low"]).to_value(u.TeV),
        ls="",
    )
    plt.ylabel(r"$\theta^2$-cut")
    plt.xlabel(r"$E_\mathrm{reco} / \mathrm{TeV}$")
    plt.xscale("log")
    plt.grid(which="both")
    fig_name = f"Cut_Theta2"
    save_plt(
        n=fig_name, rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    fix, ax = plt.subplots(figsize=(10, 8))
    plt.errorbar(
        0.5 * (gh_cuts["low"] + gh_cuts["high"]).to_value(u.TeV),
        gh_cuts["cut"],
        xerr=0.5 * (gh_cuts["high"] - gh_cuts["low"]).to_value(u.TeV),
        ls="",
    )
    plt.ylabel("G/H-cut")
    plt.xlabel(r"$E_\mathrm{reco} / \mathrm{TeV}$")
    plt.xscale("log")
    plt.grid(which="both")
    fig_name = f"Cut_GH"
    save_plt(
        n=fig_name, rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    # --- Angular Resolution ---
    fix, ax = plt.subplots(figsize=(10, 8))

    selected_events_gh = table.vstack(
        gammas[gammas["selected_gh"]], protons[protons["selected_gh"]]
    )

    ang_res = angular_resolution(
        selected_events_gh[selected_events_gh["selected_gh"]], sensitivity_bins,
    )

    plt.errorbar(
        0.5 * (ang_res["true_energy_low"] + ang_res["true_energy_high"]),
        ang_res["angular_resolution"],
        xerr=0.5 * (ang_res["true_energy_high"] - ang_res["true_energy_low"]),
        ls="",
    )

    # Style settings
    # plt.xlim(1.0e-2, 2.0e2)
    # plt.ylim(0.5e-1, 1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True energy / TeV")
    plt.ylabel("Angular Resolution / deg")
    plt.grid(which="both")
    fig_name = f"Angular_Resolution"
    save_plt(
        n=fig_name, rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    # --- Energy resolution ---
    fix, ax = plt.subplots(figsize=(10, 8))
    selected_events = table.vstack(
        gammas[gammas["selected"]], protons[protons["selected"]]
    )
    selected_events = table.vstack(
        gammas[gammas["selected"]], protons[protons["selected"]]
    )
    bias_resolution = energy_bias_resolution(selected_events, sensitivity_bins,)

    # Plot function
    plt.errorbar(
        0.5
        * (bias_resolution["true_energy_low"] + bias_resolution["true_energy_high"]),
        bias_resolution["resolution"],
        xerr=0.5
        * (bias_resolution["true_energy_high"] - bias_resolution["true_energy_low"]),
        ls="",
    )
    plt.xscale("log")

    # Style settings
    plt.xlabel(r"$E_\mathrm{True} / \mathrm{TeV}$")
    plt.ylabel("Energy resolution")
    plt.grid(which="both")
    plt.legend(loc="best")
    fig_name = f"Energy_Resolution"
    save_plt(
        n=fig_name, rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    # --- Reco Alt/Az for MC selected events ---
    fix, ax = plt.subplots()

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 4))
    emin_bins = [0.0, 0.1, 0.5, 1, 5] * u.TeV
    emax_bins = [0.1, 0.5, 1, 5, 10] * u.TeV

    for i, ax in enumerate(axs):
        events = selected_events[
            (selected_events["reco_energy"] > emin_bins[i])
            & (selected_events["reco_energy"] < emax_bins[i])
        ]
        pcm = ax.hist2d(
            events["reco_az"].to_value(u.deg),
            events["reco_alt"].to_value(u.deg),
            bins=50,
        )
        ax.title.set_text(
            "%.1f-%.1f TeV" % (emin_bins[i].to_value(), emax_bins[i].to_value())
        )
        ax.set_xlabel(r"Az ($\mathrm{\deg}$)")
        ax.set_ylabel(r"Alt ($\mathrm{\deg}$)")
        fig.colorbar(pcm[3], ax=ax)
    fig_name = f"Reco_AltAz"

    # --- Checks on number of islands ---
    fig, ax = plt.subplots(figsize=(10, 8))
    gammas_selected = gammas[
        (gammas["selected"]) & (gammas["reco_energy"] > 1.0 * u.TeV)
    ]
    plt.hist(gammas_selected["num_islands"], bins=10, range=(0.5, 10.5))
    plt.yscale("log")
    fig_name = f"Num_Islands_Gamma"
    save_plt(
        n=fig_name, rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    protons_selected = protons[
        (protons["selected"]) & (protons["reco_energy"] > 1.0 * u.TeV)
    ]
    plt.hist(protons_selected["num_islands"], bins=10, range=(0.5, 10.5))
    plt.yscale("log")
    fig_name = f"Num_Islands_Proton"
    save_plt(
        n=fig_name, rdir=cfg["irfs"]["save_dir"], vect="pdf",
    )

    # --- Save results dictionary ---
    results_file = os.path.join(cfg["irfs"]["save_dir"], "IRFs.yaml")
    save_yaml_np(res, results_file)
    # yaml.dump(res, open(results_file, "w"), default_flow_style=False)


if __name__ == "__main__":
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()

    make_irfs_MAGIC_LST(kwargs["config_file"])

    print_elapsed_time(start_time, time.time())
