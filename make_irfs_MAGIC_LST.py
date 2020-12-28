import os
import sys
import glob
import operator
import numpy as np
import pandas as pd
from astropy import table
import astropy.units as u
from astropy.io import fits
from astropy.table import QTable
from tables import open_file
import matplotlib.pyplot as plt

from ctapipe.io import HDF5TableReader
from ctapipe.containers import MCHeaderContainer
from ctapipe.core.container import Container, Field

from lstchain.io.io import read_dl2_to_pyirf, dl2_params_lstcam_key
from lstchain.mc import plot_utils

from pyirf.simulations import SimulatedEventsInfo
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


# ### Initial variables


# Observation time for sensitivity
T_OBS = 50 * u.hour
# scaling between on and off region.
# Make off region 5 times larger than on region for better
# background statistics
ALPHA = 1 / 5  # 1/5

# Radius to use for calculating bg rate
MAX_BG_RADIUS = 5.0 * u.deg  # 1.0 * u.deg

# gamma efficiency used for first calculation of the binned theta cuts
# initial theta cuts are calculated using a fixed g/h cut corresponding to this efficiency
# then g/h cuts are optimized after applying these initial theta cuts.
INITIAL_GH_CUT_EFFICENCY = 0.4

# gamma efficiency used for gh cuts calculation
MAX_GH_CUT_EFFICIENCY = 0.8  # 0.8
GH_CUT_EFFICIENCY_STEP = 0.01

# Number of energy bins
N_EBINS = 20
# Energy range
EMIN = 0.05  # 0.05  #TeV
EMAX = 50  # 50  #TeV

# Fixed cuts
INTENSITY_CUT = 100  # 100
LEAKAGE2_CUT = 0.2  # 0.2


def read_mc_header(file):
    return pd.read_hdf(file, key="dl2/mc_header")


def read_simu_info_mcp_sum_num_showers(file_list, max_files):
    d = read_mc_header(file_list[0])
    num_showers = 0
    for i, file in enumerate(file_list):
        if max_files > 0 and (i + 1) == max_files:
            break
        num_showers += int(read_mc_header(file)["num_showers"])
    d["num_showers"] = num_showers
    return d


def convert_simu_info_mcp_to_pyirf(file_list, max_files):
    simu_info = read_simu_info_mcp_sum_num_showers(file_list, max_files)
    pyirf_simu_info = SimulatedEventsInfo(
        n_showers=int(simu_info.num_showers) * int(simu_info.shower_reuse),
        energy_min=float(simu_info.energy_range_min) * u.TeV,
        energy_max=float(simu_info.energy_range_max) * u.TeV,
        max_impact=float(simu_info.max_scatter_range) * u.m,
        spectral_index=float(simu_info.spectral_index),
        viewcone=float(simu_info.max_viewcone_radius) * u.deg,
    )
    return pyirf_simu_info


useless_cols = [
    "pixels_width_1",
    "pixels_width_2",
    "alt",
    "alt_uncert",
    "az",
    "az_uncert",
    "core_x",
    "core_y",
    "core_uncert",
    "h_max",
    "h_max_uncert",
    "is_valid",
    "tel_ids",
    "average_intensity",
    "goodness_of_fit",
    "multiplicity",
    "disp_reco",
    "disp_reco_err",
    "az_reco_mean",
    "alt_reco_mean",
    "energy_reco_err",
    "energy_reco_mean",
    "event_class_1",
    "event_class_0_mean",
    "event_class_1_mean",
    "x",
    "y",
    "r",
    "phi",
    "length",
    "width",
    "psi",
    "skewness",
    "kurtosis",
    "slope",
    "slope_err",
    "intercept",
    "intercept_err",
    "deviation",
]


def read_dl2_mcp_to_pyirf_MAGIC_LST_list(file_mask, reco_key="dl2/reco", max_files=0):
    """
    Read DL2 files from magic-cta-pipe (applyRFs.py) and convert into pyirf 
    internal format
    Parameters
    ----------
    filename: path
    Returns
    -------
    `astropy.table.QTable`, `pyirf.simulations.SimulatedEventsInfo`
    """
    name_mapping = {
        "tel_alt": "pointing_alt",
        "tel_az": "pointing_az",
        "energy_reco": "reco_energy",
        "alt_reco": "reco_alt",
        "az_reco": "reco_az",
        "intensity_width_1": "leakage_intensity_width_1",
        "intensity_width_2": "leakage_intensity_width_2",
        "event_class_0": "gh_score",
        "pos_angle_shift_reco": "reco_source_fov_offset",  # ???
    }

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
    pyirf_simu_info = convert_simu_info_mcp_to_pyirf(file_list, max_files)

    for i, file in enumerate(file_list):
        events_ = pd.read_hdf(file, key=reco_key).rename(columns=name_mapping)
        events_ = events_.drop(useless_cols, axis=1, errors="ignore")
        if i == 0:
            events = events_
        else:
            events = events.append(events_)
        if max_files > 0 and (i + 1) == max_files:
            break
    events = table.QTable.from_pandas(events)
    for k, v in unit_mapping.items():
        events[k] *= v

    return events, pyirf_simu_info


# ### Read hdf5 files into pyirf format


t_ = "MAGIC_4LST"
tag = ""
fm = f"/fefs/aswg/workspace/davide.depaoli/CTA-MC/Prod5/Analysis/{t_}/DL2/gamma_00deg_test_reco/*.h5"
events_g, simu_info_g = read_dl2_mcp_to_pyirf_MAGIC_LST_list(fm)
fm = f"/fefs/aswg/workspace/davide.depaoli/CTA-MC/Prod5/Analysis/{t_}/DL2/proton_test_reco/*.h5"
events_p, simu_info_p = read_dl2_mcp_to_pyirf_MAGIC_LST_list(fm)


# ### Apply quality cuts


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


# Manage MC gammas:
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


# Manage MC protons:
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


# ## Calculate the best cuts for sensitivity

# ### Define bins


# Sensitivity energy bins
sensitivity_bins = np.logspace(np.log10(EMIN), np.log10(EMAX), N_EBINS + 1) * u.TeV


# ### Data to optimize best cuts


signal = gammas
background = protons


# Calculate an initial GH cut for calculating initial theta cuts, based on INITIAL_GH_CUT_EFFICIENCY
INITIAL_GH_CUT = np.quantile(signal["gh_score"], (1 - INITIAL_GH_CUT_EFFICENCY))
INITIAL_GH_CUT


# ### Initial $\theta$ cut


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


# ### Run block below for G/H cut optimization based on best sensitivity


log.info("Optimizing G/H separation cut for best sensitivity")
gh_cut_efficiencies = np.arange(
    GH_CUT_EFFICIENCY_STEP,
    MAX_GH_CUT_EFFICIENCY + GH_CUT_EFFICIENCY_STEP / 2,
    GH_CUT_EFFICIENCY_STEP,
)
sensitivity_step_2, gh_cuts = optimize_gh_cut(
    signal[signal["selected_theta"]],
    #     signal,
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


# ### Setting of $\theta$ cut as 68% containment of events surviving the cuts
theta_cuts_opt = calculate_percentile_cut(
    gammas[gammas["selected_gh"]]["theta"],
    gammas[gammas["selected_gh"]]["reco_energy"],
    sensitivity_bins,
    percentile=68,
    fill_value=0.32 * u.deg,
)


# ### Evaluate optimized cuts


gammas["selected_theta"] = evaluate_binned_cut(
    gammas["theta"], gammas["reco_energy"], theta_cuts_opt, operator.le
)
gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]

protons["selected"] = protons["selected_gh"]

print(f"Selected gammas:  {gammas['selected'].sum()}")
print(f"Selected protons: {protons['selected'].sum()}")


# ### Crate event histograms


gamma_hist = create_histogram_table(gammas[gammas["selected"]], bins=sensitivity_bins)
proton_hist = estimate_background(
    protons[protons["selected"]],
    reco_energy_bins=sensitivity_bins,
    theta_cuts=theta_cuts_opt,
    alpha=ALPHA,
    background_radius=MAX_BG_RADIUS,
)


# ### Sensitivity with MC gammas and protons


sensitivity_mc = calculate_sensitivity(gamma_hist, proton_hist, alpha=ALPHA)
sensitivity_mc


# ### Plot Sensitivity curves


# scale relative sensitivity by Crab flux to get the flux sensitivity
spectrum = particles["gamma"]["target_spectrum"]
sensitivity_mc["flux_sensitivity"] = sensitivity_mc["relative_sensitivity"] * spectrum(
    sensitivity_mc["reco_energy_center"]
)


plt.figure(figsize=(12, 8))
ax = plt.axes()
unit = u.Unit("TeV cm-2 s-1")

base_name = t_

e = sensitivity_mc["reco_energy_center"]

s_mc = e ** 2 * sensitivity_mc["flux_sensitivity"]

plt.errorbar(
    e.to_value(u.GeV),
    s_mc.to_value(unit),
    xerr=(
        sensitivity_mc["reco_energy_high"] - sensitivity_mc["reco_energy_low"]
    ).to_value(u.GeV)
    / 2,
    label=f"MC gammas/protons {base_name}",
)

# Plot magic sensitivity
s = np.loadtxt("magic_sensitivity.txt", skiprows=1,)
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
plt.title("Minimal Flux Needed for 5σ Detection in 50 hours")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Reconstructed energy [GeV]")
plt.ylabel(rf"$(E^2 \cdot \mathrm{{Flux Sensitivity}}) /$ ({unit.to_string('latex')})")
plt.grid(which="both")
plt.legend()
rdir = "Results"
if not os.path.exists(rdir):
    os.mkdir(rdir)
fig_name = os.path.join(rdir, f"Sensitivity_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)
# plt.show()


# ### Rates

fix, ax = plt.subplots()
rate_gammas = gamma_hist["n_weighted"] / T_OBS.to(u.min)
area_ratio_p = (1 - np.cos(theta_cuts_opt["cut"])) / (1 - np.cos(MAX_BG_RADIUS))
rate_proton = proton_hist["n_weighted"] * area_ratio_p / T_OBS.to(u.min)

plt.errorbar(
    0.5
    * (gamma_hist["reco_energy_low"] + gamma_hist["reco_energy_high"]).to_value(u.TeV),
    rate_gammas.to_value(1 / u.min),
    xerr=0.5
    * (gamma_hist["reco_energy_high"] - gamma_hist["reco_energy_low"]).to_value(u.TeV),
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
fig_name = os.path.join(rdir, f"Rates_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)
# plt.show()


# ### Cuts

fix, ax = plt.subplots()

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
fig_name = os.path.join(rdir, f"Cuts_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)
# plt.show()

fix, ax = plt.subplots()

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
fig_name = os.path.join(rdir, f"Cuts02_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)
# plt.show()


# ### Angular Resolution
fix, ax = plt.subplots()


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
plt.xlim(1.0e-2, 2.0e2)
plt.ylim(0.5e-1, 1)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True energy / TeV")
plt.ylabel("Angular Resolution / deg")
plt.grid(which="both")
fig_name = os.path.join(rdir, f"AngularRes_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)
# plt.show()


# ### Energy resolution
fix, ax = plt.subplots()


selected_events = table.vstack(gammas[gammas["selected"]], protons[protons["selected"]])


selected_events = table.vstack(gammas[gammas["selected"]], protons[protons["selected"]])

bias_resolution = energy_bias_resolution(selected_events, sensitivity_bins,)

# Plot function
plt.errorbar(
    0.5 * (bias_resolution["true_energy_low"] + bias_resolution["true_energy_high"]),
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
fig_name = os.path.join(rdir, f"EnergyRes_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)
# plt.show()


# ### Reco Alt/Az for MC selected events
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
        events["reco_az"].to_value(u.deg), events["reco_alt"].to_value(u.deg), bins=50
    )
    ax.title.set_text(
        "%.1f-%.1f TeV" % (emin_bins[i].to_value(), emax_bins[i].to_value())
    )
    ax.set_xlabel("Az (º)")
    ax.set_ylabel("Alt (º)")
    fig.colorbar(pcm[3], ax=ax)
fig_name = os.path.join(rdir, f"RecoAltAz_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)
# plt.show()


# ### Reco camera coordinates for real on events

# ### Checks on number of islands

fig, ax = plt.subplots()
gammas_selected = gammas[(gammas["selected"]) & (gammas["reco_energy"] > 1.0 * u.TeV)]
plt.hist(gammas_selected["num_islands"], bins=10, range=(0.5, 10.5))
plt.yscale("log")
fig_name = os.path.join(rdir, f"NumIslandsGamma_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)


fig, ax = plt.subplots()
protons_selected = protons[
    (protons["selected"]) & (protons["reco_energy"] > 1.0 * u.TeV)
]
plt.hist(protons_selected["num_islands"], bins=10, range=(0.5, 10.5))
plt.yscale("log")
fig_name = os.path.join(rdir, f"NumIslandsProton_{base_name}{tag}.png")
plt.savefig(fig_name, dpi=300)

