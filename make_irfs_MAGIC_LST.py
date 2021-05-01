import os
import glob
import time
import shutil
import logging
import operator
import argparse

import numpy as np
from astropy import table
import astropy.units as u
from astropy.io import fits

from pyirf.io.eventdisplay import read_eventdisplay_fits
from pyirf.binning import (
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution

from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
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
from magicctapipe.utils.utils import *
from magicctapipe.utils.tels import *

import matplotlib.pylab as plt
from lstchain.mc import plot_utils

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
    """Make IRFs for MAGIC and/or LST array using pyirf

    Parameters
    ----------
    config_file : str
        configuration file
    """
    print_title("Make IRFs")

    cfg = load_cfg_file(config_file)

    consider_electron = get_key_if_exists(cfg["irfs"], "consider_electron", False)
    tel_ids, tel_ids_LST, tel_ids_MAGIC = check_tel_ids(cfg)

    # --- Check out folder ---
    check_folder(cfg["irfs"]["save_dir"])

    log = logging.getLogger("pyirf")

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

    MIN_GH_CUT_EFFICIENCY = get_key_if_exists(
        cfg["irfs"], "MIN_GH_CUT_EFFICIENCY", GH_CUT_EFFICIENCY_STEP
    )

    INTENSITY_CUT = cfg["irfs"]["INTENSITY_CUT"]
    LEAKAGE1_CUT = cfg["irfs"]["LEAKAGE1_CUT"]

    particles = {
        "gamma": {
            "file": cfg["data_files"]["mc"]["test_sample"]["reco_h5"],
            "max_files": get_key_if_exists(cfg["irfs"], "max_files_gamma", 0),
            "target_spectrum": CRAB_HEGRA,
        },
        "proton": {
            "file": cfg["data_files"]["data"]["test_sample"]["reco_h5"],
            "max_files": get_key_if_exists(cfg["irfs"], "max_files_proton", 0),
            "target_spectrum": IRFDOC_PROTON_SPECTRUM,
        },
    }
    if consider_electron:
        particles["electron"] = {
            "file": cfg["irfs"]["electron_dl2"],
            "max_files": get_key_if_exists(cfg["irfs"], "max_files_electron", 0),
            "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
        }

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyirf").setLevel(logging.DEBUG)

    # Read hdf5 files into pyirf format
    for particle_type, p in particles.items():
        log.info(f"Simulated {particle_type.title()} Events:")
        p["events"], p["simulation_info"] = read_dl2_mcp_to_pyirf_MAGIC_LST_list(
            file_mask=p["file"],
            useless_cols=get_key_if_exists(cfg["irfs"], "useless_cols", []),
            max_files=p["max_files"],
            verbose=get_key_if_exists(cfg["irfs"], "verbose", False),
            eval_mean_events=True,
        )

        # Multiplicity cut
        cut_mult = get_key_if_exists(cfg["irfs"], "cut_on_multiplicity", len(tel_ids))
        p["events"] = p["events"][p["events"]["multiplicity"] >= cut_mult].copy()

        # Applying cuts
        good_ = (p["events"]["intensity"] >= INTENSITY_CUT) & (
            p["events"]["intensity_width_1"] <= LEAKAGE1_CUT
        )
        p["events"] = p["events"][good_]

        l_ = len(p["events"])
        log.info(f"Number of events: {l_}")
        p["events"]["particle_type"] = particle_type

        p["simulated_spectrum"] = PowerLaw.from_simulation(p["simulation_info"], T_OBS)
        p["events"]["weight"] = calculate_event_weights(
            p["events"]["true_energy"], p["target_spectrum"], p["simulated_spectrum"]
        )
        for prefix in ("true", "reco"):
            k = f"{prefix}_source_fov_offset"
            p["events"][k] = calculate_source_fov_offset(p["events"], prefix=prefix)

        # calculate theta / distance between reco and assuemd source position
        # we handle only ON observations here, so the assumed source position
        # is the pointing position
        p["events"]["theta"] = calculate_theta(
            p["events"],
            assumed_source_az=p["events"]["pointing_az"],
            assumed_source_alt=p["events"]["pointing_alt"],
        )
        log.info(p["simulation_info"])
        log.info("")

    gammas = particles["gamma"]["events"]
    # background table composed of both electrons and protons
    if consider_electron:
        background = table.vstack(
            [particles["proton"]["events"], particles["electron"]["events"]]
        )
    else:
        background = table.vstack([particles["proton"]["events"]])

    INITIAL_GH_CUT = np.quantile(gammas["gh_score"], (1 - INITIAL_GH_CUT_EFFICENCY))
    log.info(f"Using fixed G/H cut of {INITIAL_GH_CUT} to calculate theta cuts")

    # event display uses much finer bins for the theta cut than
    # for the sensitivity

    theta_bins = add_overflow_bins(
        create_bins_per_decade(
            e_min=10 ** (cfg["irfs"]["theta_bins_exp_low"]) * u.TeV,
            e_max=10 ** (cfg["irfs"]["theta_bins_exp_high"]) * u.TeV,
            bins_per_decade=cfg["irfs"]["theta_bins_num"],
        )
    )

    # theta cut is 68 percent containmente of the gammas
    # for now with a fixed global, unoptimized score cut
    mask_theta_cuts = gammas["gh_score"] >= INITIAL_GH_CUT
    theta_cuts = calculate_percentile_cut(
        gammas["theta"][mask_theta_cuts],
        gammas["reco_energy"][mask_theta_cuts],
        bins=theta_bins,
        min_value=0.05 * u.deg,
        fill_value=0.32 * u.deg,
        max_value=0.32 * u.deg,
        percentile=68,
    )

    # same bins as event display uses
    sensitivity_bins = add_overflow_bins(
        create_bins_per_decade(
            e_min=10 ** (cfg["irfs"]["sensitivity_bins_exp_low"]) * u.TeV,
            e_max=10 ** (cfg["irfs"]["sensitivity_bins_exp_high"]) * u.TeV,
            bins_per_decade=cfg["irfs"]["sensitivity_bins_num"],
        )
    )

    log.info("Optimizing G/H separation cut for best sensitivity")
    gh_cut_efficiencies = np.arange(
        MIN_GH_CUT_EFFICIENCY,
        MAX_GH_CUT_EFFICIENCY + GH_CUT_EFFICIENCY_STEP / 2,
        GH_CUT_EFFICIENCY_STEP,
    )

    # Fixed the desired efficiencies, find the best gh cuts
    sensitivity_step_2, gh_cuts = optimize_gh_cut(
        gammas,
        background,
        reco_energy_bins=sensitivity_bins,
        gh_cut_efficiencies=gh_cut_efficiencies,
        op=operator.ge,
        theta_cuts=theta_cuts,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )

    # now that we have the optimized gh cuts, we recalculate the theta
    # cut as 68 percent containment on the events surviving these cuts.
    log.info("Recalculating theta cut for optimized GH Cuts")
    for tab in (gammas, background):
        tab["selected_gh"] = evaluate_binned_cut(
            tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
        )

    theta_cuts_opt = calculate_percentile_cut(
        gammas[gammas["selected_gh"]]["theta"],
        gammas[gammas["selected_gh"]]["reco_energy"],
        theta_bins,
        percentile=68,
        fill_value=0.32 * u.deg,
        max_value=0.32 * u.deg,
        min_value=0.05 * u.deg,
    )

    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts_opt, operator.le
    )
    gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]

    # calculate sensitivity
    signal_hist = create_histogram_table(
        gammas[gammas["selected"]], bins=sensitivity_bins
    )
    background_hist = estimate_background(
        background[background["selected_gh"]],
        reco_energy_bins=sensitivity_bins,
        theta_cuts=theta_cuts_opt,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )
    sensitivity = calculate_sensitivity(signal_hist, background_hist, alpha=ALPHA)

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    spectrum = particles["gamma"]["target_spectrum"]
    for s in (sensitivity_step_2, sensitivity):
        s["flux_sensitivity"] = s["relative_sensitivity"] * spectrum(
            s["reco_energy_center"]
        )

    log.info("Calculating IRFs")
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
        fits.BinTableHDU(sensitivity_step_2, name="SENSITIVITY_STEP_2"),
        fits.BinTableHDU(theta_cuts, name="THETA_CUTS"),
        fits.BinTableHDU(theta_cuts_opt, name="THETA_CUTS_OPT"),
        fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
    ]

    masks = {
        "": gammas["selected"],
        "_NO_CUTS": slice(None),
        "_ONLY_GH": gammas["selected_gh"],
        "_ONLY_THETA": gammas["selected_theta"],
    }

    # binnings for the irfs
    true_energy_bins = add_overflow_bins(
        create_bins_per_decade(10 ** -1.9 * u.TeV, 10 ** 2.31 * u.TeV, 10)
    )
    reco_energy_bins = add_overflow_bins(
        create_bins_per_decade(10 ** -1.9 * u.TeV, 10 ** 2.31 * u.TeV, 5)
    )
    fov_offset_bins = [0, 0.5] * u.deg
    source_offset_bins = np.arange(0, 1 + 1e-4, 1e-3) * u.deg
    energy_migration_bins = np.geomspace(0.2, 5, 200)

    for label, mask in masks.items():
        effective_area = effective_area_per_energy(
            gammas[mask],
            particles["gamma"]["simulation_info"],
            true_energy_bins=true_energy_bins,
        )
        hdus.append(
            create_aeff2d_hdu(
                effective_area[..., np.newaxis],  # add one dimension for FOV offset
                true_energy_bins,
                fov_offset_bins,
                extname="EFFECTIVE_AREA" + label,
            )
        )
        edisp = energy_dispersion(
            gammas[mask],
            true_energy_bins=true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=energy_migration_bins,
        )
        hdus.append(
            create_energy_dispersion_hdu(
                edisp,
                true_energy_bins=true_energy_bins,
                migration_bins=energy_migration_bins,
                fov_offset_bins=fov_offset_bins,
                extname="ENERGY_DISPERSION" + label,
            )
        )

    bias_resolution = energy_bias_resolution(
        gammas[gammas["selected"]], reco_energy_bins, energy_type="reco"
    )
    ang_res = angular_resolution(
        gammas[gammas["selected_gh"]], reco_energy_bins, energy_type="reco"
    )
    ang_res = QTable(ang_res)
    psf = psf_table(
        gammas[gammas["selected_gh"]],
        true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        source_offset_bins=source_offset_bins,
    )

    background_rate = background_2d(
        background[background["selected_gh"]],
        reco_energy_bins,
        fov_offset_bins=np.arange(0, 11) * u.deg,
        t_obs=T_OBS,
    )

    hdus.append(
        create_background_2d_hdu(
            background_rate, reco_energy_bins, fov_offset_bins=np.arange(0, 11) * u.deg,
        )
    )
    hdus.append(
        create_psf_table_hdu(
            psf, true_energy_bins, source_offset_bins, fov_offset_bins,
        )
    )
    hdus.append(
        create_rad_max_hdu(
            theta_cuts_opt["cut"][:, np.newaxis], theta_bins, fov_offset_bins
        )
    )
    hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
    hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))

    # --- Store results ---
    log.info("Writing outputfile")

    irfs_subdir = "IRFs_MinEff%s_cut%s/" % (str(MIN_GH_CUT_EFFICIENCY), str(cut_mult))
    irfs_dir = os.path.join(cfg["irfs"]["save_dir"], irfs_subdir)
    check_folder(irfs_dir)

    fits.HDUList(hdus).writeto(
        os.path.join(irfs_dir, "pyirf.fits.gz"), overwrite=True,
    )

    shutil.copyfile(
        kwargs["config_file"],
        os.path.join(irfs_dir, os.path.basename(kwargs["config_file"])),
    )

    return irfs_dir


if __name__ == "__main__":
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()

    irfs_dir = make_irfs_MAGIC_LST(kwargs["config_file"])

    plot_irfs_MAGIC_LST(kwargs["config_file"], irfs_dir)

    print_elapsed_time(start_time, time.time())
