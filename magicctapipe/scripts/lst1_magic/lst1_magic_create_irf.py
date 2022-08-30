#!/usr/bin/env python
# coding: utf-8

"""
This script creates the IRFs with input MC DL2 data. Now it can create
only point-like IRFs, i.e., the effective area, energy migration and
background model. If the input data is only gamma MC, it also skips the
creation of the background model.

There are 4 different IRF types which can be specified by the "irf_type"
setting in the configuration file. The "hardware" type is supposed for
the hardware trigger condition, allowing for the events of all the
telescope combinations. The "software(_only_3tel)" types are supposed
for the software coincidence condition with LST-mono and MAGIC-stereo,
allowing for only the events triggering both M1 and M2. The "software"
allows for the events of the any 2 telescope combinations except the M1
and M2 combination, and the "software_only_3tel" allows only for the
events of the three telescopes combination. The "magic_only" allows only
for the events of M1 and M2 telescopes combination.

There are 2 types of gammaness/theta cuts allowed, which are "global" or
"dynamic". In case of the dynamic cuts, the optimal cuts satisfying an
efficiency will be computed in each energy bin specified in the
configuration file.

Usage:
$ python lst1_magic_create_irf.py
--input-file-gamma ./dl2/dl2_gamma_40deg_90deg_off0.4deg_LST-1_MAGIC.h5
--output-dir ./irf
--config-file ./config.yaml
(--input-file-proton)
(--input-file-electron)
"""

import argparse
import logging
import operator
import time
from pathlib import Path

import numpy as np
import yaml
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable, vstack
from magicctapipe.io import create_gh_cuts_hdu, load_mc_dl2_data_file
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
)
from pyirf.irf import background_2d, effective_area_per_energy, energy_dispersion
from pyirf.spectral import (
    IRFDOC_ELECTRON_SPECTRUM,
    IRFDOC_PROTON_SPECTRUM,
    PowerLaw,
    calculate_event_weights,
)

__all__ = ["create_irf"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def create_irf(
    input_file_gamma, input_file_proton, input_file_electron, output_dir, config
):
    """
    Creates IRF hdus with input gamma and background MC DL2 data.

    Parameters
    ----------
    input_file_gamma: str
        Path to an input gamma MC DL2 data file
    input_file_proton: str
        Path to an input proton MC DL2 data file
    input_file_electron: str
        Path to an input electron MC DL2 data file
    output_dir: str
        Path to a directory where to save an output IRF file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_irf = config["create_irf"]

    quality_cuts = config_irf["quality_cuts"]
    irf_type = config_irf["irf_type"]
    dl2_weight = config_irf["dl2_weight"]
    irf_obs_time = u.Quantity(config_irf.get("irf_obs_time", "50 h"))

    # Load the input gamma MC data file
    logger.info(f"\nInput gamma MC DL2 data file:\n{input_file_gamma}")

    table_gamma, pnt_gamma, sim_info_gamma = load_mc_dl2_data_file(
        input_file_gamma, quality_cuts, irf_type, dl2_weight
    )

    # Check the FoV offset
    mean_offset = np.round(table_gamma["true_source_fov_offset"].to(u.deg).mean(), 1)
    fov_offset_bins = u.Quantity([mean_offset - 0.1 * u.deg, mean_offset + 0.1 * u.deg])

    logger.info(
        f"\nMean FoV offset: {mean_offset}\n--> FoV offset bins: {fov_offset_bins}"
    )

    # Load the background data files
    table_bkg = QTable()

    if input_file_proton is not None:

        # Load the input proton MC file
        logger.info(f"\nInput proton MC DL2 data file:\n{input_file_proton}")

        table_proton, pnt_proton, sim_info_proton = load_mc_dl2_data_file(
            input_file_proton, quality_cuts, irf_type, dl2_weight
        )

        if np.any(pnt_proton != pnt_gamma):
            raise ValueError(
                f"Proton MC pointing direction {pnt_proton} deg "
                f"do not match with that of gamma MC {pnt_gamma} deg."
            )

        simulated_spectrum_proton = PowerLaw.from_simulation(
            sim_info_proton, irf_obs_time
        )

        table_proton["weight"] = calculate_event_weights(
            true_energy=table_proton["true_energy"],
            target_spectrum=IRFDOC_PROTON_SPECTRUM,
            simulated_spectrum=simulated_spectrum_proton,
        )

        table_bkg = vstack([table_bkg, table_proton])

    if input_file_electron is not None:

        # Load the input electron MC file
        logger.info(f"\nInput electron MC DL2 data file:\n{input_file_electron}")

        table_electron, pnt_electron, sim_info_electron = load_mc_dl2_data_file(
            input_file_electron, quality_cuts, irf_type, dl2_weight
        )

        if np.any(pnt_electron != pnt_gamma):
            raise ValueError(
                f"Electron MC pointing direction {pnt_electron} deg "
                f"do not match with that of gamma MC {pnt_gamma} deg."
            )

        simulated_spectrum_electron = PowerLaw.from_simulation(
            sim_info_electron, irf_obs_time
        )

        table_electron["weight"] = calculate_event_weights(
            true_energy=table_electron["true_energy"],
            target_spectrum=IRFDOC_ELECTRON_SPECTRUM,
            simulated_spectrum=simulated_spectrum_electron,
        )

        table_bkg = vstack([table_bkg, table_electron])

    bkg_exists = len(table_bkg) > 0

    # Prepare for creating IRFs
    config_eng_bins = config_irf["energy_bins"]

    logger.info("\nEnergy bins (log space):")
    for key, value in config_eng_bins.items():
        logger.info(f"\t{key}: {value}")

    energy_bins = np.geomspace(
        start=u.Quantity(config_eng_bins["start"]).to_value(u.TeV).round(3),
        stop=u.Quantity(config_eng_bins["stop"]).to_value(u.TeV).round(3),
        num=config_eng_bins["n_edges"],
    ) * u.TeV

    config_migra_bins = config_irf["migration_bins"]

    logger.info("\nMigration bins (log space):")
    for key, value in config_migra_bins.items():
        logger.info(f"\t{key}: {value}")

    migration_bins = np.geomspace(
        start=config_migra_bins["start"],
        stop=config_migra_bins["stop"],
        num=config_migra_bins["n_edges"],
    )

    if bkg_exists:
        config_bkg_fov_bins = config_irf["bkg_fov_offset_bins"]

        logger.info("\nBackground FoV offset bins (linear space):")
        for key, value in config_bkg_fov_bins.items():
            logger.info(f"\t{key}: {value}")

        bkg_fov_offset_bins = np.linspace(
            start=u.Quantity(config_bkg_fov_bins["start"]).to_value(u.deg),
            stop=u.Quantity(config_bkg_fov_bins["stop"]).to_value(u.deg),
            num=config_bkg_fov_bins["n_edges"],
        ) * u.deg

    extra_header = {
        "TELESCOP": "CTA-N",
        "INSTRUME": "LST-1_MAGIC",
        "FOVALIGN": "RADEC",
        "PNT_ZD": (pnt_gamma[0], "deg"),
        "PNT_AZ": (pnt_gamma[1], "deg"),
        "IRF_TYPE": irf_type,
        "DL2_WEIG": dl2_weight,
    }

    if quality_cuts is not None:
        extra_header["QUAL_CUT"] = quality_cuts

    hdus_irf = fits.HDUList([fits.PrimaryHDU()])

    # Apply the gammaness cut
    gam_cut_type = config_irf["gammaness"]["cut_type"]

    if gam_cut_type == "global":
        logger.info("\nApplying the global gammaness cut:")

        global_gam_cut = config_irf["gammaness"]["global_cut_value"]
        logger.info(f"\tGlobal cut value: {global_gam_cut}")

        # Apply the global gammaness cut
        table_gamma = table_gamma[table_gamma["gammaness"] > global_gam_cut]

        if bkg_exists:
            table_bkg = table_bkg[table_bkg["gammaness"] > global_gam_cut]

        gam_cut_config = f"gam_glob{global_gam_cut}"
        extra_header["GH_CUT"] = global_gam_cut

    elif gam_cut_type == "dynamic":
        logger.info("\nApplying the dynamic gammaness cuts:")

        gh_efficiency = config_irf["gammaness"]["efficiency"]
        gh_cut_min = config_irf["gammaness"]["min_cut"]
        gh_cut_max = config_irf["gammaness"]["max_cut"]

        logger.info(f"\tEfficiency: {gh_efficiency}")
        logger.info(f"\tMinimum cut allowed: {gh_cut_min}")
        logger.info(f"\tMaximum cut allowed: {gh_cut_max}")

        gam_cut_config = f"gam_dyn{gh_efficiency}"

        extra_header["GH_EFF"] = (gh_efficiency, "gh efficiency")
        extra_header["GH_MIN"] = gh_cut_min
        extra_header["GH_MAX"] = gh_cut_max

        # Compute the cuts satisfying the efficiency
        gh_percentile = 100 * (1 - gh_efficiency)

        cut_table_gh = calculate_percentile_cut(
            values=table_gamma["gammaness"],
            bin_values=table_gamma["reco_energy"],
            bins=energy_bins,
            fill_value=gh_cut_min,
            percentile=gh_percentile,
            min_value=gh_cut_min,
            max_value=gh_cut_max,
        )

        logger.info(f"\nGammaness cut table:\n{cut_table_gh}")

        # Apply the cuts to the data
        mask_gh_gamma = evaluate_binned_cut(
            values=table_gamma["gammaness"],
            bin_values=table_gamma["reco_energy"],
            cut_table=cut_table_gh,
            op=operator.ge,
        )

        table_gamma = table_gamma[mask_gh_gamma]

        if bkg_exists:
            mask_gh_bkg = evaluate_binned_cut(
                values=table_bkg["gammaness"],
                bin_values=table_bkg["reco_energy"],
                cut_table=cut_table_gh,
                op=operator.ge,
            )

            table_bkg = table_bkg[mask_gh_bkg]

        # Create a gammaness-cut HDU
        logger.info("\nCreating a gammaness-cut HDU...")

        hdu_gh_cuts = create_gh_cuts_hdu(
            gh_cuts=cut_table_gh["cut"][:, np.newaxis],
            reco_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            extname="GH_CUTS",
            **extra_header,
        )

        hdus_irf.append(hdu_gh_cuts)

    else:
        raise ValueError(f"Unknown gammaness cut type '{gam_cut_type}'.")

    # Apply the theta cut
    theta_cut_type = config_irf["theta"]["cut_type"]

    if theta_cut_type == "global":
        logger.info("\nApplying the global theta cut:")

        global_theta_cut = u.Quantity(config_irf["theta"]["global_cut_value"])
        logger.info(f"\tGlobal cut value: {global_theta_cut}")

        theta_cut_config = f"theta_glob{global_theta_cut.value}"
        extra_header["RAD_MAX"] = (global_theta_cut.to_value(u.deg), "deg")

        # Apply the global theta cut
        table_gamma = table_gamma[table_gamma["theta"] < global_theta_cut]

    elif theta_cut_type == "dynamic":
        logger.info("\nApplying the dynamic theta cuts:")

        theta_efficiency = config_irf["theta"]["efficiency"]
        theta_cut_min = u.Quantity(config_irf["theta"]["min_cut"])
        theta_cut_max = u.Quantity(config_irf["theta"]["max_cut"])

        logger.info(f"\tEfficiency: {theta_efficiency}")
        logger.info(f"\tMinimum cut allowed: {theta_cut_min}")
        logger.info(f"\tMaximum cut allowed: {theta_cut_max}")

        theta_cut_config = f"theta_dyn{theta_efficiency}"

        extra_header["TH_EFF"] = (theta_efficiency, "gamma efficiency")
        extra_header["TH_MIN"] = (theta_cut_min.to_value(u.deg), "deg")
        extra_header["TH_MAX"] = (theta_cut_max.to_value(u.deg), "deg")

        # Compute the cuts satisfying the efficiency
        theta_percentile = 100 * theta_efficiency

        cut_table_theta = calculate_percentile_cut(
            values=table_gamma["theta"],
            bin_values=table_gamma["reco_energy"],
            bins=energy_bins,
            fill_value=theta_cut_max,
            percentile=theta_percentile,
            min_value=theta_cut_min,
            max_value=theta_cut_max,
        )

        logger.info(f"\nTheta cut table:\n{cut_table_theta}")

        # Apply the cuts to the data
        mask_theta = evaluate_binned_cut(
            values=table_gamma["theta"],
            bin_values=table_gamma["reco_energy"],
            cut_table=cut_table_theta,
            op=operator.le,
        )

        table_gamma = table_gamma[mask_theta]

        # Create a rad-max HDU
        logger.info("\nCreating a rad-max HDU...")

        hdu_rad_max = create_rad_max_hdu(
            rad_max=cut_table_theta["cut"][:, np.newaxis],
            reco_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            point_like=True,
            extname="RAD_MAX",
            **extra_header,
        )

        hdus_irf.append(hdu_rad_max)

    else:
        raise ValueError(f"Unknown theta cut type '{theta_cut_type}'.")

    # Create the IRFs
    logger.info("\nCreating an effective area HDU...")

    with np.errstate(invalid="ignore", divide="ignore"):

        aeff = effective_area_per_energy(
            selected_events=table_gamma,
            simulation_info=sim_info_gamma,
            true_energy_bins=energy_bins,
        )

        hdu_aeff = create_aeff2d_hdu(
            effective_area=aeff[:, np.newaxis],
            true_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            point_like=True,
            extname="EFFECTIVE AREA",
            **extra_header,
        )

    hdus_irf.append(hdu_aeff)

    logger.info("Creating an energy dispersion HDU...")

    edisp = energy_dispersion(
        selected_events=table_gamma,
        true_energy_bins=energy_bins,
        fov_offset_bins=fov_offset_bins,
        migration_bins=migration_bins,
    )

    hdu_edisp = create_energy_dispersion_hdu(
        energy_dispersion=edisp,
        true_energy_bins=energy_bins,
        migration_bins=migration_bins,
        fov_offset_bins=fov_offset_bins,
        point_like=True,
        extname="ENERGY DISPERSION",
        **extra_header,
    )

    hdus_irf.append(hdu_edisp)

    if bkg_exists:
        logger.info("Creating a background HDU...")

        bkg = background_2d(
            events=table_bkg,
            reco_energy_bins=energy_bins,
            fov_offset_bins=bkg_fov_offset_bins,
            t_obs=irf_obs_time,
        )

        hdu_bkg = create_background_2d_hdu(
            background_2d=bkg.T,
            reco_energy_bins=energy_bins,
            fov_offset_bins=bkg_fov_offset_bins,
            extname="BACKGROUND",
            **extra_header,
        )

        hdus_irf.append(hdu_bkg)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_file = (
        f"{output_dir}/irf_zd_{pnt_gamma[0]}deg_az_{pnt_gamma[1]}deg"
        f"_{irf_type}_{gam_cut_config}_{theta_cut_config}.fits.gz"
    )

    hdus_irf.writeto(output_file, overwrite=True)

    logger.info(f"\nOutput file:\n{output_file}")


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file-gamma",
        "-g",
        dest="input_file_gamma",
        type=str,
        required=True,
        help="Path to an input gamma MC DL2 data file.",
    )

    parser.add_argument(
        "--input-file-proton",
        "-p",
        dest="input_file_proton",
        type=str,
        help="Path to an input proton MC DL2 data file.",
    )

    parser.add_argument(
        "--input-file-electron",
        "-e",
        dest="input_file_electron",
        type=str,
        help="Path to an input electron MC DL2 data file.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output IRF file.",
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

    print(args)

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Create the IRFs
    create_irf(
        input_file_gamma=args.input_file_gamma,
        input_file_proton=args.input_file_proton,
        input_file_electron=args.input_file_electron,
        output_dir=args.output_dir,
        config=config,
    )

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
