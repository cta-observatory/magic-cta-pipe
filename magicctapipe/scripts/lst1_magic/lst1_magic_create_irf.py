#!/usr/bin/env python
# coding: utf-8

"""
This script processes MC DL2 events and creates the IRFs. Now it can
create only point-like IRFs, i.e., the effective area, energy migration
and background model. If the input data is only gamma MC, it also skips
the creation of the background model.

There are 4 different IRF types which can be specified by the "irf_type"
setting in the configuration file. The "hardware" type is supposed for
the hardware trigger between LST-1 and MAGIC, allowing for the events of
all the telescope combinations. The "software(_only_3tel)" types are
supposed for the software coincidence with LST-mono and MAGIC-stereo,
allowing for only the events triggering both M1 and M2. The "software"
type allows for the events of the any 2 telescope combinations except
the M1 and M2 combination, which are not coincident with LST-1 events.
The "software_only_3tel" type allows only for the events of the three
telescopes combination. The "magic_only" type allows only for the events
of M1 and M2 telescopes combination.

There are 2 types for gammaness and theta cuts, which are "global" and
"dynamic". In case of the dynamic cuts, the optimal cuts satisfying an
efficiency will be calculated in each energy bin specified in the
configuration file.

Usage:
$ python lst1_magic_create_irf.py
--input-file-gamma dl2/dl2_gamma_40deg_90deg.h5
(--input-file-proton dl2/dl2_proton_40deg_90deg.h5)
(--input-file-electron dl2/dl2_electron_40deg_90deg.h5)
(--output-dir irf)
(--config-file config.yaml)
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
    Processes MC DL2 events and creates the IRFs.

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

    Raises
    ------
    RuntimeError
        If the pointing direction does not match between the input MCs
    ValueError
        If the input type of gammaness or theta cut is not known
    """

    config_irf = config["create_irf"]

    quality_cuts = config_irf["quality_cuts"]
    irf_type = config_irf["irf_type"]
    dl2_weight_type = config_irf["dl2_weight_type"]

    logger.info(
        f"\nQuality cuts: {quality_cuts}"
        f"\nIRF type: {irf_type}"
        f"\nDL2 weight type: {dl2_weight_type}"
    )

    # Load the input gamma MC DL2 data file
    logger.info(f"\nInput gamma MC DL2 data file:\n{input_file_gamma}")

    event_table_gamma, pnt_gamma, sim_info_gamma = load_mc_dl2_data_file(
        input_file_gamma, quality_cuts, irf_type, dl2_weight_type
    )

    mean_offset = event_table_gamma["true_source_fov_offset"].to(u.deg).mean().round(1)
    fov_offset_bins = u.Quantity([mean_offset - 0.1 * u.deg, mean_offset + 0.1 * u.deg])

    logger.info(
        f"\nMean FoV offset: {mean_offset}" f"\n--> FoV offset bins: {fov_offset_bins}"
    )

    # Load the background data files
    event_table_bkg = QTable()

    is_proton_mc = input_file_proton is not None
    is_electron_mc = input_file_electron is not None

    if is_proton_mc and (not is_electron_mc):
        logger.info(
            "\nWARNING: Proton MC exists but electron MC does not. "
            "Skipping the creation of the background model..."
        )

    elif is_electron_mc and (not is_proton_mc):
        logger.info(
            "\nWARNING: Electron MC exists but proton MC does not. "
            "Skipping the creation of the background model..."
        )

    is_bkg_mc = np.all([is_proton_mc, is_electron_mc])

    if is_bkg_mc:

        irf_obs_time = u.Quantity(config_irf["irf_obs_time"])
        logger.info(f"\nIRF observation time: {irf_obs_time}")

        # Load the input proton MC DL2 data file
        logger.info(f"\nInput proton MC DL2 data file:\n{input_file_proton}")

        event_table_proton, pnt_proton, sim_info_proton = load_mc_dl2_data_file(
            input_file_proton, quality_cuts, irf_type, dl2_weight_type
        )

        if np.any(pnt_proton != pnt_gamma):
            raise RuntimeError(
                f"Pointing direction of the proton MC {pnt_proton} "
                f"does not match with that of the gamma MC {pnt_gamma}."
            )

        simulated_spectrum_proton = PowerLaw.from_simulation(
            sim_info_proton, irf_obs_time
        )

        event_table_proton["weight"] = calculate_event_weights(
            true_energy=event_table_proton["true_energy"],
            target_spectrum=IRFDOC_PROTON_SPECTRUM,
            simulated_spectrum=simulated_spectrum_proton,
        )

        event_table_bkg = vstack([event_table_bkg, event_table_proton])

        # Load the input electron MC DL2 data file
        logger.info(f"\nInput electron MC DL2 data file:\n{input_file_electron}")

        event_table_electron, pnt_electron, sim_info_electron = load_mc_dl2_data_file(
            input_file_electron, quality_cuts, irf_type, dl2_weight_type
        )

        if np.any(pnt_electron != pnt_gamma):
            raise RuntimeError(
                f"Pointing direction of the electron MC {pnt_electron} "
                f"does not match with that of the gamma MC {pnt_gamma}."
            )

        simulated_spectrum_electron = PowerLaw.from_simulation(
            sim_info_electron, irf_obs_time
        )

        event_table_electron["weight"] = calculate_event_weights(
            true_energy=event_table_electron["true_energy"],
            target_spectrum=IRFDOC_ELECTRON_SPECTRUM,
            simulated_spectrum=simulated_spectrum_electron,
        )

        event_table_bkg = vstack([event_table_bkg, event_table_electron])

    # Prepare for creating IRFs
    eng_bins_start = u.Quantity(config_irf["energy_bins"]["start"])
    eng_bins_stop = u.Quantity(config_irf["energy_bins"]["stop"])
    eng_bins_n_edges = config_irf["energy_bins"]["n_edges"]

    logger.info(
        "\nEnergy bins (log space):"
        f"\n\tstart: {eng_bins_start}"
        f"\n\tstop: {eng_bins_stop}"
        f"\n\tn_edges: {eng_bins_n_edges}"
    )

    energy_bins = u.Quantity(
        value=np.geomspace(
            start=eng_bins_start.to_value(u.TeV).round(3),
            stop=eng_bins_stop.to_value(u.TeV).round(3),
            num=eng_bins_n_edges,
        ),
        unit=u.TeV,
    )

    migra_bins_start = config_irf["migration_bins"]["start"]
    migra_bins_stop = config_irf["migration_bins"]["stop"]
    migra_bins_n_edges = config_irf["migration_bins"]["n_edges"]

    logger.info(
        "\nMigration bins (log space):"
        f"\n\tstart: {migra_bins_start}"
        f"\n\tstop: {migra_bins_stop}"
        f"\n\tn_edges: {migra_bins_n_edges}"
    )

    migration_bins = np.geomspace(migra_bins_start, migra_bins_stop, migra_bins_n_edges)

    if is_bkg_mc:

        bkg_fov_bins_start = u.Quantity(config_irf["bkg_fov_offset_bins"]["start"])
        bkg_fov_bins_stop = u.Quantity(config_irf["bkg_fov_offset_bins"]["stop"])
        bkg_fov_bins_n_edges = config_irf["bkg_fov_offset_bins"]["n_edges"]

        logger.info(
            "\nBackground FoV offset bins (linear space):"
            f"\n\tstart: {bkg_fov_bins_start}"
            f"\n\tstop: {bkg_fov_bins_stop}"
            f"\n\tn_edges: {bkg_fov_bins_n_edges}"
        )

        bkg_fov_offset_bins = u.Quantity(
            value=np.linspace(
                start=bkg_fov_bins_start.to_value(u.deg).round(1),
                stop=bkg_fov_bins_stop.to_value(u.deg).round(1),
                num=bkg_fov_bins_n_edges,
            ),
            unit=u.deg,
        )

    extra_header = {
        "TELESCOP": "CTA-N",
        "INSTRUME": "LST-1_MAGIC",
        "FOVALIGN": "RADEC",
        "PNT_ZD": (pnt_gamma[0].value, "deg"),
        "PNT_AZ": (pnt_gamma[1].value, "deg"),
        "IRF_TYPE": irf_type,
        "DL2_WEIG": dl2_weight_type,
    }

    if quality_cuts is not None:
        extra_header["QUAL_CUT"] = quality_cuts

    irf_hdus = fits.HDUList([fits.PrimaryHDU()])

    # Apply the gammaness cut
    gh_cut_type = config_irf["gammaness"]["cut_type"]

    if gh_cut_type == "global":

        gh_cut_value = config_irf["gammaness"]["global_cut_value"]
        logger.info("\nGlobal gammaness cut:" f"\n\tcut_value: {gh_cut_value}")

        gh_cut_config = f"gh_glob{gh_cut_value}"
        extra_header["GH_CUT"] = gh_cut_value

        # Apply the global gammaness cut
        logger.info("\nApplying the global gammaness cut...")

        mask_gh_gamma = event_table_gamma["gammaness"] > gh_cut_value
        event_table_gamma = event_table_gamma[mask_gh_gamma]

        if is_bkg_mc:
            mask_gh_bkg = event_table_bkg["gammaness"] > gh_cut_value
            event_table_bkg = event_table_bkg[mask_gh_bkg]

    elif gh_cut_type == "dynamic":

        gh_efficiency = config_irf["gammaness"]["efficiency"]
        gh_cut_min = config_irf["gammaness"]["min_cut"]
        gh_cut_max = config_irf["gammaness"]["max_cut"]

        logger.info(
            "\nDynamic gammaness cuts:"
            f"\n\tefficiency: {gh_efficiency}"
            f"\n\tmin_cut: {gh_cut_min}"
            f"\n\tmax_cut: {gh_cut_max}"
        )

        gh_cut_config = f"gh_dyn{gh_efficiency}"

        extra_header["GH_EFF"] = gh_efficiency
        extra_header["GH_MIN"] = gh_cut_min
        extra_header["GH_MAX"] = gh_cut_max

        # Apply the gammaness cuts satisfying the efficiency
        gh_percentile = 100 * (1 - gh_efficiency)

        gh_cut_table = calculate_percentile_cut(
            values=event_table_gamma["gammaness"],
            bin_values=event_table_gamma["reco_energy"],
            bins=energy_bins,
            fill_value=gh_cut_min,
            percentile=gh_percentile,
            min_value=gh_cut_min,
            max_value=gh_cut_max,
        )

        logger.info(
            f"\nGammaness-cut table:\n\n{gh_cut_table}"
            "\n\nApplying the dynamic gammaness cuts..."
        )

        mask_gh_gamma = evaluate_binned_cut(
            values=event_table_gamma["gammaness"],
            bin_values=event_table_gamma["reco_energy"],
            cut_table=gh_cut_table,
            op=operator.ge,
        )

        event_table_gamma = event_table_gamma[mask_gh_gamma]

        if is_bkg_mc:
            mask_gh_bkg = evaluate_binned_cut(
                values=event_table_bkg["gammaness"],
                bin_values=event_table_bkg["reco_energy"],
                cut_table=gh_cut_table,
                op=operator.ge,
            )

            event_table_bkg = event_table_bkg[mask_gh_bkg]

        # Create a gammaness-cut HDU
        logger.info("Creating a gammaness-cut HDU...")

        hdu_gh_cuts = create_gh_cuts_hdu(
            gh_cuts=gh_cut_table["cut"][:, np.newaxis],
            reco_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            **extra_header,
        )

        irf_hdus.append(hdu_gh_cuts)

    else:
        raise ValueError(f"Unknown gammaness-cut type '{gh_cut_type}'.")

    # Apply the theta cut
    theta_cut_type = config_irf["theta"]["cut_type"]

    if theta_cut_type == "global":

        theta_cut_value = u.Quantity(config_irf["theta"]["global_cut_value"])
        logger.info("\nGlobal theta cut:" f"\n\tcut_value: {theta_cut_value}")

        theta_cut_config = f"theta_glob{theta_cut_value.to_value(u.deg)}deg"
        extra_header["RAD_MAX"] = (theta_cut_value.to_value(u.deg), "deg")

        # Apply the global theta cut
        logger.info("\nApplying the global theta cut...")

        mask_theta = event_table_gamma["theta"] < theta_cut_value
        event_table_gamma = event_table_gamma[mask_theta]

    elif theta_cut_type == "dynamic":

        theta_efficiency = config_irf["theta"]["efficiency"]
        theta_cut_min = u.Quantity(config_irf["theta"]["min_cut"])
        theta_cut_max = u.Quantity(config_irf["theta"]["max_cut"])

        logger.info(
            "\nDynamic theta cuts:"
            f"\n\tefficiency: {theta_efficiency}"
            f"\n\tmin_cut: {theta_cut_min}"
            f"\n\tmax_cut: {theta_cut_max}"
        )

        theta_cut_config = f"theta_dyn{theta_efficiency}"

        extra_header["TH_EFF"] = theta_efficiency
        extra_header["TH_MIN"] = (theta_cut_min.to_value(u.deg), "deg")
        extra_header["TH_MAX"] = (theta_cut_max.to_value(u.deg), "deg")

        # Apply the theta cuts satisfying the efficiency
        theta_percentile = 100 * theta_efficiency

        theta_cut_table = calculate_percentile_cut(
            values=event_table_gamma["theta"],
            bin_values=event_table_gamma["reco_energy"],
            bins=energy_bins,
            fill_value=theta_cut_max,
            percentile=theta_percentile,
            min_value=theta_cut_min,
            max_value=theta_cut_max,
        )

        logger.info(
            f"\nTheta-cut table:\n\n{theta_cut_table}"
            "\n\nApplying the dynamic theta cuts..."
        )

        mask_theta = evaluate_binned_cut(
            values=event_table_gamma["theta"],
            bin_values=event_table_gamma["reco_energy"],
            cut_table=theta_cut_table,
            op=operator.le,
        )

        event_table_gamma = event_table_gamma[mask_theta]

        # Create a rad-max HDU
        logger.info("Creating a rad-max HDU...")

        hdu_rad_max = create_rad_max_hdu(
            rad_max=theta_cut_table["cut"][:, np.newaxis],
            reco_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            point_like=True,
            extname="RAD_MAX",
            **extra_header,
        )

        irf_hdus.append(hdu_rad_max)

    else:
        raise ValueError(f"Unknown theta-cut type '{theta_cut_type}'.")

    # Create an effective-are HDU
    logger.info("\nCreating an effective-area HDU...")

    with np.errstate(invalid="ignore", divide="ignore"):

        aeff = effective_area_per_energy(
            selected_events=event_table_gamma,
            simulation_info=sim_info_gamma,
            true_energy_bins=energy_bins,
        )

        aeff_hdu = create_aeff2d_hdu(
            effective_area=aeff[:, np.newaxis],
            true_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            point_like=True,
            extname="EFFECTIVE AREA",
            **extra_header,
        )

    irf_hdus.append(aeff_hdu)

    # Create an energy-dispersion HDU
    logger.info("Creating an energy-dispersion HDU...")

    edisp = energy_dispersion(
        selected_events=event_table_gamma,
        true_energy_bins=energy_bins,
        fov_offset_bins=fov_offset_bins,
        migration_bins=migration_bins,
    )

    edisp_hdu = create_energy_dispersion_hdu(
        energy_dispersion=edisp,
        true_energy_bins=energy_bins,
        migration_bins=migration_bins,
        fov_offset_bins=fov_offset_bins,
        point_like=True,
        extname="ENERGY DISPERSION",
        **extra_header,
    )

    irf_hdus.append(edisp_hdu)

    # Create a background HDU
    if is_bkg_mc:
        logger.info("Creating a background HDU...")

        bkg = background_2d(
            events=event_table_bkg,
            reco_energy_bins=energy_bins,
            fov_offset_bins=bkg_fov_offset_bins,
            t_obs=irf_obs_time,
        )

        bkg_hdu = create_background_2d_hdu(
            background_2d=bkg.T,
            reco_energy_bins=energy_bins,
            fov_offset_bins=bkg_fov_offset_bins,
            extname="BACKGROUND",
            **extra_header,
        )

        irf_hdus.append(bkg_hdu)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_file = (
        f"{output_dir}/irf_zd_{pnt_gamma[0].value}deg_az_{pnt_gamma[1].value}deg"
        f"_{irf_type}_{gh_cut_config}_{theta_cut_config}.fits.gz"
    )

    irf_hdus.writeto(output_file, overwrite=True)

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
        help="Path to a configuration file.",
    )

    args = parser.parse_args()

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
