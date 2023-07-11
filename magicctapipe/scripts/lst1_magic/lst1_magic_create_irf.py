#!/usr/bin/env python
# coding: utf-8

"""
This script processes MC DL2 events and creates the IRFs. It can create
two different IRF types, "POINT-LIKE" or "FULL-ENCLOSURE". The effective
area and energy dispersion HDUs are created in case of the "POINT_LIKE"
IRFs, and in addition the PSF table and background HDUs in case of the
"FULL-ENCLOSURE" IRFs.

When the input gamma MC is point-like or ring-wobble data, it creates
one FoV offset bin around the true offset, regardless of the settings in
the configuration file, and creates only the "POINT-LIKE" IRFs. In case
of diffuse data, it creates FoV offset bins based on the configuration
file and creates the "FULL-ENCLOSURE" if the number of FoV offset bins
is more than one. In case the number is one, it creates the "POINT-LIKE"
IRFs, which allows us to perform the 1D spectral analysis even if only
diffuse data is available for test MCs.

There are four different event types with which the IRFs are created.
The "hardware" type stands for the hardware trigger between LST
and MAGIC, allowing for the events of all the telescope combinations.
The "software_3tels_or_more" type stands for the software event
coincidence with any combination of 3 or more telescopes (e.g. LST2, LST3, and
MAGIC-I observations). The "software_6_tel" type allows for the events of any 
2,3,4,5 or 6 telescope combinations (except the combination MAGIC-I + MAGIC-II).
The "software" type is similar to "software_6_tel", but requires that
we the events are tagged as "stereo_magic". The "magic_only" type allows for
only the events of the MAGIC-stereo combination.

There are two types of gammaness and theta cuts, "global" and "dynamic".
In case of the dynamic cuts, the optimal cut satisfying a given
efficiency will be calculated for every energy bin.

Usage:
$ python lst1_magic_create_irf.py
--input-file-gamma dl2/dl2_gamma_40deg_90deg.h5
(--input-file-proton dl2/dl2_proton_40deg_90deg.h5)
(--input-file-electron dl2/dl2_electron_40deg_90deg.h5)
(--output-dir irf)
(--config-file config.yaml)

Broader usage:
This script is called automatically from the script "IRF.py".
If you want to analyse a target, this is the way to go. See this other script for more details.
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
from magicctapipe.io import create_gh_cuts_hdu, format_object, load_mc_dl2_data_file
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
    create_rad_max_hdu,
)
from pyirf.irf import (
    background_2d,
    effective_area_per_energy,
    effective_area_per_energy_and_fov,
    energy_dispersion,
    psf_table,
)
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
    event_type = config_irf["event_type"]
    weight_type_dl2 = config_irf["weight_type_dl2"]

    logger.info(f"\nQuality cuts: {quality_cuts}")
    logger.info(f"Event type: {event_type}")
    logger.info(f"DL2 weight type: {weight_type_dl2}")

    # Load the input gamma MC DL2 data file
    logger.info(f"\nInput gamma MC DL2 data file: {input_file_gamma}")

    event_table_gamma, pnt_gamma, sim_info_gamma = load_mc_dl2_data_file(
        config, input_file_gamma, quality_cuts, event_type, weight_type_dl2
    )

    is_diffuse_mc = sim_info_gamma.viewcone.to_value("deg") > 0
    logger.info(f"\nIs diffuse MC: {is_diffuse_mc}")

    if is_diffuse_mc:
        # Create FoV offset bins based on the configuration
        config_fov_bins = config_irf["fov_offset_bins"]

        logger.info("\nFov offset bins (linear scale):")
        logger.info(format_object(config_fov_bins))

        fov_bins_start = u.Quantity(config_fov_bins["start"])
        fov_bins_stop = u.Quantity(config_fov_bins["stop"])

        fov_offset_bins = u.deg * np.linspace(
            start=fov_bins_start.to_value("deg").round(1),
            stop=fov_bins_stop.to_value("deg").round(1),
            num=config_fov_bins["n_edges"],
        )

    else:
        # Create one FoV offset bin around the true offset
        true_fov_offset = event_table_gamma["true_source_fov_offset"].to("deg")
        mean_true_fov_offset = true_fov_offset.mean().round(1)

        fov_offset_bins = mean_true_fov_offset + [-0.1, 0.1] * u.deg

        logger.info(f"\nMean true FoV offset: {mean_true_fov_offset}")
        logger.info(f"--> FoV offset bin: {fov_offset_bins}")

    # Here we decide the IRF type based on the number of FoV offset
    # bins - "POINT-LIKE" in case of 1 bin and "FULL-ENCLOSURE" in the
    # other cases. It allows for creating the "POINT-LIKE" IRFs with
    # diffuse gamma MCs by selecting a given FoV offset region.

    n_fov_offset_bins = len(fov_offset_bins) - 1
    is_point_like = n_fov_offset_bins == 1

    if is_point_like:
        logger.info("\nIRF type: POINT-LIKE")
    else:
        logger.info("\nIRF type: FULL-ENCLOSURE")

    # Check the existence of background MC data
    is_proton_mc = input_file_proton is not None
    is_electron_mc = input_file_electron is not None

    logger.info(f"\nIs proton MC: {is_proton_mc}")
    logger.info(f"Is electron MC: {is_electron_mc}")

    is_bkg_mc = all([is_proton_mc, is_electron_mc])

    if is_point_like and is_bkg_mc:
        logger.warning(
            "\nWARNING: Will skip the creation of a background model, "
            "since it is not needed for the 'POINT-LIKE' IRFs."
        )

    if (not is_point_like) and (not is_bkg_mc):
        logger.warning(
            "\nWARNING: Will skip the creation of a background model, "
            "since both or either of background MCs are missing."
        )

    event_table_bkg = QTable()

    if not is_point_like and is_bkg_mc:
        # Load the input proton MC DL2 data file
        logger.info(f"\nInput proton MC DL2 data file: {input_file_proton}")

        event_table_proton, pnt_proton, sim_info_proton = load_mc_dl2_data_file(
            config, input_file_proton, quality_cuts, event_type, weight_type_dl2
        )

        if any(pnt_proton != pnt_gamma):
            raise RuntimeError(
                f"Pointing direction of the proton MC {pnt_proton.tolist()} deg "
                f"does not match with that of the gamma MC {pnt_gamma.tolist()} deg."
            )

        # Load the input electron MC DL2 data file
        logger.info(f"\nInput electron MC DL2 data file: {input_file_electron}")

        event_table_electron, pnt_electron, sim_info_electron = load_mc_dl2_data_file(
            config, input_file_electron, quality_cuts, event_type, weight_type_dl2
        )

        if any(pnt_electron != pnt_gamma):
            raise RuntimeError(
                f"Pointing direction of the electron MC {pnt_electron.tolist()} deg "
                f"does not match with that of the gamma MC {pnt_gamma.tolist()} deg."
            )

        # Calculate event weights
        obs_time = config_irf["obs_time_irf"]
        logger.info(f"\nIRF observation time: {obs_time}")

        obs_time = u.Quantity(obs_time)

        sim_spectrum_proton = PowerLaw.from_simulation(sim_info_proton, obs_time)
        sim_spectrum_electron = PowerLaw.from_simulation(sim_info_electron, obs_time)

        event_table_proton["weight"] = calculate_event_weights(
            true_energy=event_table_proton["true_energy"],
            target_spectrum=IRFDOC_PROTON_SPECTRUM,
            simulated_spectrum=sim_spectrum_proton,
        )

        event_table_electron["weight"] = calculate_event_weights(
            true_energy=event_table_electron["true_energy"],
            target_spectrum=IRFDOC_ELECTRON_SPECTRUM,
            simulated_spectrum=sim_spectrum_electron,
        )

        # Combine the background MCs
        event_table_bkg = vstack([event_table_proton, event_table_electron])

    # Prepare for creating IRFs
    config_eng_bins = config_irf["energy_bins"]
    config_mig_bins = config_irf["migration_bins"]

    logger.info("\nEnergy bins (log space):")
    logger.info(format_object(config_eng_bins))

    eng_bins_start = u.Quantity(config_eng_bins["start"])
    eng_bins_stop = u.Quantity(config_eng_bins["stop"])

    energy_bins = u.TeV * np.geomspace(
        start=eng_bins_start.to_value("TeV").round(3),
        stop=eng_bins_stop.to_value("TeV").round(3),
        num=config_eng_bins["n_edges"],
    )

    logger.info("\nMigration bins (log space):")
    logger.info(format_object(config_mig_bins))

    migration_bins = np.geomspace(
        config_mig_bins["start"], config_mig_bins["stop"], config_mig_bins["n_edges"]
    )

    if not is_point_like:
        config_src_bins = config_irf["source_offset_bins"]

        logger.info("\nSource offset bins (linear space):")
        logger.info(format_object(config_src_bins))

        src_bins_start = u.Quantity(config_src_bins["start"])
        src_bins_stop = u.Quantity(config_src_bins["stop"])

        source_offset_bins = u.deg * np.linspace(
            start=src_bins_start.to_value("deg").round(1),
            stop=src_bins_stop.to_value("deg").round(1),
            num=config_src_bins["n_edges"],
        )

        if is_bkg_mc:
            config_bkg_bins = config_irf["bkg_fov_offset_bins"]

            logger.info("\nBackground FoV offset bins (linear space):")
            logger.info(format_object(config_bkg_bins))

            bkg_bins_start = u.Quantity(config_bkg_bins["start"])
            bkg_bins_stop = u.Quantity(config_bkg_bins["stop"])

            bkg_fov_offset_bins = u.deg * np.linspace(
                start=bkg_bins_start.to_value("deg").round(1),
                stop=bkg_bins_stop.to_value("deg").round(1),
                num=config_bkg_bins["n_edges"],
            )

    extra_header = {
        "TELESCOP": "CTA-N",
        "INSTRUME": "LST-1_MAGIC",
        "FOVALIGN": "RADEC",
        "PNT_ZD": (pnt_gamma[0], "deg"),
        "PNT_AZ": (pnt_gamma[1], "deg"),
        "EVT_TYPE": event_type,
        "DL2_WEIG": weight_type_dl2,
    }

    if quality_cuts is not None:
        extra_header["QUAL_CUT"] = quality_cuts

    if is_bkg_mc:
        extra_header["IRF_OBST"] = (obs_time.to_value("h"), "h")

    irf_hdus = fits.HDUList([fits.PrimaryHDU()])

    # Apply the gammaness cut
    config_gh_cuts = config_irf["gammaness"]
    cut_type_gh = config_gh_cuts.pop("cut_type")

    if cut_type_gh == "global":
        cut_value_gh = config_gh_cuts["global_cut_value"]
        logger.info(f"\nGlobal gammaness cut: {cut_value_gh}")

        extra_header["GH_CUT"] = cut_value_gh
        output_suffix = f"gh_glob{cut_value_gh}"

        # Apply the global gammaness cut
        mask_gh = event_table_gamma["gammaness"] > cut_value_gh
        event_table_gamma = event_table_gamma[mask_gh]

        if is_bkg_mc:
            mask_gh = event_table_bkg["gammaness"] > cut_value_gh
            event_table_bkg = event_table_bkg[mask_gh]

    elif cut_type_gh == "dynamic":
        config_gh_cuts.pop("global_cut_value", None)

        logger.info("\nDynamic gammaness cuts:")
        logger.info(format_object(config_gh_cuts))

        gh_efficiency = config_gh_cuts["efficiency"]
        gh_cut_min = config_gh_cuts["min_cut"]
        gh_cut_max = config_gh_cuts["max_cut"]

        extra_header["GH_EFF"] = gh_efficiency
        extra_header["GH_MIN"] = gh_cut_min
        extra_header["GH_MAX"] = gh_cut_max

        output_suffix = f"gh_dyn{gh_efficiency}"

        # Calculate dynamic gammaness cuts
        gh_percentile = 100 * (1 - gh_efficiency)

        cut_table_gh = calculate_percentile_cut(
            values=event_table_gamma["gammaness"],
            bin_values=event_table_gamma["reco_energy"],
            bins=energy_bins,
            fill_value=gh_cut_min,
            percentile=gh_percentile,
            min_value=gh_cut_min,
            max_value=gh_cut_max,
        )

        logger.info(f"\nGammaness-cut table:\n\n{cut_table_gh}")

        # Apply the dynamic gammaness cuts
        mask_gh = evaluate_binned_cut(
            values=event_table_gamma["gammaness"],
            bin_values=event_table_gamma["reco_energy"],
            cut_table=cut_table_gh,
            op=operator.ge,
        )

        event_table_gamma = event_table_gamma[mask_gh]

        if is_bkg_mc:
            mask_gh = evaluate_binned_cut(
                values=event_table_bkg["gammaness"],
                bin_values=event_table_bkg["reco_energy"],
                cut_table=cut_table_gh,
                op=operator.ge,
            )

            event_table_bkg = event_table_bkg[mask_gh]

        # Add one dimension for the FoV offset bin
        gh_cuts = cut_table_gh["cut"][:, np.newaxis]

        # Create a gammaness-cut HDU
        logger.info("\nCreating a gammaness-cut HDU...")

        hdu_gh_cuts = create_gh_cuts_hdu(
            gh_cuts=gh_cuts,
            reco_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            **extra_header,
        )

        irf_hdus.append(hdu_gh_cuts)

    else:
        raise ValueError(f"Unknown gammaness-cut type '{cut_type_gh}'.")

    if is_point_like:
        # Apply the theta cut
        config_theta_cuts = config_irf["theta"]
        cut_type_theta = config_theta_cuts.pop("cut_type")

        if cut_type_theta == "global":
            cut_value_theta = config_theta_cuts["global_cut_value"]
            logger.info(f"\nGlobal theta cut: {cut_value_theta}")

            cut_value_theta = u.Quantity(cut_value_theta).to_value("deg")

            extra_header["RAD_MAX"] = (cut_value_theta, "deg")
            output_suffix += f"_theta_glob{cut_value_theta}deg"

            # Apply the global theta cut
            mask_theta = event_table_gamma["theta"].to_value("deg") < cut_value_theta
            event_table_gamma = event_table_gamma[mask_theta]

        elif cut_type_theta == "dynamic":
            config_theta_cuts.pop("global_cut_value", None)

            logger.info("\nDynamic theta cuts:")
            logger.info(format_object(config_theta_cuts))

            theta_efficiency = config_theta_cuts["efficiency"]
            theta_cut_min = u.Quantity(config_theta_cuts["min_cut"])
            theta_cut_max = u.Quantity(config_theta_cuts["max_cut"])

            extra_header["TH_EFF"] = theta_efficiency
            extra_header["TH_MIN"] = (theta_cut_min.to_value("deg"), "deg")
            extra_header["TH_MAX"] = (theta_cut_max.to_value("deg"), "deg")

            output_suffix += f"_theta_dyn{theta_efficiency}"

            # Calculate dynamic theta cuts
            theta_percentile = 100 * theta_efficiency

            cut_table_theta = calculate_percentile_cut(
                values=event_table_gamma["theta"],
                bin_values=event_table_gamma["reco_energy"],
                bins=energy_bins,
                fill_value=theta_cut_max,
                percentile=theta_percentile,
                min_value=theta_cut_min,
                max_value=theta_cut_max,
            )

            logger.info(f"\nTheta-cut table:\n\n{cut_table_theta}")

            # Apply the dynamic theta cuts
            mask_theta = evaluate_binned_cut(
                values=event_table_gamma["theta"],
                bin_values=event_table_gamma["reco_energy"],
                cut_table=cut_table_theta,
                op=operator.le,
            )

            event_table_gamma = event_table_gamma[mask_theta]

            # Add one dimension for the FoV offset bin
            theta_cuts = cut_table_theta["cut"][:, np.newaxis]

            # Create a rad-max HDU
            logger.info("\nCreating a rad-max HDU...")

            hdu_rad_max = create_rad_max_hdu(
                rad_max=theta_cuts,
                reco_energy_bins=energy_bins,
                fov_offset_bins=fov_offset_bins,
                extname="RAD_MAX",
                **extra_header,
            )

            irf_hdus.append(hdu_rad_max)

        else:
            raise ValueError(f"Unknown theta-cut type '{cut_type_theta}'.")

    # Create an effective-area HDU
    logger.info("\nCreating an effective-area HDU...")

    with np.errstate(invalid="ignore", divide="ignore"):
        if is_diffuse_mc:
            aeff = effective_area_per_energy_and_fov(
                selected_events=event_table_gamma,
                simulation_info=sim_info_gamma,
                true_energy_bins=energy_bins,
                fov_offset_bins=fov_offset_bins,
            )

        else:
            aeff = effective_area_per_energy(
                selected_events=event_table_gamma,
                simulation_info=sim_info_gamma,
                true_energy_bins=energy_bins,
            )

            # Add one dimension for the FoV offset bin
            aeff = aeff[:, np.newaxis]

        aeff_hdu = create_aeff2d_hdu(
            effective_area=aeff,
            true_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            point_like=is_point_like,
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
        point_like=is_point_like,
        extname="ENERGY DISPERSION",
        **extra_header,
    )

    irf_hdus.append(edisp_hdu)

    if not is_point_like:
        # Create a PSF table HDU
        logger.info("Creating a PSF table HDU...")

        psf = psf_table(
            events=event_table_gamma,
            true_energy_bins=energy_bins,
            source_offset_bins=source_offset_bins,
            fov_offset_bins=fov_offset_bins,
        )

        psf_hdu = create_psf_table_hdu(
            psf=psf,
            true_energy_bins=energy_bins,
            source_offset_bins=source_offset_bins,
            fov_offset_bins=fov_offset_bins,
            extname="PSF",
            **extra_header,
        )

        irf_hdus.append(psf_hdu)

        if is_bkg_mc:
            # Create a background HDU
            logger.info("Creating a background HDU...")

            bkg = background_2d(
                events=event_table_bkg,
                reco_energy_bins=energy_bins,
                fov_offset_bins=bkg_fov_offset_bins,
                t_obs=obs_time,
            )

            bkg_hdu = create_background_2d_hdu(
                background_2d=bkg,
                reco_energy_bins=energy_bins,
                fov_offset_bins=bkg_fov_offset_bins,
                extname="BACKGROUND",
                **extra_header,
            )

            irf_hdus.append(bkg_hdu)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_file = (
        f"{output_dir}/irf_zd_{pnt_gamma[0]}deg_az_{pnt_gamma[1]}deg_"
        f"{event_type}_{output_suffix}.fits.gz"
    )

    irf_hdus.writeto(output_file, overwrite=True)

    logger.info(f"\nOutput file: {output_file}")


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file-gamma",
        "-g",
        dest="input_file_gamma",
        type=str,
        required=True,
        help="Path to an input gamma MC DL2 data file",
    )

    parser.add_argument(
        "--input-file-proton",
        "-p",
        dest="input_file_proton",
        type=str,
        help="Path to an input proton MC DL2 data file",
    )

    parser.add_argument(
        "--input-file-electron",
        "-e",
        dest="input_file_electron",
        type=str,
        help="Path to an input electron MC DL2 data file",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output IRF file",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config.yaml",
        help="Path to a configuration file",
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
