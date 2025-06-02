#!/usr/bin/env python
# coding: utf-8

"""
This script processes DL2 events and creates a DL3 data file with the
IRFs. At first it reads the configurations of the IRFs and checks the
consistency, and then applies the same condition cuts to DL2 events.

There are three methods for the interpolation of the IRFs, "nearest",
"linear" and "cubic", which can be specified in the configuration file.
The "nearest" method just selects the IRFs of the closest pointing
direction in (cos(Zd), Az), and the other methods work only when there
are multiple IRFs available from different pointing directions.

Usage:
$ python lst1_magic_dl2_to_dl3.py
--input-file-dl2 dl2_LST-1_MAGIC.Run03265.h5
--input-dir-irf irf
(--output-dir dl3)
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
from astropy.coordinates import Angle
from astropy.coordinates.angle_utilities import angular_separation
from astropy.io import fits
from astropy.table import QTable
from pyirf.cuts import evaluate_binned_cut
from pyirf.interpolation import GridDataInterpolator
from pyirf.io import (
    create_aeff2d_hdu,
    create_background_2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
    create_rad_max_hdu,
)
from pyirf.utils import cone_solid_angle
from scipy.interpolate import griddata

from magicctapipe.io import (
    create_event_hdu,
    create_gh_cuts_hdu,
    create_gti_hdu,
    create_pointing_hdu,
    format_object,
    load_dl2_data_file,
    load_irf_files,
)
from magicctapipe.utils import OUTSIDE_INTERPOLATION_RANGE

__all__ = ["dl2_to_dl3"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def dl2_to_dl3(input_file_dl2, input_dir_irf, output_dir, config):
    """
    Processes DL2 events and creates a DL3 data file with the IRFs.

    Parameters
    ----------
    input_file_dl2 : str
        Path to an input DL2 data file
    input_dir_irf : str
        Path to a directory where input IRF files are stored
    output_dir : str
        Path to a directory where to save an output DL3 data file
    config : dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_dl3 = config["dl2_to_dl3"]

    # Load the input IRF data files
    logger.info(f"\nInput IRF directory: {input_dir_irf}")

    irf_data, extra_header = load_irf_files(input_dir_irf)

    logger.info("\nGrid points in (cos(Zd), Az):")
    logger.info(format_object(irf_data["grid_points"]))

    logger.info("\nExtra header:")
    logger.info(format_object(extra_header))

    # Load the input DL2 data file
    logger.info(f"\nInput DL2 data file: {input_file_dl2}")

    quality_cuts = extra_header.get("QUAL_CUT")
    event_type = extra_header["EVT_TYPE"]
    dl2_weight_type = extra_header["DL2_WEIG"]

    event_table, on_time, deadc = load_dl2_data_file(
        input_file_dl2, quality_cuts, event_type, dl2_weight_type
    )

    # Calculate the mean pointing direction for the target point of the
    # IRF interpolation. Please note that the azimuth could make a full
    # 2 pi turn, whose mean angle may indicate an opposite direction.
    # Thus, here we calculate the STDs of the azimuth angles with two
    # ranges, i.e., 0 <= az < 360 deg and -180 <= az < 180 deg, and then
    # calculate the mean with the range of smaller STD.

    pnt_coszd_mean = np.sin(event_table["pointing_alt"]).mean().value

    pnt_az_wrap_360deg = Angle(event_table["pointing_az"]).wrap_at("360 deg")
    pnt_az_wrap_180deg = Angle(event_table["pointing_az"]).wrap_at("180 deg")

    if pnt_az_wrap_360deg.std() <= pnt_az_wrap_180deg.std():
        pnt_az_mean = pnt_az_wrap_360deg.mean().to_value("rad")
    else:
        pnt_az_mean = pnt_az_wrap_180deg.mean().wrap_at("360 deg").to_value("rad")

    distances = (
        angular_separation(
            pnt_az_mean,
            np.pi / 2 - np.arccos(pnt_coszd_mean),
            irf_data["grid_points"][:, 1],
            np.pi / 2 - np.arccos(irf_data["grid_points"][:, 0]),
        )
        * u.rad
    ).to("deg")

    scheme = config_dl3.pop("interpolation_scheme")
    if scheme == "cosZdAz":
        target_point = np.array([pnt_coszd_mean, pnt_az_mean])
    elif scheme == "cosZd":
        target_point = np.array([pnt_coszd_mean])
        irf_data["grid_points"] = irf_data["grid_points"][:, 0]
    else:
        logger.error(f"Not recognized interpolation scheme: {scheme}, exiting")
        exit(1)

    logger.info(
        f"\nTarget point: {target_point.round(5).tolist()} with scheme: {scheme}"
    )

    if "max_distance" in config_dl3:
        max_distance = u.Quantity(config_dl3.pop("max_distance"))
        logger.info(f"selecting only nodes up to {max_distance} from the data")

        idx = np.where(distances < max_distance)
        keys = [s for s in irf_data.keys() if "_bins" not in s]
        for key in keys:
            irf_data[key] = irf_data[key][idx]
        for i in range(len(idx[0])):
            logger.info(f"{irf_data['file_names'][i]}: {distances[idx][i]:.2f}")

    # Prepare for the IRF interpolations
    interpolation_method = config_dl3.pop("interpolation_method")
    logger.info(f"\nInterpolation method: {interpolation_method}")
    if interpolation_method != "nearest":
        coszd_margin = 0.02
        if scheme == "cosZd":
            coszds = irf_data["grid_points"]
        else:
            coszds = irf_data["grid_points"][:, 0]
        mincoszd = min(coszds)
        maxcoszd = max(coszds)
        if (pnt_coszd_mean < mincoszd and pnt_coszd_mean > mincoszd - coszd_margin) or (
            pnt_coszd_mean > maxcoszd and pnt_coszd_mean < maxcoszd + coszd_margin
        ):
            logger.warning(
                f"point {target_point} outside of IRF ranges, but within {coszd_margin}. Falling back to nearest point"
            )
            interpolation_method = "nearest"
        if (
            pnt_coszd_mean < mincoszd - coszd_margin
            or pnt_coszd_mean > maxcoszd + coszd_margin
        ):
            logger.error(
                f"point {target_point} outside of IRF range, more then {coszd_margin}, exiting"
            )
            exit(OUTSIDE_INTERPOLATION_RANGE)

    extra_header["IRF_INTP"] = interpolation_method

    hdus = fits.HDUList([fits.PrimaryHDU()])

    # Interpolate the effective area
    logger.info("\nInterpolating the effective area...")
    if len(irf_data["grid_points"]) > 2:
        effective_area = irf_data["effective_area"].to_value("m2")
        # due to large changes in Aeff we will interpolate in log space,
        # meaning that we need to set a minimal value (1 m^2)
        effective_area[effective_area < 1] = 1
        interpolator = GridDataInterpolator(
            grid_points=irf_data["grid_points"],
            params=np.log(effective_area),
            method=interpolation_method,
        )
        aeff_interp = np.exp(interpolator.interpolate(target_point)[0])
        # setting values below the minimal value back to 0, allowing for 10% error margin
        aeff_interp[aeff_interp < 1.1] = 0
        aeff_interp *= u.Unit("m2")
    else:
        aeff_interp = irf_data["effective_area"][0]
        print("skipping interpolation since only one point is given")

    aeff_hdu = create_aeff2d_hdu(
        effective_area=aeff_interp,
        true_energy_bins=irf_data["energy_bins"],
        fov_offset_bins=irf_data["fov_offset_bins"],
        point_like=True,
        extname="EFFECTIVE AREA",
        **extra_header,
    )
    hdus.append(aeff_hdu)

    # Interpolate the energy dispersion with a custom way,
    # TBD: use pyirf quantile interpolation instead
    logger.info("Interpolating the energy dispersion...")

    if len(irf_data["grid_points"]) > 2:
        edisp_interp = griddata(
            points=irf_data["grid_points"],
            values=irf_data["energy_dispersion"],
            xi=target_point,
            method=interpolation_method,
        )
    else:
        edisp_interp = irf_data["energy_dispersion"]

    edisp_interp = edisp_interp[0]  # Remove the dimension of the grid points

    norm = np.sum(edisp_interp, axis=1, keepdims=True)  # Along the migration axis
    mask_zeros = norm != 0

    edisp_interp = np.divide(
        edisp_interp, norm, out=np.zeros_like(edisp_interp), where=mask_zeros
    )

    # according to GDAF standard migration matrix is normalized to integral of bins
    widths = np.diff(irf_data["migration_bins"])
    edisp_interp /= widths[np.newaxis, :, np.newaxis]

    edisp_hdu = create_energy_dispersion_hdu(
        energy_dispersion=edisp_interp,
        true_energy_bins=irf_data["energy_bins"],
        migration_bins=irf_data["migration_bins"],
        fov_offset_bins=irf_data["fov_offset_bins"],
        point_like=True,
        extname="ENERGY DISPERSION",
    )

    hdus.append(edisp_hdu)

    if "psf_table" in irf_data:
        # Interpolate the PSF table with a custom way, since there is a
        # bug in the function of pyirf v0.6.0 about the renormalization
        logger.info("Interpolating the PSF table...")

        psf_interp = griddata(
            points=irf_data["grid_points"],
            values=irf_data["psf_table"].to_value("sr-1"),
            xi=target_point,
            method=interpolation_method,
        )

        # Remove the dimension of the grid points and add the unit
        psf_interp = psf_interp[0] * u.Unit("sr-1")

        # Re-normalize along the source offset axis
        omegas = np.diff(cone_solid_angle(irf_data["source_offset_bins"]))

        norm = np.sum(psf_interp * omegas, axis=2, keepdims=True)
        mask_zeros = norm != 0

        psf_interp = np.divide(
            psf_interp, norm, out=np.zeros_like(psf_interp), where=mask_zeros
        )

        # Create a PSF table HDU
        psf_hdu = create_psf_table_hdu(
            psf=psf_interp,
            true_energy_bins=irf_data["energy_bins"],
            source_offset_bins=irf_data["source_offset_bins"],
            fov_offset_bins=irf_data["fov_offset_bins"],
            extname="PSF",
            **extra_header,
        )

        hdus.append(psf_hdu)

    if "background" in irf_data:
        # Interpolate the background model
        logger.info("Interpolating the background model...")

        bkg = griddata(
            points=irf_data["grid_points"],
            values=irf_data["background"].to_value("MeV-1 s-1 sr-1"),
            xi=target_point,
            method=interpolation_method,
        )

        # Remove the dimension of the grid points and add the unit
        bkg = bkg[0] * u.Unit("MeV-1 s-1 sr-1")

        bkg_hdu = create_background_2d_hdu(
            background_2d=bkg,
            reco_energy_bins=irf_data["energy_bins"],
            fov_offset_bins=irf_data["fov_offset_bins"],
            extname="BACKGROUND",
        )

        hdus.append(bkg_hdu)

    if "gh_cuts" in irf_data:
        # Interpolate the dynamic gammaness cuts
        logger.info("Interpolating the dynamic gammaness cuts...")

        if len(irf_data["grid_points"]) > 2:
            gh_cuts_interp = griddata(
                points=irf_data["grid_points"],
                values=irf_data["gh_cuts"],
                xi=target_point,
                method=interpolation_method,
            )
        else:
            gh_cuts_interp = irf_data["gh_cuts"]

        # Remove the dimension of the grid points
        gh_cuts_interp = gh_cuts_interp[0]

        gh_cuts_hdu = create_gh_cuts_hdu(
            gh_cuts=gh_cuts_interp,
            reco_energy_bins=irf_data["energy_bins"],
            fov_offset_bins=irf_data["fov_offset_bins"],
            **extra_header,
        )

        hdus.append(gh_cuts_hdu)

    if "rad_max" in irf_data:
        # Interpolate the dynamic theta cuts
        logger.info("Interpolating the dynamic theta cuts...")

        rad_max_interp = griddata(
            points=irf_data["grid_points"],
            values=irf_data["rad_max"].to_value("deg"),
            xi=target_point,
            method=interpolation_method,
        )

        # Remove the dimension of the grid points and add the unit
        rad_max_interp = rad_max_interp[0] * u.deg

        rad_max_hdu = create_rad_max_hdu(
            rad_max=rad_max_interp,
            reco_energy_bins=irf_data["energy_bins"],
            fov_offset_bins=irf_data["fov_offset_bins"],
            point_like=True,
            extname="RAD_MAX",
            **extra_header,
        )

        hdus.append(rad_max_hdu)

    if "GH_CUT" in extra_header:
        # Apply the global gammaness cut
        mask_gh = event_table["gammaness"] > extra_header["GH_CUT"]
        event_table = event_table[mask_gh]

    else:
        # Apply the dynamic gammaness cuts
        gh_cut_table = QTable(
            data={
                "low": irf_data["energy_bins"][:-1],
                "high": irf_data["energy_bins"][1:],
                "cut": gh_cuts_interp.T[0],
            }
        )

        logger.info(f"\nGammaness cut table:\n\n{gh_cut_table}")

        mask_gh = evaluate_binned_cut(
            values=event_table["gammaness"],
            bin_values=event_table["reco_energy"],
            cut_table=gh_cut_table,
            op=operator.ge,
        )

        event_table = event_table[mask_gh]

    # Create an event HDU
    logger.info("\nCreating an event HDU...")

    event_hdu = create_event_hdu(event_table, on_time, deadc, **config_dl3)

    hdus.append(event_hdu)

    # Create a GTI table
    logger.info("Creating a GTI HDU...")

    gti_hdu = create_gti_hdu(event_table)

    hdus.append(gti_hdu)

    # Create a pointing table
    logger.info("Creating a pointing HDU...")

    pnt_hdu = create_pointing_hdu(event_table)

    hdus.append(pnt_hdu)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    input_file_name = Path(input_file_dl2).name

    output_file_name = input_file_name.replace("dl2", "dl3").replace(".h5", ".fits.gz")
    output_file = f"{output_dir}/{output_file_name}"

    hdus.writeto(output_file, overwrite=True)

    logger.info(f"\nOutput file: {output_file}")


def main():
    """Main function."""
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file-dl2",
        "-d",
        dest="input_file_dl2",
        type=str,
        required=True,
        help="Path to an input DL2 data file",
    )

    parser.add_argument(
        "--input-dir-irf",
        "-i",
        dest="input_dir_irf",
        type=str,
        required=True,
        help="Path to a directory where input IRF files are stored",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL3 data file",
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

    # Process the input data
    dl2_to_dl3(args.input_file_dl2, args.input_dir_irf, args.output_dir, config)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
