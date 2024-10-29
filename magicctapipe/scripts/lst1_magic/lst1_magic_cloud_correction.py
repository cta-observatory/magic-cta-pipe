#!/usr/bin/env python
# coding: utf-8

"""
This script corrects LST-1 and MAGIC data for the cloud affection. The script works on DL 1 stereo files with saved images. As output, the script creates DL 1 files with corrected parameters.

Usage:
$ python lst_m1_m2_cloud_correction.py
--input_file dl1_stereo/dl1_LST-1_MAGIC.Run03265.0040.h5
(--output_dir dl1_corrected)
(--config_file config.yaml)
"""
import argparse
import logging
import os
import time
from pathlib import Path

import astropy.units as u
import ctapipe
import numpy as np
import pandas as pd
import tables
import yaml
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from ctapipe.coordinates import TelescopeFrame
from ctapipe.image import (
    concentration_parameters,
    hillas_parameters,
    leakage_parameters,
    timing_parameters,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import read_table
from scipy.interpolate import interp1d

import magicctapipe
from magicctapipe.io import save_pandas_data_in_table

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def model0(imp, h, zd):
    """
    Calculates the geometrical part of the model relating the emission height with the angular distance from the arrival direction

    Parameters
    ----------
    imp : astropy.units.quantity.Quantity
        Impact
    h : astropy.units.quantity.Quantity
        Array with heights of each cloud layer a.g.l.
    zd : numpy.float64
        Zenith distance in deg

    Returns
    -------
    numpy ndarray
        Angular distance in units of degree
    """
    d = h / np.cos(np.deg2rad(zd))
    return np.arctan((imp / d).to("")).to_value("deg")


def model2(imp, h, zd):
    """
    Calculates the phenomenological correction to the distances obtained with model0

    Parameters
    ----------
    imp : astropy.units.quantity.Quantity
        Impact
    h : astropy.units.quantity.Quantity
        Array with heights of each cloud layer a.g.l.
    zd : numpy.float64
        Zenith distance in deg

    Returns
    -------
    numpy ndarray
        Angular distance corrected for bias in units of degrees
    """
    H0 = 2.2e3 * u.m
    bias = 0.877 + 0.015 * ((h + H0) / (7.0e3 * u.m))
    return bias * model0(imp, h, zd)


def trans_height(x, Hc, dHc, trans):
    """
    Calculates transmission from a geometrically broad cloud at a set of heights.
    Cloud is assumed to be homegeneous.

    Parameters
    ----------
    x : astropy.units.quantity.Quantity
        Array with heights of each cloud layer a.g.l.
    Hc : astropy.units.quantity.Quantity
        Height of the base of the cloud a.g.l.
    dHc : astropy.units.quantity.Quantity
        Cloud thickness
    trans : numpy.float64
        Transmission of the cloud

    Returns
    -------
    numpy.ndarray
        Cloud transmission
    """
    t = pow(trans, ((x - Hc) / dHc).to_value(""))
    t = np.where(x < Hc, 1, t)
    t = np.where(x > Hc + dHc, trans, t)
    return t


def lidar_cloud_interpolation(
    mean_subrun_timestamp, max_gap_lidar_shots, lidar_report_file
):
    """
    Retrieves or interpolates LIDAR cloud parameters based on the closest timestamps to an input mean timestamp of the processed subrun.

    Parameters
    -----------
    mean_subrun_timestamp : int or float
        The target timestamp (format: unix).

    max_gap_lidar_shots : int or float
        Maximum allowed time gap for interpolation (in seconds).

    lidar_report_file : str
        Path to the CSV file containing LIDAR laser reports with columns:
        - "timestamp" (format: ISO 8601)
        - "base_height", "top_height", "transmission", "lidar_zenith"

    Returns
    --------
    tuple or None A tuple containing interpolated or nearest values for:
        - base_height (float): The base height of the cloud layer in meters.
        - top_height (float): The top height of the cloud layer in meters.
        - vertical_transmission (float): Transmission factor of the cloud layer adjusted for vertical angle.

        If no nodes are found within the maximum allowed time gap, returns None.
    """

    if not os.path.isfile(lidar_report_file):
        raise FileNotFoundError(f"LIDAR report file not found: {lidar_report_file}")

    df = pd.read_csv(lidar_report_file, sep=",", skiprows=[1])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["unix_timestamp"] = df["timestamp"].astype(np.int64) / 10**9
    df["time_diff"] = df["unix_timestamp"] - mean_subrun_timestamp

    df["vertical_transmission"] = df["transmission"] * np.cos(
        np.deg2rad(df["lidar_zenith"])
    )

    closest_node_before = df[
        (df["time_diff"] < 0) & (np.abs(df["time_diff"]) <= max_gap_lidar_shots)
    ].nlargest(1, "time_diff")
    closest_node_after = df[
        (df["time_diff"] > 0) & (np.abs(df["time_diff"]) <= max_gap_lidar_shots)
    ].nsmallest(1, "time_diff")

    # Check whether the conditions for interpolation are met or not
    if not closest_node_before.empty and not closest_node_after.empty:
        node_before = closest_node_before.iloc[0]
        node_after = closest_node_after.iloc[0]
        logger.info(
            f"\nFound suitable interpolation nodes within the allowed temporal gap for timestamp {Time(mean_subrun_timestamp, format='unix').iso}"
            f"\nUsing folowing interpolation nodes:\n"
            f"\n******************** Node before ******************* \n{node_before}"
            f"\n******************** Node after ******************** \n{node_after}"
            f"\n\nInterpolation results:"
        )

        interp_values = {}
        for param in ["base_height", "top_height", "vertical_transmission"]:
            interp_func = interp1d(
                [node_before["unix_timestamp"], node_after["unix_timestamp"]],
                [node_before[param], node_after[param]],
                kind="linear",
                bounds_error=False,
            )
            interp_values[param] = interp_func(mean_subrun_timestamp)
            logger.info(f"\t {param}: {interp_values[param]:.4f}")

        return (
            interp_values["base_height"],
            interp_values["top_height"],
            interp_values["vertical_transmission"],
        )

    # Handle cases where only one node is available
    closest_node = (
        closest_node_before if not closest_node_before.empty else closest_node_after
    )

    if closest_node is not None and not closest_node.empty:
        closest = closest_node.iloc[0]
        logger.info(
            f"\nOnly one suitable LIDAR report found for timestamp {Time(mean_subrun_timestamp, format='unix').iso} "
            f"within the maximum allowed temporal gap. \nSkipping interpolation. Using nearest node values instead."
            f"\n\n{closest}"
        )
        return (
            closest["base_height"],
            closest["top_height"],
            closest["vertical_transmission"],
        )

    logger.info(
        f"\nNo node is within the maximum allowed temporal gap for timestamp {Time(mean_subrun_timestamp, format='unix').iso}. Exiting ..."
    )
    return None


def process_telescope_data(input_file, config, tel_id, camgeom, focal_eff):
    """
    Corrects LST-1 and MAGIC data affected by a cloud presence

    Parameters
    ----------
    input_file : str
        Path to an input .h5 DL1 file
    config : dict
        Configuration for the LST-1 + MAGIC analysis
    tel_id : numpy.int16
        LST-1 and MAGIC telescope ids
    camgeom : ctapipe.instrument.camera.geometry.CameraGeometry
        An instance of the CameraGeometry class containing information about the
        camera's configuration, including pixel type, number of pixels, rotation
        angles, and the reference frame.
    focal_eff : astropy.units.quantity.Quantity
        Effective focal length

    Returns
    -------
    pandas.core.frame.DataFrame
        Data frame of corrected DL1 parameters
    """

    assigned_tel_ids = config["mc_tel_ids"]

    correction_params = config.get("cloud_correction", {})
    max_gap_lidar_shots = u.Quantity(
        correction_params.get("max_gap_lidar_shots")
    )  # set it to 900 seconds
    lidar_report_file = correction_params.get(
        "lidar_report_file"
    )  # path to the lidar report
    nlayers = correction_params.get("number_of_layers")

    all_params_list = []

    dl1_params = read_table(input_file, "/events/parameters")

    image_node_path = "/events/dl1/image_" + str(tel_id)

    try:
        dl1_images = read_table(input_file, image_node_path)
        logger.info(f"Found images for telescope with ID {tel_id}. Processing...")
    except tables.NoSuchNodeError:
        logger.info(f"No image for telescope with ID {tel_id}. Skipping.")
        return None

    m2deg = np.rad2deg(1) / focal_eff * u.degree

    mean_subrun_timestamp = np.mean(dl1_params["timestamp"])
    mean_subrun_zenith = np.mean(90.0 - np.rad2deg(dl1_params["pointing_alt"]))

    cloud_params = lidar_cloud_interpolation(
        mean_subrun_timestamp, max_gap_lidar_shots, lidar_report_file
    )

    Hc = u.Quantity(cloud_params[0], u.m)
    dHc = u.Quantity(cloud_params[1] - cloud_params[0], u.m)
    incl_trans = u.Quantity(cloud_params[2])
    trans = incl_trans ** (
        1 / np.cos(np.deg2rad(mean_subrun_zenith))
    )  # vertical transmission
    Hcl = np.linspace(Hc, Hc + dHc, nlayers)  # position of each layer
    transl = trans_height(Hcl, Hc, dHc, trans)  # transmissions of each layer
    transl = np.append(transl, transl[-1])

    inds = np.where(
        np.logical_and(dl1_params["intensity"] > 0.0, dl1_params["tel_id"] == tel_id)
    )[0]
    for index in inds:
        event_id_lst = dl1_params["event_id_lst"][index]
        obs_id_lst = dl1_params["obs_id_lst"][index]
        event_id = dl1_params["event_id"][index]
        obs_id = dl1_params["obs_id"][index]
        event_id_magic = dl1_params["event_id_magic"][index]
        obs_id_magic = dl1_params["obs_id_magic"][index]
        timestamp = dl1_params["timestamp"][index]
        multiplicity = dl1_params["multiplicity"][index]
        combo_type = dl1_params["combo_type"][index]

        if assigned_tel_ids["LST-1"] == tel_id:
            event_id_image, obs_id_image = event_id_lst, obs_id_lst
        else:
            event_id_image, obs_id_image = event_id_magic, obs_id_magic

        pointing_az = dl1_params["pointing_az"][index]
        pointing_alt = dl1_params["pointing_alt"][index]
        time_diff = dl1_params["time_diff"][index]
        n_islands = dl1_params["n_islands"][index]
        signal_pixels = dl1_params["n_pixels"][index]

        alt_rad = np.deg2rad(dl1_params["alt"][index])
        az_rad = np.deg2rad(dl1_params["az"][index])

        impact = dl1_params["impact"][index] * u.m
        cog_x = (dl1_params["x"][index] * m2deg).value * u.deg
        cog_y = (dl1_params["y"][index] * m2deg).value * u.deg

        # Source position
        reco_pos = SkyCoord(alt=alt_rad * u.rad, az=az_rad * u.rad, frame=AltAz())
        telescope_pointing = SkyCoord(
            alt=pointing_alt * u.rad,
            az=pointing_az * u.rad,
            frame=AltAz(),
        )

        tel_frame = TelescopeFrame(telescope_pointing=telescope_pointing)
        tel = reco_pos.transform_to(tel_frame)

        src_x = tel.fov_lat
        src_y = tel.fov_lon

        # Transform to Engineering camera
        src_x, src_y = -src_y, -src_x
        cog_x, cog_y = -cog_y, -cog_x

        psi = np.arctan2(src_x - cog_x, src_y - cog_y)

        pix_x_tel = (camgeom.pix_x * m2deg).to(u.deg)
        pix_y_tel = (camgeom.pix_y * m2deg).to(u.deg)

        distance = np.abs(
            (pix_y_tel - src_y) * np.cos(psi) + (pix_x_tel - src_x) * np.sin(psi)
        )

        d2_cog_src = (cog_x - src_x) ** 2 + (cog_y - src_y) ** 2
        d2_cog_pix = (cog_x - pix_x_tel) ** 2 + (cog_y - pix_y_tel) ** 2
        d2_src_pix = (src_x - pix_x_tel) ** 2 + (src_y - pix_y_tel) ** 2

        distance[d2_cog_pix > d2_cog_src + d2_src_pix] = 0
        dist_corr_layer = model2(impact, Hcl, mean_subrun_zenith) * u.deg

        ilayer = np.digitize(distance, dist_corr_layer)
        trans_pixels = transl[ilayer]

        inds_img = np.where(
            (dl1_images["event_id"] == event_id_image)
            & (dl1_images["tel_id"] == tel_id)
            & (dl1_images["obs_id"] == obs_id_image)
        )[0]

        if len(inds_img) == 0:
            raise ValueError("Error: 'inds_img' list is empty!")
        index_img = inds_img[0]
        image = dl1_images["image"][index_img]
        cleanmask = dl1_images["image_mask"][index_img]
        peak_time = dl1_images["peak_time"][index_img]
        image /= trans_pixels
        corr_image = image.copy()

        clean_image = corr_image[cleanmask]
        clean_camgeom = camgeom[cleanmask]

        hillas_params = hillas_parameters(clean_camgeom, clean_image)
        timing_params = timing_parameters(
            clean_camgeom, clean_image, peak_time[cleanmask], hillas_params
        )
        leakage_params = leakage_parameters(camgeom, corr_image, cleanmask)
        if assigned_tel_ids["LST-1"] == tel_id:
            conc_params = concentration_parameters(
                clean_camgeom, clean_image, hillas_params
            )  # For LST-1 we compute concentration from the cleaned image and for MAGIC from the full image to reproduce the current behaviour in the standard code
        else:
            conc_params = concentration_parameters(camgeom, corr_image, hillas_params)

        event_params = {
            **hillas_params,
            **timing_params,
            **leakage_params,
        }

        prefixed_conc_params = {
            f"concentration_{key}": value for key, value in conc_params.items()
        }
        event_params.update(prefixed_conc_params)

        event_info_dict = {
            "obs_id": obs_id,
            "event_id": event_id,
            "tel_id": tel_id,
            "pointing_alt": pointing_alt,
            "pointing_az": pointing_az,
            "time_diff": time_diff,
            "n_pixels": signal_pixels,
            "n_islands": n_islands,
            "event_id_lst": event_id_lst,
            "obs_id_lst": obs_id_lst,
            "event_id_magic": event_id_magic,
            "obs_id_magic": obs_id_magic,
            "combo_type": combo_type,
            "timestamp": timestamp,
            "multiplicity": multiplicity,
        }
        event_params.update(event_info_dict)

        all_params_list.append(event_params)

    df = pd.DataFrame(all_params_list)
    return df


def main():
    """Main function."""
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        "-i",
        dest="input_file",
        type=str,
        required=True,
        help="Path to an input .h5 DL1 data file",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output corrected DL1 file",
    )

    parser.add_argument(
        "--config_file",
        "-c",
        dest="config_file",
        type=str,
        default="./resources/config.yaml",
        help="Path to a configuration file",
    )

    args = parser.parse_args()

    subarray_info = SubarrayDescription.from_hdf(args.input_file)

    tel_descriptions = subarray_info.tel
    camgeom = {}
    for telid, telescope in tel_descriptions.items():
        camgeom[telid] = telescope.camera.geometry

    optics_table = read_table(
        args.input_file, "/configuration/instrument/telescope/optics"
    )
    focal_eff = {}

    for telid, telescope in tel_descriptions.items():
        optics_row = optics_table[optics_table["optics_name"] == telescope.name]
        if len(optics_row) > 0:
            focal_eff[telid] = optics_row["effective_focal_length"][0] * u.m
        else:
            raise ValueError(f"No optics data found for telescope: {telescope.name}")

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    tel_ids = config["mc_tel_ids"]

    dfs = []  # Initialize an empty list to store DataFrames

    for tel_name, tel_id in tel_ids.items():
        if tel_id != 0:  # Only process telescopes that have a non-zero ID
            df = process_telescope_data(
                args.input_file, config, tel_id, camgeom[tel_id], focal_eff[tel_id]
            )

            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    columns_to_convert = [
        "x",
        "y",
        "r",
        "phi",
        "length",
        "length_uncertainty",
        "width",
        "width_uncertainty",
        "psi",
        "slope",
    ]

    for col in columns_to_convert:
        df_all[col] = df_all[col].apply(
            lambda x: x.value if isinstance(x, u.Quantity) else x
        )

    df_all["psi"] = np.degrees(df_all["psi"])
    df_all["phi"] = np.degrees(df_all["phi"])

    for col in columns_to_convert:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_all = df_all.drop(columns=["deviation"], errors="ignore")

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    input_file_name = Path(args.input_file).name
    output_file_name = input_file_name.replace("dl1_stereo", "dl1_corr")
    output_file = f"{args.output_dir}/{output_file_name}"

    save_pandas_data_in_table(
        df_all, output_file, group_name="/events", table_name="parameters"
    )

    subarray_info.to_hdf(output_file)

    correction_params = config.get("cloud_correction", {})

    logger.info(f"Correction parameters: {correction_params}")
    logger.info(f"ctapipe version: {ctapipe.__version__}")
    logger.info(f"magicctapipe version: {magicctapipe.__version__}")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")
    logger.info(f"\nOutput file: {output_file}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
