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

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import AltAz, SkyCoord
from ctapipe.coordinates import TelescopeFrame
from ctapipe.image import (
    concentration_parameters,
    hillas_parameters,
    leakage_parameters,
    timing_parameters,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import read_table

from magicctapipe.io import save_pandas_data_in_table

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def model0(imp, h, zd):
    """
    Calculated the geometrical part of the model relating the emission height with the angular distance from the arrival direction

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
    d = h / np.cos(zd)
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
    Hc = u.Quantity(correction_params.get("base_height"))
    dHc = u.Quantity(correction_params.get("thickness"))
    trans0 = correction_params.get("vertical_transmission")
    nlayers = correction_params.get("number_of_layers")

    all_params_list = []

    dl1_params = read_table(input_file, "/events/parameters")
    dl1_images = read_table(input_file, "/events/dl1/image_" + str(tel_id))

    m2deg = np.rad2deg(1) / focal_eff * u.degree

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
        zenith = 90.0 - np.rad2deg(pointing_alt)
        psi = dl1_params["psi"][index] * u.deg
        time_diff = dl1_params["time_diff"][index]
        n_islands = dl1_params["n_islands"][index]
        signal_pixels = dl1_params["n_pixels"][index]

        trans = trans0 ** (1 / np.cos(zenith))
        Hcl = np.linspace(Hc, Hc + dHc, nlayers)  # position of each layer
        transl = trans_height(Hcl, Hc, dHc, trans)  # transmissions of each layer
        transl = np.append(transl, transl[-1])

        alt_rad = np.deg2rad(dl1_params["alt"][index])
        az_rad = np.deg2rad(dl1_params["az"][index])

        impact = dl1_params["impact"][index] * u.m
        psi = dl1_params["psi"][index] * u.deg
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

        pix_x_tel = (camgeom.pix_x * m2deg).to(u.deg)
        pix_y_tel = (camgeom.pix_y * m2deg).to(u.deg)

        distance = np.abs(
            (pix_y_tel - src_y) * np.cos(psi) + (pix_x_tel - src_x) * np.sin(psi)
        )

        d2_cog_src = (cog_x - src_x) ** 2 + (cog_y - src_y) ** 2
        d2_cog_pix = (cog_x - pix_x_tel) ** 2 + (cog_y - pix_y_tel) ** 2
        d2_src_pix = (src_x - pix_x_tel) ** 2 + (src_y - pix_y_tel) ** 2

        distance[d2_cog_pix > d2_cog_src + d2_src_pix] = 0
        dist_corr_layer = model2(impact, Hcl, zenith) * u.deg

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
            )  # "For LST-1 we compute concentration from the cleaned image and for MAGIC from the full image to reproduce the current behaviour in the standard code
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
        "--output_file",
        "-o",
        dest="output_file",
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

    for col in columns_to_convert:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_all = df_all.drop(columns=["deviation"], errors="ignore")

    save_pandas_data_in_table(
        df_all, args.output_file, group_name="/events", table_name="parameters"
    )

    subarray_info.to_hdf(args.output_file)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
