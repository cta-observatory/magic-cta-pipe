#!/usr/bin/env python
# coding: utf-8

"""
This script processes DL1 events and reconstructs the geometrical stereo
parameters with more than one telescope information. The quality cuts
specified in the configuration file are applied to the events before the
reconstruction.

When the input is real data containing LST and MAGIC events, it checks
the angular distances of their pointing directions and excludes the
events taken with larger distances than the limit specified in the
configuration file. This is in principle to avoid the reconstruction of
the events taken in too-mispointing situations. For example, DL1 data
may contain the coincident events taken with different wobble offsets
between the systems.

If the `--magic-only` argument is given, it reconstructs the stereo
parameters using only MAGIC events.

Usage:
$ python lst1_magic_stereo_reco.py
--input-file dl1_coincidence/dl1_LST-1_MAGIC.Run03265.0040.h5
(--output-dir dl1_stereo)
(--config-file config.yaml)
(--magic-only)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from astropy.coordinates import Angle, angular_separation
from ctapipe.containers import (
    ArrayEventContainer,
    CameraHillasParametersContainer,
    ImageParametersContainer,
    LeakageContainer,
    MorphologyContainer,
    TimingParametersContainer,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import read_table, write_table
from ctapipe.reco import HillasReconstructor
from lstchain.io import HDF5_ZSTD_FILTERS
from numpy.linalg import LinAlgError

from magicctapipe.io import (
    check_input_list,
    format_object,
    get_stereo_events,
    save_pandas_data_in_table,
)
from magicctapipe.utils import (
    NO_EVENTS_WITHIN_MAXIMUM_DISTANCE,
    calculate_mean_direction,
)

__all__ = ["calculate_pointing_separation", "stereo_reconstruction"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def calculate_pointing_separation(event_data, config):
    """
    Calculates the angular distance of the LST and MAGIC pointing
    directions.

    Parameters
    ----------
    event_data : pandas.core.frame.DataFrame
        Data frame of LST and MAGIC events
    config : dict
        Configuration for the LST + MAGIC analysis

    Returns
    -------
    pandas.core.series.Series
        Angular distance of the LST array and MAGIC pointing directions
        in units of degree
    """

    assigned_tel_ids = config[
        "mc_tel_ids"
    ]  # This variable becomes a dictionary, e.g.: {'LST-1': 1, 'LST-2': 0, 'LST-3': 0, 'LST-4': 0, 'MAGIC-I': 2, 'MAGIC-II': 3}
    LSTs_IDs = np.asarray(list(assigned_tel_ids.values())[0:4])
    LSTs_IDs = list(LSTs_IDs[LSTs_IDs > 0])  # Here we list only the LSTs in use
    MAGICs_IDs = np.asarray(list(assigned_tel_ids.values())[4:6])
    MAGICs_IDs = list(MAGICs_IDs[MAGICs_IDs > 0])  # Here we list only the MAGICs in use

    # Extract LST events
    df_lst = event_data.query(f"tel_id == {LSTs_IDs}")

    # Extract the coincident events observed by both MAGICs and LSTs
    df_magic = event_data.query(f"tel_id == {MAGICs_IDs}")
    df_magic = df_magic.loc[df_lst.index]

    # Calculate the mean of the LSTs, and also of the M1 and M2 pointing directions
    pnt_az_LST, pnt_alt_LST = calculate_mean_direction(
        lon=df_lst["pointing_az"], lat=df_lst["pointing_alt"], unit="rad"
    )
    pnt_az_magic, pnt_alt_magic = calculate_mean_direction(
        lon=df_magic["pointing_az"], lat=df_magic["pointing_alt"], unit="rad"
    )

    # Calculate the angular distance of their pointing directions
    theta = angular_separation(
        lon1=u.Quantity(pnt_az_LST, unit="rad"),
        lat1=u.Quantity(pnt_alt_LST, unit="rad"),
        lon2=u.Quantity(pnt_az_magic, unit="rad"),
        lat2=u.Quantity(pnt_alt_magic, unit="rad"),
    )

    theta = pd.Series(data=theta.to_value("deg"), index=df_lst.index)

    return theta


def stereo_reconstruction(input_file, output_dir, config, magic_only_analysis=False):
    """
    Processes DL1 events and reconstructs the geometrical stereo
    parameters with more than one telescope information.

    Parameters
    ----------
    input_file : str
        Path to an input DL1 data file
    output_dir : str
        Path to a directory where to save an output DL1-stereo data file
    config : dict
        Configuration file for the stereo LST + MAGIC analysis, i.e. config_stereo.yaml
    magic_only_analysis : bool, optional
        If `True`, it reconstructs the stereo parameters using only
        MAGIC events
    """

    config_stereo = config["stereo_reco"]
    assigned_tel_ids = config[
        "mc_tel_ids"
    ]  # This variable becomes a dictionary, e.g.: {'LST-1': 1, 'LST-2': 0, 'LST-3': 0, 'LST-4': 0, 'MAGIC-I': 2, 'MAGIC-II': 3}

    save_images = config.get("save_images", False)

    # Load the input file
    logger.info(f"\nInput file: {input_file}")

    event_data = pd.read_hdf(input_file, key="events/parameters")
    # It sometimes happens that there are MAGIC events whose event and
    # telescope IDs are duplicated, so here we exclude those events
    event_data.drop_duplicates(
        subset=["obs_id", "event_id", "tel_id"], keep=False, inplace=True
    )

    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    is_simulation = "true_energy" in event_data.columns
    logger.info(f"\nIs simulation: {is_simulation}")

    subarray = SubarrayDescription.from_hdf(input_file)
    tel_positions = subarray.positions

    logger.info("\nTelescope positions:")
    logger.info(format_object(tel_positions))

    # Apply the event cuts
    logger.info(f"\nMAGIC-only analysis: {magic_only_analysis}")

    LSTs_IDs = np.asarray(list(assigned_tel_ids.values())[0:4])

    if magic_only_analysis:
        tel_id = np.asarray(list(assigned_tel_ids.values())[:])
        used_id = tel_id[tel_id != 0]
        magic_ids = [item for item in used_id if item not in LSTs_IDs]
        event_data.query(
            f"tel_id in {magic_ids}", inplace=True
        )  # Here we select only the events with the MAGIC tel_ids

    logger.info(f"\nQuality cuts: {config_stereo['quality_cuts']}")
    event_data = get_stereo_events(
        event_data, config=config, quality_cuts=config_stereo["quality_cuts"]
    )

    # Check the angular distance of the LST and MAGIC pointing directions
    tel_ids = np.unique(event_data.index.get_level_values("tel_id")).tolist()

    Number_of_LSTs_in_use = len(LSTs_IDs[LSTs_IDs > 0])
    MAGICs_IDs = np.asarray(list(assigned_tel_ids.values())[4:6])
    Number_of_MAGICs_in_use = len(MAGICs_IDs[MAGICs_IDs > 0])
    Two_arrays_are_used = Number_of_LSTs_in_use * Number_of_MAGICs_in_use > 0

    if (not is_simulation) and (Two_arrays_are_used):
        logger.info(
            "\nChecking the angular distances of "
            "the LST and MAGIC pointing directions..."
        )

        event_data.reset_index(level="tel_id", inplace=True)

        # Calculate the angular distance
        theta = calculate_pointing_separation(event_data, config)
        theta_uplim = u.Quantity(config_stereo["theta_uplim"])
        mask = u.Quantity(theta, unit="deg") < theta_uplim

        try:
            if all(mask):
                logger.info(
                    "--> All the events were taken with smaller angular distances "
                    f"than the limit {theta_uplim}."
                )

            elif not any(mask):
                raise Exception(
                    f"--> All the events were taken with larger angular distances "
                    f"than the limit {theta_uplim}. Exiting..."
                )

            else:
                logger.info(
                    f"--> Exclude {np.count_nonzero(mask)} stereo events whose "
                    f"angular distances are larger than the limit {theta_uplim}."
                )
                event_data = event_data.loc[theta[mask].index]
        except Exception as e:
            logger.error(f"{e}")
            sys.exit(NO_EVENTS_WITHIN_MAXIMUM_DISTANCE)

        event_data.set_index("tel_id", append=True, inplace=True)

    # Configure the HillasReconstructor
    hillas_reconstructor = HillasReconstructor(subarray)

    # Calculate the mean pointing direction
    pnt_az_mean, pnt_alt_mean = calculate_mean_direction(
        lon=event_data["pointing_az"], lat=event_data["pointing_alt"], unit="rad"
    )

    # Loop over every shower event
    logger.info("\nReconstructing the stereo parameters...")

    multi_indices = event_data.groupby(["obs_id", "event_id"]).size().index

    for i_evt, (obs_id, event_id) in enumerate(multi_indices):
        if i_evt % 100 == 0:
            logger.info(f"{i_evt} events")

        # Create an array event container
        event = ArrayEventContainer()

        # Assign the mean pointing direction
        event.pointing.array_altitude = pnt_alt_mean.loc[(obs_id, event_id)] * u.rad
        event.pointing.array_azimuth = pnt_az_mean.loc[(obs_id, event_id)] * u.rad

        # Extract the data frame of the shower event
        df_evt = event_data.loc[(obs_id, event_id, slice(None))]

        # Loop over every telescope
        tel_ids = df_evt.index.get_level_values("tel_id").values

        event.trigger.tels_with_trigger = tel_ids

        for tel_id in tel_ids:
            df_tel = df_evt.loc[tel_id]

            # Assign the telescope information
            event.pointing.tel[tel_id].altitude = df_tel["pointing_alt"] * u.rad
            event.pointing.tel[tel_id].azimuth = df_tel["pointing_az"] * u.rad

            hillas_params = CameraHillasParametersContainer(
                intensity=float(df_tel["intensity"]),
                x=u.Quantity(df_tel["x"], unit="m"),
                y=u.Quantity(df_tel["y"], unit="m"),
                r=u.Quantity(df_tel["r"], unit="m"),
                phi=Angle(df_tel["phi"], unit="deg"),
                psi=Angle(df_tel["psi"], unit="deg"),
                length=u.Quantity(df_tel["length"], unit="m"),
                length_uncertainty=u.Quantity(df_tel["length_uncertainty"], unit="m"),
                width=u.Quantity(df_tel["width"], unit="m"),
                width_uncertainty=u.Quantity(df_tel["width_uncertainty"], unit="m"),
                skewness=float(df_tel["skewness"]),
                kurtosis=float(df_tel["kurtosis"]),
            )
            time_par = TimingParametersContainer(
                slope=u.Quantity(df_tel["slope"], unit="1/deg"),
                intercept=float(df_tel["intercept"]),
            )
            leak_par = LeakageContainer(
                intensity_width_1=float(df_tel["intensity_width_1"]),
                intensity_width_2=float(df_tel["intensity_width_2"]),
                pixels_width_1=float(df_tel["pixels_width_1"]),
                pixels_width_2=float(df_tel["pixels_width_2"]),
            )

            morph_par = MorphologyContainer(
                n_islands=int(df_tel["n_islands"]),
                # n_large_islands=int(df_morph_tel["n_large_islands"]),
                # n_medium_islands=int(df_morph_tel["n_medium_islands"]),
                # n_small_islands=int(df_morph_tel["n_small_islands"]),
                n_large_islands=np.nan,
                n_medium_islands=np.nan,
                n_small_islands=np.nan,
                n_pixels=int(df_tel["n_pixels"]),
            )

            event.dl1.tel[tel_id].parameters = ImageParametersContainer(
                hillas=hillas_params,
                timing=time_par,
                leakage=leak_par,
                morphology=morph_par,
            )
            # event.dl1.tel[tel_id].is_valid=True

        # Reconstruct the stereo parameters
        try:
            hillas_reconstructor(event)
        except LinAlgError:
            logger.info(
                f"--> event {i_evt} (event ID {event_id}) failed to reconstruct valid "
                "stereo parameters, caught LinAlgError exception. Skipping..."
            )
            continue

        reconstructor_name = "HillasReconstructor"

        stereo_params = event.dl2.stereo.geometry[reconstructor_name]

        if not stereo_params.is_valid:
            logger.info(
                f"--> event {i_evt} (event ID {event_id}) failed to reconstruct valid "
                "stereo parameters, maybe due to the images of zero width. Skipping..."
            )
            continue

        stereo_params.az.wrap_at("360 deg", inplace=True)

        for tel_id in tel_ids:
            # Set the stereo parameters to the data frame
            params = {
                "alt": stereo_params.alt.to_value("deg"),
                "alt_uncert": stereo_params.alt_uncert.to_value("deg"),
                "az": stereo_params.az.to_value("deg"),
                "az_uncert": stereo_params.az_uncert.to_value("deg"),
                "core_x": stereo_params.core_x.to_value("m"),
                "core_y": stereo_params.core_y.to_value("m"),
                "impact": event.dl2.tel[tel_id]
                .impact[reconstructor_name]
                .distance.to_value("m"),
                "h_max": stereo_params.h_max.to_value("m"),
                "is_valid": int(stereo_params.is_valid),
            }

            event_data.loc[(obs_id, event_id, tel_id), params.keys()] = params.values()

    n_events_processed = i_evt + 1
    logger.info(f"{n_events_processed} events")

    event_data.reset_index(inplace=True)

    # Save the data in an output file
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    input_file_name = Path(input_file).name

    if magic_only_analysis:
        output_file_name = input_file_name.replace("dl1", "dl1_stereo_magic_only")
    else:
        output_file_name = input_file_name.replace("dl1", "dl1_stereo")

    output_file = f"{output_dir}/{output_file_name}"

    save_pandas_data_in_table(
        event_data, output_file, group_name="/events", table_name="parameters"
    )

    if save_images:
        tel_ids = np.unique(event_data["tel_id"]).tolist()
        for tel_id in tel_ids:
            key = f"/events/dl1/image_{tel_id}"
            image = read_table(input_file, key)
            mask = np.zeros(len(image), dtype=bool)
            tag = "lst" if tel_id in LSTs_IDs else "magic"
            obs_evt_ids = event_data.query(f"tel_id=={tel_id}")[
                [f"obs_id_{tag}", f"event_id_{tag}"]
            ].to_numpy()
            for obs_id, evt_id in obs_evt_ids:
                this_mask = np.logical_and(
                    image["obs_id"] == obs_id, image["event_id"] == evt_id
                )
                mask = np.logical_or(mask, this_mask)
            logger.info(f"found {sum(mask)} images of tel_id={tel_id}")
            write_table(
                image[mask],
                output_file,
                key,
                overwrite=True,
                filters=HDF5_ZSTD_FILTERS,
            )

    # Save the subarray description
    subarray.to_hdf(output_file)

    if is_simulation:
        # Save the simulation configuration
        sim_config = pd.read_hdf(input_file, key="simulation/config")

        save_pandas_data_in_table(
            input_data=sim_config,
            output_file=output_file,
            group_name="/simulation",
            table_name="config",
            mode="a",
        )

    logger.info(f"\nOutput file: {output_file}")


def main():
    """Main function."""
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        "-i",
        dest="input_file",
        type=str,
        required=True,
        help="Path to an input DL1 data file",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL1-stereo data file",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config.yaml",
        help="Path to a configuration file",
    )

    parser.add_argument(
        "--magic-only",
        dest="magic_only",
        action="store_true",
        help="Reconstruct the stereo parameters using only MAGIC events",
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Checking if the input telescope list is properly organized:
    check_input_list(config)

    # Process the input data
    stereo_reconstruction(args.input_file, args.output_dir, config, args.magic_only)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
