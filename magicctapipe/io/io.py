#!/usr/bin/env python
# coding: utf-8

import glob
import logging

import numpy as np
import pandas as pd
import tables
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from ctapipe.containers import EventType
from ctapipe.coordinates import CameraFrame
from ctapipe.instrument import SubarrayDescription
from lstchain.reco.utils import add_delta_t_key
from magicctapipe.utils import calculate_mean_direction, transform_altaz_to_radec
from pyirf.binning import join_bin_lo_hi
from pyirf.simulations import SimulatedEventsInfo
from pyirf.utils import calculate_source_fov_offset, calculate_theta

__all__ = [
    "get_stereo_events",
    "get_dl2_mean",
    "load_lst_dl1_data_file",
    "load_magic_dl1_data_files",
    "load_train_data_files",
    "load_mc_dl2_data_file",
    "load_dl2_data_file",
    "load_irf_files",
    "save_pandas_data_in_table",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# The telescope IDs and names
TEL_NAMES = {1: "LST-1", 2: "MAGIC-I", 3: "MAGIC-II"}

# The telescope combination types
TEL_COMBINATIONS = {
    "m1_m2": [2, 3],  # combo_type = 0
    "lst1_m1": [1, 2],  # combo_type = 1
    "lst1_m2": [1, 3],  # combo_type = 2
    "lst1_m1_m2": [1, 2, 3],  # combo_type = 3
}

# The pandas multi index to classify the events simulated by different
# telescope pointing directions but have the same observation ID
GROUP_INDEX_TRAIN = ["obs_id", "event_id", "true_alt", "true_az"]

# Event weight for training RFs, but it is set to 1, meaning no weights.
# ToBeChecked: what weights are best for training RFs?
EVENT_WEIGHT = 1

# The LST nominal and effective focal lengths
NOMINAL_FOCLEN_LST = 28 * u.m
EFFECTIVE_FOCLEN_LST = 29.30565 * u.m

# The upper limit of the trigger time differences of consecutive events,
# allowed when calculating the ON time and dead time correction factor
TIME_DIFF_UPLIM = 0.1 * u.s

# The LST-1 and MAGIC readout dead times
DEAD_TIME_LST = 7.6 * u.us
DEAD_TIME_MAGIC = 26 * u.us


def get_stereo_events(
    event_data, quality_cuts=None, group_index=["obs_id", "event_id"]
):
    """
    Gets the stereo events surviving specified quality cuts.

    It adds the telescope multiplicity and combination types to the
    output data frame.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Data frame of shower events
    quality_cuts: str
        Quality cuts applied to the input data
    group_index: list
        Index to group telescope-wise events

    Returns
    -------
    event_data_stereo: pandas.core.frame.DataFrame
        Data frame of the stereo events surviving the quality cuts
    """

    event_data_stereo = event_data.copy()

    # Apply the quality cuts
    if quality_cuts is not None:
        event_data_stereo.query(quality_cuts, inplace=True)

    # Extract stereo events
    event_data_stereo["multiplicity"] = event_data_stereo.groupby(group_index).size()
    event_data_stereo.query("multiplicity == [2, 3]", inplace=True)

    # Check the telescope combination types
    n_events_total = len(event_data_stereo.groupby(group_index).size())
    logger.info(f"\nIn total {n_events_total} stereo events are found:")

    # Loop over every telescope combination type
    for combo_type, (tel_combo, tel_ids) in enumerate(TEL_COMBINATIONS.items()):

        multiplicity = len(tel_ids)

        df_events = event_data_stereo.query(
            f"(tel_id == {tel_ids}) & (multiplicity == {multiplicity})"
        ).copy()

        # Here we recalculate the multiplicity and apply the cut again,
        # since with the above cut the events belonging to other
        # combination types are also extracted. For example, in case of
        # tel_id = [1, 2], the tel 1 events of the combination [1, 3]
        # and the tel 2 events of the combination [2, 3] remain in the
        # data frame, whose multiplicity will be recalculated as 1 and
        # so will be removed with the following cuts.

        df_events["multiplicity"] = df_events.groupby(group_index).size()
        df_events.query(f"multiplicity == {multiplicity}", inplace=True)

        n_events = len(df_events.groupby(group_index).size())
        percentage = 100 * n_events / n_events_total

        logger.info(
            f"\t{tel_combo} (type {combo_type}): "
            f"{int(n_events)} events ({np.round(percentage, 1)}%)"
        )

        event_data_stereo.loc[df_events.index, "combo_type"] = combo_type

    # Convert the combo_type from float to int
    event_data_stereo = event_data_stereo.astype({"combo_type": int})

    return event_data_stereo


def get_dl2_mean(event_data, weight_type="simple", group_index=["obs_id", "event_id"]):
    """
    Gets the mean DL2 parameters per shower event.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Data frame of shower events
    weight_type: str
        Type of the weights for the telescope-wise DL2 parameters -
        "simple" does not use any weights for calculations,
        "variance" uses the inverse of the RF variance, and
        "intensity" uses the linear-scale intensity parameter
    group_index: list
        Index to group telescope-wise events

    Returns
    -------
    event_data_mean: pandas.core.frame.DataFrame
        Data frame of the shower events with the mean DL2 parameters

    Raises
    ------
    ValueError
        If the input weight type is not known
    """

    is_simulation = "true_energy" in event_data.columns

    # Create a mean data frame
    if is_simulation:
        params = ["combo_type", "multiplicity", "true_energy", "true_alt", "true_az"]
    else:
        params = ["combo_type", "multiplicity", "timestamp"]

    event_data_mean = event_data[params].groupby(group_index).mean()
    event_data_mean = event_data_mean.astype({"combo_type": int, "multiplicity": int})

    event_indicies = event_data.groupby(group_index).mean().index.to_frame()
    event_data_mean = pd.concat([event_indicies, event_data_mean], axis=1)

    # Calculate the mean pointing direction
    pnt_az_mean, pnt_alt_mean = calculate_mean_direction(
        lon=event_data["pointing_az"], lat=event_data["pointing_alt"], unit="rad"
    )

    event_data_mean["pointing_alt"] = pnt_alt_mean
    event_data_mean["pointing_az"] = pnt_az_mean

    # Define the weights for the DL2 parameters
    if weight_type == "simple":
        energy_weights = 1
        direction_weights = None
        gammaness_weights = 1

    elif weight_type == "variance":
        energy_weights = 1 / event_data["reco_energy_var"]
        direction_weights = 1 / event_data["reco_disp_var"]
        gammaness_weights = 1 / event_data["gammaness_var"]

    elif weight_type == "intensity":
        energy_weights = event_data["intensity"]
        direction_weights = event_data["intensity"]
        gammaness_weights = event_data["intensity"]

    else:
        raise ValueError(f"Unknown weight type '{weight_type}'.")

    # Calculate the mean DL2 parameters
    df_events = pd.DataFrame(
        data={
            "energy_weight": energy_weights,
            "gammaness_weight": gammaness_weights,
            "weighted_log_energy": np.log10(event_data["reco_energy"]) * energy_weights,
            "weighted_gammaness": event_data["gammaness"] * gammaness_weights,
        }
    )

    group_sum = df_events.groupby(group_index).sum()

    log_energy_mean = group_sum["weighted_log_energy"] / group_sum["energy_weight"]
    gammaness_mean = group_sum["weighted_gammaness"] / group_sum["gammaness_weight"]

    reco_az_mean, reco_alt_mean = calculate_mean_direction(
        lon=event_data["reco_az"],
        lat=event_data["reco_alt"],
        unit="deg",
        weights=direction_weights,
    )

    event_data_mean["reco_energy"] = 10 ** log_energy_mean
    event_data_mean["reco_alt"] = reco_alt_mean
    event_data_mean["reco_az"] = reco_az_mean
    event_data_mean["gammaness"] = gammaness_mean

    # Transform the Alt/Az directions to the RA/Dec coordinate
    if not is_simulation:

        timestamps_mean = Time(
            event_data_mean["timestamp"].to_numpy(), format="unix", scale="utc"
        )

        pnt_ra_mean, pnt_dec_mean = transform_altaz_to_radec(
            alt=u.Quantity(pnt_alt_mean.to_numpy(), u.rad),
            az=u.Quantity(pnt_az_mean.to_numpy(), u.rad),
            obs_time=timestamps_mean,
        )

        reco_ra_mean, reco_dec_mean = transform_altaz_to_radec(
            alt=u.Quantity(reco_alt_mean.to_numpy(), u.deg),
            az=u.Quantity(reco_az_mean.to_numpy(), u.deg),
            obs_time=timestamps_mean,
        )

        event_data_mean["pointing_ra"] = pnt_ra_mean.to_value(u.deg)
        event_data_mean["pointing_dec"] = pnt_dec_mean.to_value(u.deg)
        event_data_mean["reco_ra"] = reco_ra_mean.to_value(u.deg)
        event_data_mean["reco_dec"] = reco_dec_mean.to_value(u.deg)

    return event_data_mean


def load_lst_dl1_data_file(input_file):
    """
    Loads a LST-1 DL1 data file and arranges the contents for the event
    coincidence with MAGIC.

    Parameters
    ----------
    input_file: str
        Path to an input LST-1 data file

    Returns
    -------
    event_data: pandas.core.frame.DataFrame
        Data frame of LST-1 events
    subarray: ctapipe.instrument.subarray.SubarrayDescription
        LST-1 subarray description
    """

    # Load the input file
    event_data = pd.read_hdf(
        input_file, key="dl1/event/telescope/parameters/LST_LSTCam"
    )

    # Add the trigger time differences of consecutive events
    event_data = add_delta_t_key(event_data)

    # Exclude interleaved events
    event_data.query(f"event_type == {EventType.SUBARRAY.value}", inplace=True)

    # Exclude poorly reconstructed events
    event_data.dropna(
        subset=["intensity", "time_gradient", "alt_tel", "az_tel"], inplace=True
    )

    # Exclude the events with duplicated event IDs.
    # ToBeChecked: if it still happens in recently processed data
    event_ids, counts = np.unique(event_data["event_id"], return_counts=True)

    if np.any(counts > 1):
        event_ids_dup = event_ids[counts > 1].tolist()
        event_data.query(f"event_id != {event_ids_dup}", inplace=True)

        logger.warning(
            f"WARNING: Excluded the events with duplicated event IDs: {event_ids_dup}"
        )

    logger.info(f"LST-1: {len(event_data)} events")

    # Rename the columns
    event_data.rename(
        columns={
            "obs_id": "obs_id_lst",
            "event_id": "event_id_lst",
            "delta_t": "time_diff",
            "alt_tel": "pointing_alt",
            "az_tel": "pointing_az",
            "leakage_pixels_width_1": "pixels_width_1",
            "leakage_pixels_width_2": "pixels_width_2",
            "leakage_intensity_width_1": "intensity_width_1",
            "leakage_intensity_width_2": "intensity_width_2",
            "time_gradient": "slope",
        },
        inplace=True,
    )

    # Change the units of parameters
    optics = pd.read_hdf(input_file, key="configuration/instrument/telescope/optics")
    focal_length = optics["equivalent_focal_length"][0]

    event_data["length"] = focal_length * np.tan(np.deg2rad(event_data["length"]))
    event_data["width"] = focal_length * np.tan(np.deg2rad(event_data["width"]))

    event_data["phi"] = np.rad2deg(event_data["phi"])
    event_data["psi"] = np.rad2deg(event_data["psi"])

    # Set the multi index
    event_data.set_index(["obs_id_lst", "event_id_lst", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    # Read the subarray description
    subarray = SubarrayDescription.from_hdf(input_file)

    if focal_length == NOMINAL_FOCLEN_LST:
        # Set the effective focal length to the subarray
        subarray.tel[1].optics.equivalent_focal_length = EFFECTIVE_FOCLEN_LST
        subarray.tel[1].camera.geometry.frame = CameraFrame(
            focal_length=EFFECTIVE_FOCLEN_LST
        )

    return event_data, subarray


def load_magic_dl1_data_files(input_dir):
    """
    Loads MAGIC DL1 data files for the event coincidence with LST-1.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input MAGIC DL1 data files are stored

    Returns
    -------
    event_data: pandas.core.frame.DataFrame
        Data frame of MAGIC events
    subarray: ctapipe.instrument.subarray.SubarrayDescription
        MAGIC subarray description

    Raises
    ------
    FileNotFoundError
        If any DL1 data files could not be found in the input directory
    """

    # Find the input files
    file_mask = f"{input_dir}/dl1_*.h5"

    input_files = glob.glob(file_mask)
    input_files.sort()

    if len(input_files) == 0:
        raise FileNotFoundError(
            "Could not find any DL1 data files in the input directory."
        )

    # Load the input files
    logger.info("\nThe following DL1 data files are found:")

    data_list = []

    for input_file in input_files:

        logger.info(input_file)

        df_events = pd.read_hdf(input_file, key="events/parameters")
        data_list.append(df_events)

    event_data = pd.concat(data_list)

    event_data.drop_duplicates(
        subset=["obs_id", "event_id", "tel_id"], keep=False, inplace=True
    )

    event_data.rename(
        columns={"obs_id": "obs_id_magic", "event_id": "event_id_magic"}, inplace=True
    )
    event_data.set_index(["obs_id_magic", "event_id_magic", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

    for tel_id in tel_ids:
        n_events = len(event_data.query(f"tel_id == {tel_id}"))
        logger.info(f"{TEL_NAMES[tel_id]}: {n_events} events")

    # Read the subarray description from the first input file, assuming
    # that it is consistent with the others
    subarray = SubarrayDescription.from_hdf(input_files[0])

    return event_data, subarray


def load_train_data_files(
    input_dir, offaxis_min=None, offaxis_max=None, true_event_class=None
):
    """
    Loads DL1-stereo data files and separates the shower events per
    telescope combination type for training RFs.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input DL1-stereo files are stored
    offaxis_min: str
        Minimum shower off-axis angle allowed, whose format should be
        acceptable by `astropy.units.quantity.Quantity`
    offaxis_max: str
        Maximum shower off-axis angle allowed, whose format should be
        acceptable by `astropy.units.quantity.Quantity`
    true_event_class: int
        True event class of the input events

    Returns
    -------
    data_train: dict
        Data frames of the shower events separated by the telescope
        combination types

    Raises
    ------
    FileNotFoundError
        If any DL1-stereo data files could not be found in the input
        directory
    """

    # Find the input files
    file_mask = f"{input_dir}/dl1_stereo_*.h5"

    input_files = glob.glob(file_mask)
    input_files.sort()

    if len(input_files) == 0:
        raise FileNotFoundError(
            "Could not find any DL1-stereo data files in the input directory."
        )

    # Load the input files
    logger.info("\nThe following DL1-stereo data files are found:")

    data_list = []

    for input_file in input_files:

        logger.info(input_file)

        df_events = pd.read_hdf(input_file, key="events/parameters")
        data_list.append(df_events)

    event_data = pd.concat(data_list)
    event_data.set_index(GROUP_INDEX_TRAIN, inplace=True)
    event_data.sort_index(inplace=True)

    logger.info(
        f"\nMinimum off-axis angle allowed: {offaxis_min}"
        f"\nMaximum off-axis angle allowed: {offaxis_max}"
    )

    if offaxis_min is not None:
        offaxis_min = u.Quantity(offaxis_min).to_value(u.deg)
        event_data.query(f"off_axis >= {offaxis_min}", inplace=True)

    if offaxis_max is not None:
        offaxis_max = u.Quantity(offaxis_max).to_value(u.deg)
        event_data.query(f"off_axis <= {offaxis_max}", inplace=True)

    if true_event_class is not None:
        event_data["true_event_class"] = true_event_class

    event_data["event_weight"] = EVENT_WEIGHT

    # Separate the events by the telescope combination types
    n_events_total = len(event_data.groupby(GROUP_INDEX_TRAIN).size())
    logger.info(f"\nIn total {n_events_total} stereo events are found:")

    data_train = {}

    for tel_combo, tel_ids in TEL_COMBINATIONS.items():

        multiplicity = len(tel_ids)

        df_events = event_data.query(
            f"(tel_id == {tel_ids}) & (multiplicity == {multiplicity})"
        ).copy()

        df_events["multiplicity"] = df_events.groupby(GROUP_INDEX_TRAIN).size()
        df_events.query(f"multiplicity == {multiplicity}", inplace=True)

        n_events = len(df_events.groupby(GROUP_INDEX_TRAIN).size())
        logger.info(f"\t{tel_combo}: {n_events} events")

        if n_events > 0:
            data_train[tel_combo] = df_events

    return data_train


def load_mc_dl2_data_file(input_file, quality_cuts, event_type, dl2_weight_type):
    """
    Loads a MC DL2 data file for creating the IRFs.

    Parameters
    ----------
    input_file: str
        Path to an input MC DL2 data file
    quality_cuts: str
        Quality cuts applied to the input events
    event_type: str
        Type of the events which will be used -
        "software" uses software coincident events,
        "software_only_3tel" uses only 3-tel combination events,
        "magic_only" uses only MAGIC-stereo combination events, and
        "hardware" uses all the telescope combination events
    dl2_weight_type: str
        Type of the weight for averaging telescope-wise DL2 parameters -
        "simple", "variance" or "intensity" are allowed

    Returns
    -------
    event_table: astropy.table.table.QTable
        Table of the MC DL2 events surviving the cuts
    pointing: astropy.units.quantity.Quantity
        Telescope mean pointing direction (Zenith, Azimuth)
    sim_info: pyirf.simulations.SimulatedEventsInfo
        Container of the simulation information

    Raises
    ------
    ValueError
        If the input event type is not known
    """

    # Load the input file
    df_events = pd.read_hdf(input_file, key="events/parameters")
    df_events.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    df_events.sort_index(inplace=True)

    df_events = get_stereo_events(df_events, quality_cuts)

    # Extract the events of the specified event type
    logger.info(f"\nExtracting the events of the '{event_type}' type...")

    if event_type == "software":
        df_events.query("(combo_type > 0) & (magic_stereo == True)", inplace=True)

    elif event_type == "software_only_3tel":
        df_events.query("combo_type == 3", inplace=True)

    elif event_type == "magic_only":
        df_events.query("combo_type == 0", inplace=True)

    elif event_type != "hardware":
        raise ValueError(f"Unknown event type '{event_type}'.")

    n_events = len(df_events.groupby(["obs_id", "event_id"]).size())
    logger.info(f"--> {n_events} stereo events")

    # Get the mean DL2 parameters
    df_dl2_mean = get_dl2_mean(df_events, dl2_weight_type)
    df_dl2_mean.reset_index(inplace=True)

    # Convert the pandas data frame to astropy QTable
    event_table = QTable.from_pandas(df_dl2_mean)

    event_table["pointing_alt"] *= u.rad
    event_table["pointing_az"] *= u.rad
    event_table["true_alt"] *= u.deg
    event_table["true_az"] *= u.deg
    event_table["reco_alt"] *= u.deg
    event_table["reco_az"] *= u.deg
    event_table["true_energy"] *= u.TeV
    event_table["reco_energy"] *= u.TeV

    event_table["theta"] = calculate_theta(
        events=event_table,
        assumed_source_az=event_table["true_az"],
        assumed_source_alt=event_table["true_alt"],
    )

    event_table["true_source_fov_offset"] = calculate_source_fov_offset(event_table)
    event_table["reco_source_fov_offset"] = calculate_source_fov_offset(
        event_table, prefix="reco"
    )

    # Get the telescope pointing direction
    pointing_zd = 90 * u.deg - event_table["pointing_alt"].mean().to(u.deg)
    pointing_az = event_table["pointing_az"].mean().to(u.deg)
    pointing = u.Quantity([pointing_zd.round(3), pointing_az.round(3)])

    # Get the simulation configuration
    sim_config = pd.read_hdf(input_file, key="simulation/config")

    n_total_showers = (
        sim_config["num_showers"][0]
        * sim_config["shower_reuse"][0]
        * len(np.unique(event_table["obs_id"]))
    )

    min_viewcone_radius = sim_config["min_viewcone_radius"][0] * u.deg
    max_viewcone_radius = sim_config["max_viewcone_radius"][0] * u.deg

    viewcone_diff = max_viewcone_radius - min_viewcone_radius

    if viewcone_diff < u.Quantity(0.001, u.deg):
        # Handle ring-wobble MCs as same as point-like MCs
        viewcone = 0 * u.deg
    else:
        viewcone = max_viewcone_radius

    sim_info = SimulatedEventsInfo(
        n_showers=n_total_showers,
        energy_min=u.Quantity(sim_config["energy_range_min"][0], u.TeV),
        energy_max=u.Quantity(sim_config["energy_range_max"][0], u.TeV),
        max_impact=u.Quantity(sim_config["max_scatter_range"][0], u.m),
        spectral_index=sim_config["spectral_index"][0],
        viewcone=viewcone,
    )

    return event_table, pointing, sim_info


def load_dl2_data_file(input_file, quality_cuts, event_type, dl2_weight_type):
    """
    Loads a DL2 data file for processing to DL3.

    Parameters
    ----------
    input_file: str
        Path to an input DL2 data file
    quality_cuts: str
        Quality cuts applied to the input events
    event_type: str
        Type of the events which will be used -
        "software" uses software coincident events,
        "software_only_3tel" uses only 3-tel combination events,
        "magic_only" uses only MAGIC-stereo combination events, and
        "hardware" uses all the telescope combination events
    dl2_weight_type: str
        Type of the weight for averaging telescope-wise DL2 parameters -
        "simple", "variance" or "intensity" are allowed

    Returns
    -------
    event_table: astropy.table.table.QTable
        Table of the MC DL2 events surviving the cuts
    on_time: astropy.units.quantity.Quantity
        ON time of the input data
    deadc: float
        Dead time correction factor
    """

    # Load the input file
    event_data = pd.read_hdf(input_file, key="events/parameters")
    event_data.set_index(["obs_id", "event_id", "tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    event_data = get_stereo_events(event_data, quality_cuts)

    # Extract the events of the specified event type
    logger.info(f"\nExtracting the events of the '{event_type}' type...")

    if event_type == "software":
        event_data.query("combo_type > 0", inplace=True)

    elif event_type == "software_only_3tel":
        event_data.query("combo_type == 3", inplace=True)

    elif event_type == "magic_only":
        event_data.query("combo_type == 0", inplace=True)

    elif event_type == "hardware":
        logger.warning(
            "WARNING: Please confirm that this type is correct for the input data, "
            "since the hardware trigger between LST-1 and MAGIC may NOT be used."
        )

    n_events = len(event_data.groupby(["obs_id", "event_id"]).size())
    logger.info(f"--> {n_events} stereo events")

    # Get the mean DL2 parameters
    df_dl2_mean = get_dl2_mean(event_data, dl2_weight_type)
    df_dl2_mean.reset_index(inplace=True)

    # Convert the pandas data frame to astropy QTable
    event_table = QTable.from_pandas(df_dl2_mean)

    event_table["pointing_alt"] *= u.rad
    event_table["pointing_az"] *= u.rad
    event_table["pointing_ra"] *= u.deg
    event_table["pointing_dec"] *= u.deg
    event_table["reco_alt"] *= u.deg
    event_table["reco_az"] *= u.deg
    event_table["reco_ra"] *= u.deg
    event_table["reco_dec"] *= u.deg
    event_table["reco_energy"] *= u.TeV
    event_table["timestamp"] *= u.s

    # Calculate the ON time
    time_diffs = np.diff(event_table["timestamp"])
    on_time = time_diffs[time_diffs < TIME_DIFF_UPLIM].sum()

    # Calculate the dead time correction factor. Here we use the
    # following equations to get the correction factor "deadc", where
    # <time_diff> is the mean of the trigger time differences of
    # consecutive events:

    # rate = 1 / (<time_diff> - dead_time)
    # deadc = 1 / (1 + rate * dead_time) = 1 - dead_time / <time_diff>

    logger.info("\nCalculating the dead time correction factor...")

    event_data.query(f"0 < time_diff < {TIME_DIFF_UPLIM.to_value(u.s)}", inplace=True)

    deadc_list = []

    # Calculate the LST-1 correction factor
    time_diffs_lst = event_data.query("tel_id == 1")["time_diff"]

    if len(time_diffs_lst) > 0:
        deadc_lst = 1 - DEAD_TIME_LST.to_value(u.s) / time_diffs_lst.mean()
        logger.info(f"LST-1: {deadc_lst.round(3)}")

        deadc_list.append(deadc_lst)

    # Calculate the MAGIC correction factor with one of the telescopes
    # whose number of events is larger than the other
    time_diffs_m1 = event_data.query("tel_id == 2")["time_diff"]
    time_diffs_m2 = event_data.query("tel_id == 3")["time_diff"]

    if len(time_diffs_m1) > len(time_diffs_m2):
        deadc_magic = 1 - DEAD_TIME_MAGIC.to_value(u.s) / time_diffs_m1.mean()
        logger.info(f"MAGIC(-I): {deadc_magic.round(3)}")
    else:
        deadc_magic = 1 - DEAD_TIME_MAGIC.to_value(u.s) / time_diffs_m2.mean()
        logger.info(f"MAGIC(-II): {deadc_magic.round(3)}")

    deadc_list.append(deadc_magic)

    # Calculate the total correction factor as the multiplicity of the
    # telescope-wise correction factors
    deadc = np.prod(deadc_list)
    logger.info(f"--> Total correction factor: {deadc.round(3)}")

    return event_table, on_time, deadc


def load_irf_files(input_dir_irf):
    """
    Loads input IRF data files and checks the consistency of their
    configurations for the IRF interpolation.

    Parameters
    ----------
    input_dir_irf: str
        Path to a directory where input IRF data files are stored

    Returns
    -------
    irf_data: dict
        Combined IRF data
    extra_header: dict
        Extra header of input IRF data files

    Raises
    ------
    FileNotFoundError
        If any IRF data files could not be found in the input directory
    RuntimeError
        If the configurations of the input IRFs are not consistent
    """

    extra_header = {
        "TELESCOP": [],
        "INSTRUME": [],
        "FOVALIGN": [],
        "QUAL_CUT": [],
        "EVT_TYPE": [],
        "DL2_WEIG": [],
        "IRF_OBST": [],
        "GH_CUT": [],
        "GH_EFF": [],
        "GH_MIN": [],
        "GH_MAX": [],
        "RAD_MAX": [],
        "TH_EFF": [],
        "TH_MIN": [],
        "TH_MAX": [],
    }

    irf_data = {
        "grid_points": [],
        "effective_area": [],
        "energy_dispersion": [],
        "psf_table": [],
        "background": [],
        "gh_cuts": [],
        "rad_max": [],
        "energy_bins": [],
        "fov_offset_bins": [],
        "migration_bins": [],
        "source_offset_bins": [],
        "bkg_fov_offset_bins": [],
    }

    # Find the input files
    irf_file_mask = f"{input_dir_irf}/irf_*.fits.gz"

    input_files_irf = glob.glob(irf_file_mask)
    input_files_irf.sort()

    n_input_files = len(input_files_irf)

    if n_input_files == 0:
        raise FileNotFoundError(
            "Could not find any IRF data files in the input directory."
        )

    # Loop over every IRF data file
    logger.info("\nThe following IRF data files are found:")

    for input_file in input_files_irf:

        logger.info(input_file)
        irf_hdus = fits.open(input_file)

        # Read the header
        header = irf_hdus["EFFECTIVE AREA"].header

        for key in extra_header.keys():
            if key in header:
                extra_header[key].append(header[key])

        # Read the pointing direction
        pointing_coszd = np.cos(np.deg2rad(header["PNT_ZD"]))
        pointing_az = np.deg2rad(header["PNT_AZ"])
        grid_point = [pointing_coszd, pointing_az]

        irf_data["grid_points"].append(grid_point)

        # Read the essential IRF data
        aeff_data = irf_hdus["EFFECTIVE AREA"].data[0]
        edisp_data = irf_hdus["ENERGY DISPERSION"].data[0]

        irf_data["effective_area"].append(aeff_data["EFFAREA"])
        irf_data["energy_dispersion"].append(edisp_data["MATRIX"].T)

        # Read the essential bins
        energy_bins = join_bin_lo_hi(aeff_data["ENERG_LO"], aeff_data["ENERG_HI"])
        fov_offset_bins = join_bin_lo_hi(aeff_data["THETA_LO"], aeff_data["THETA_HI"])
        migration_bins = join_bin_lo_hi(edisp_data["MIGRA_LO"], edisp_data["MIGRA_HI"])

        irf_data["energy_bins"].append(energy_bins)
        irf_data["fov_offset_bins"].append(fov_offset_bins)
        irf_data["migration_bins"].append(migration_bins)

        # Read additional IRF data and bins if they exist
        if "PSF" in irf_hdus:
            psf_data = irf_hdus["PSF"].data[0]
            source_offset_bins = join_bin_lo_hi(psf_data["RAD_LO"], psf_data["RAD_HI"])

            irf_data["psf_table"].append(psf_data["RPSF"].T)
            irf_data["source_offset_bins"].append(source_offset_bins)

        if "BACKGROUND" in irf_hdus:
            bkg_data = irf_hdus["BACKGROUND"].data[0]
            bkg_offset_bins = join_bin_lo_hi(bkg_data["THETA_LO"], bkg_data["THETA_HI"])

            irf_data["background"].append(bkg_data["BKG"].T)
            irf_data["bkg_fov_offset_bins"].append(bkg_offset_bins)

        if "GH_CUTS" in irf_hdus:
            ghcuts_data = irf_hdus["GH_CUTS"].data[0]
            irf_data["gh_cuts"].append(ghcuts_data["GH_CUTS"].T)

        if "RAD_MAX" in irf_hdus:
            radmax_data = irf_hdus["RAD_MAX"].data[0]
            irf_data["rad_max"].append(radmax_data["RAD_MAX"].T)

    # Check the IRF data consistency
    for key in irf_data.keys():

        n_irf_data = len(irf_data[key])

        if n_irf_data == 0:
            continue

        elif n_irf_data != n_input_files:
            raise RuntimeError(
                f"The number of '{key}' data (= {n_irf_data}) does not match "
                f"with that of the input IRF data files (= {n_input_files})."
            )

        elif "bins" in key:

            n_edges_unique = np.unique([len(bins) for bins in irf_data[key]])

            if len(n_edges_unique) > 1:
                raise RuntimeError(f"The number of edges of '{key}' does not match.")

            unique_bins = np.unique(irf_data[key], axis=0)

            if len(unique_bins) > 1:
                raise RuntimeError(f"The binning of '{key}' do not match.")

            else:
                # Set the unique bins
                irf_data[key] = unique_bins[0]

    # Check the header consistency
    for key in list(extra_header.keys()):

        n_values = len(extra_header[key])
        unique_values = np.unique(extra_header[key])

        if n_values == 0:
            # Remove the empty card
            extra_header.pop(key)

        elif (n_values != n_input_files) or len(unique_values) > 1:
            raise RuntimeError(f"The setting '{key}' does not match.")

        else:
            # Set the unique value
            extra_header[key] = unique_values[0]

    # Set units to the IRF data
    irf_data["effective_area"] *= u.m ** 2
    irf_data["psf_table"] *= u.Unit("sr-1")
    irf_data["background"] *= u.Unit("MeV-1 s-1 sr-1")
    irf_data["rad_max"] *= u.deg
    irf_data["energy_bins"] *= u.TeV
    irf_data["fov_offset_bins"] *= u.deg
    irf_data["source_offset_bins"] *= u.deg
    irf_data["bkg_fov_offset_bins"] *= u.deg

    # Convert the list to the numpy ndarray
    irf_data["grid_points"] = np.array(irf_data["grid_points"])
    irf_data["energy_dispersion"] = np.array(irf_data["energy_dispersion"])
    irf_data["gh_cuts"] = np.array(irf_data["gh_cuts"])
    irf_data["migration_bins"] = np.array(irf_data["migration_bins"])

    return irf_data, extra_header


def save_pandas_data_in_table(
    input_data, output_file, group_name, table_name, mode="w"
):
    """
    Saves a pandas data frame in a table.

    Parameters
    ----------
    input_data: pandas.core.frame.DataFrame
        Pandas data frame
    output_file: str
        Path to an output HDF file
    group_name: str
        Group name of the table
    table_name: str
        Name of the table
    mode: str
        Mode of saving the data if a file already exists at the path -
        "w" for overwriting the file with the new table, and
        "a" for appending the table to the file
    """

    values = [tuple(array) for array in input_data.to_numpy()]
    dtypes = np.dtype(list(zip(input_data.dtypes.index, input_data.dtypes.values)))

    data_array = np.array(values, dtype=dtypes)

    with tables.open_file(output_file, mode=mode) as f_out:
        f_out.create_table(group_name, table_name, createparents=True, obj=data_array)
