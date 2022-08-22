#!/usr/bin/env python
# coding: utf-8

"""
This script trains energy, direction regressors and event classifiers
with input DL1-stereo data. The RFs are currently trained per telescope
combination and per telescope type. When training event classifiers, the
number of gamma MC or background samples is adjusted so that the RFs are
trained with the same number of samples.

Please specify the RF type that will be trained by using "--train-energy",
"--train-direction" or "--train-classifier" arguments.

If the "--use-unsigned" argument is given, the RFs will be trained with
unsigned features.

Since it allows only one input file for each event class, before running
this script it would be needed to merge the files of training samples
with "magicctapipe/scripts/lst1_magic/merge_hdf_files.py".

Usage:
$ python lst1_magic_train_rfs.py
--input-file-gamma ./dl1_stereo/dl1_stereo_gamma_40deg_90deg_run1_to_500.h5
--input-file-bkg ./dl1_stereo/dl1_stereo_proton_40deg_90deg_run1_to_5000.h5
--output-dir ./rfs
--config-file ./config.yaml
(--train-energy)
(--train-direction)
(--train-classifier)
(--use-unsigned)
"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from ctapipe.instrument import SubarrayDescription
from magicctapipe.reco import DirectionRegressor, EnergyRegressor, EventClassifier

__all__ = [
    "load_train_data_file",
    "check_feature_importance",
    "get_events_at_random",
    "train_energy_regressor",
    "train_direction_regressor",
    "train_event_classifier",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

EVENT_CLASS_GAMMA = 0
EVENT_CLASS_BKG = 1

# Here event weights are set to 1, meaning no weights.
# ToBeChecked: what weights are best for training RFs?
EVENT_WEIGHT = 1

# Use true Alt/Az directions for the pandas index in order to classify
# the events simulated by different telescope pointing directions but
# have the same observation ID:
GROUP_INDEX = ["obs_id", "event_id", "true_alt", "true_az"]

TEL_COMBINATIONS = {
    "m1_m2": [2, 3],
    "lst1_m1": [1, 2],
    "lst1_m2": [1, 3],
    "lst1_m1_m2": [1, 2, 3],
}


@u.quantity_input(offaxis_min=u.deg, offaxis_max=u.deg)
def load_train_data_file(
    input_file, offaxis_min=None, offaxis_max=None, true_event_class=None
):
    """
    Loads an input DL1-stereo data file as training samples and
    separates the shower events per telescope combination type.

    Parameters
    ----------
    input_file: str
        Path to an input DL1-stereo data file
    offaxis_min: astropy.units.quantity.Quantity
        Minimum shower off-axis angle allowed
    offaxis_max: astropy.units.quantity.Quantity
        Maximum shower off-axis angle allowed
    true_event_class: int
        True event class of input events

    Returns
    -------
    data_train: dict
        Pandas data frames of the events per telescope combination type
    """

    event_data = pd.read_hdf(input_file, key="events/parameters")
    event_data.set_index(GROUP_INDEX + ["tel_id"], inplace=True)
    event_data.sort_index(inplace=True)

    event_data["event_weight"] = EVENT_WEIGHT

    if offaxis_min is not None:
        logger.info(f"Minimum off-axis angle allowed: {offaxis_min}")
        event_data.query(f"off_axis >= {offaxis_min.to_value(u.deg)}", inplace=True)

    if offaxis_max is not None:
        logger.info(f"Maximum off-axis angle allowed: {offaxis_max}")
        event_data.query(f"off_axis <= {offaxis_max.to_value(u.deg)}", inplace=True)

    if true_event_class is not None:
        event_data["true_event_class"] = true_event_class

    data_train = {}

    logger.info("\nNumber of events per telescope combination type:")

    for tel_combo, tel_ids in TEL_COMBINATIONS.items():

        df_events = event_data.query(
            f"(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})"
        ).copy()

        df_events["multiplicity"] = df_events.groupby(GROUP_INDEX).size()
        df_events.query(f"multiplicity == {len(tel_ids)}", inplace=True)

        n_events = len(df_events.groupby(GROUP_INDEX).size())
        logger.info(f"\t{tel_combo}: {n_events} events")

        if n_events > 0:
            data_train[tel_combo] = df_events

    return data_train


def check_feature_importance(estimator):
    """
    Checks the feature importance of trained RFs.

    Parameters
    ----------
    estimator: magicctapipe.reco.estimator
        Trained regressor or classifier
    """

    logger.info("\nFeature importance:")

    features = np.array(estimator.features)
    tel_ids = estimator.telescope_rfs.keys()

    for tel_id in tel_ids:

        logger.info(f"\tTelescope {tel_id}:")
        telescope_rf = estimator.telescope_rfs[tel_id]

        importances = telescope_rf.feature_importances_
        importances_sort = np.sort(importances)[::-1]
        indices_sort = np.argsort(importances)[::-1]
        features_sort = features[indices_sort]

        for feature, importance in zip(features_sort, importances_sort):
            logger.info(f"\t\t{feature}: {importance}")


def get_events_at_random(event_data, n_random):
    """
    Extracts a given number of events at random.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events
    n_random:
        The number of events to be extracted at random

    Returns
    -------
    data_selected: pandas.core.frame.DataFrame
        Pandas data frame of the events randomly extracted
    """

    data_selected = pd.DataFrame()

    n_events = len(event_data.groupby(GROUP_INDEX).size())
    indices = random.sample(range(n_events), n_random)

    tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

    for tel_id in tel_ids:
        df_events = event_data.query(f"tel_id == {tel_id}").copy()
        data_selected = data_selected.append(df_events.iloc[indices])

    data_selected.sort_index(inplace=True)

    return data_selected


def train_energy_regressor(input_file, output_dir, config, use_unsigned_features=False):
    """
    Trains energy regressors with input gamma MC DL1-stereo data.

    Parameters
    ----------
    input_file: str
        Path to an input gamma MC DL1-stereo data file
    output_dir: str
        Path to a directory where to save trained RFs
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    use_unsigned_features: bool
        If `True`, it uses unsigned features for training RFs
    """

    config_rf = config["energy_regressor"]

    offaxis_min = config_rf["gamma_offaxis"].get("min")
    offaxis_max = config_rf["gamma_offaxis"].get("max")

    if offaxis_min is not None:
        offaxis_min *= u.deg

    if offaxis_max is not None:
        offaxis_max *= u.deg

    # Load the input file:
    logger.info(f"\nInput file:\n{input_file}")

    data_train = load_train_data_file(input_file, offaxis_min, offaxis_max)

    # Configure the energy regressor:
    rf_settings = config_rf["settings"]
    features = config_rf["features"]

    logger.info("\nRF settings:")
    for key, value in rf_settings.items():
        logger.info(f"\t{key}: {value}")

    logger.info(f"\nFeatures:\n{features}")
    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    energy_regressor = EnergyRegressor(rf_settings, features, use_unsigned_features)

    # Train the regressors per telescope combination type:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    for tel_combo in data_train.keys():

        logger.info(f'\nTraining energy regressors for the "{tel_combo}" type...')
        energy_regressor.fit(data_train[tel_combo])

        check_feature_importance(energy_regressor)

        if use_unsigned_features:
            output_file = f"{output_dir}/energy_regressors_{tel_combo}_unsigned.joblib"
        else:
            output_file = f"{output_dir}/energy_regressors_{tel_combo}.joblib"

        energy_regressor.save(output_file)
        output_files.append(output_file)

    logger.info("\nOutput file(s):")
    for output_file in output_files:
        logger.info(output_file)


def train_direction_regressor(
    input_file, output_dir, config, use_unsigned_features=False
):
    """
    Trains direction regressors with input gamma MC DL1-stereo data.

    Parameters
    ----------
    input_file: str
        Path to an input gamma MC DL1-stereo data file
    output_dir: str
        Path to a directory where to save trained RFs
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    use_unsigned_features: bool
        If `True`, it uses unsigned features for training RFs
    """

    config_rf = config["direction_regressor"]

    offaxis_min = config_rf["gamma_offaxis"].get("min")
    offaxis_max = config_rf["gamma_offaxis"].get("max")

    if offaxis_min is not None:
        offaxis_min *= u.deg

    if offaxis_max is not None:
        offaxis_max *= u.deg

    # Load the input file:
    logger.info(f"\nInput file:\n{input_file}")

    data_train = load_train_data_file(input_file, offaxis_min, offaxis_max)

    subarray = SubarrayDescription.from_hdf(input_file)
    tel_descriptions = subarray.tel

    # Configure the direction regressor:
    rf_settings = config_rf["settings"]
    features = config_rf["features"]

    logger.info("\nRF settings:")
    for key, value in rf_settings.items():
        logger.info(f"\t{key}: {value}")

    logger.info(f"\nFeatures:\n{features}")
    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    direction_regressor = DirectionRegressor(
        rf_settings, features, tel_descriptions, use_unsigned_features
    )

    # Train the regressors per telescope combination:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    for tel_combo in data_train.keys():

        logger.info(f'\nTraining direction regressors for the "{tel_combo}" type...')
        direction_regressor.fit(data_train[tel_combo])

        check_feature_importance(direction_regressor)

        if use_unsigned_features:
            output_file = (
                f"{output_dir}/direction_regressors_{tel_combo}_unsigned.joblib"
            )
        else:
            output_file = f"{output_dir}/direction_regressors_{tel_combo}.joblib"

        direction_regressor.save(output_file)
        output_files.append(output_file)

    logger.info("\nOutput file(s):")
    for output_file in output_files:
        logger.info(output_file)


def train_event_classifier(
    input_file_gamma, input_file_bkg, output_dir, config, use_unsigned_features=False
):
    """
    Trains event classifiers with input gamma MC and background
    DL1-stereo data.

    Parameters
    ----------
    input_file_gamma: str
        Path to an input gamma MC DL1-stereo data file
    input_file_bkg: str
        Path to an input background DL1-stereo data file
    output_dir: str
        Path to a directory where to save trained RFs
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    use_unsigned_features: bool
        If `True`, it uses unsigned features for training RFs
    """

    config_rf = config["event_classifier"]

    offaxis_min = config_rf["gamma_offaxis"].get("min")
    offaxis_max = config_rf["gamma_offaxis"].get("max")

    if offaxis_min is not None:
        offaxis_min *= u.deg

    if offaxis_max is not None:
        offaxis_max *= u.deg

    # Load the input files:
    logger.info(f"\nInput gamma MC data file:\n{input_file_gamma}")

    data_gamma = load_train_data_file(
        input_file_gamma, offaxis_min, offaxis_max, EVENT_CLASS_GAMMA
    )

    logger.info(f"\nInput background data file:\n{input_file_bkg}")

    data_bkg = load_train_data_file(input_file_bkg, true_event_class=EVENT_CLASS_BKG)

    # Configure the event classifier:
    rf_settings = config_rf["settings"]
    features = config_rf["features"]

    logger.info("\nRF settings:")
    for key, value in rf_settings.items():
        logger.info(f"\t{key}: {value}")

    logger.info(f"\nFeatures:\n{features}")
    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    event_classifier = EventClassifier(rf_settings, features, use_unsigned_features)

    # Train the classifiers per telescope combination:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    common_combinations = set(data_gamma.keys()) & set(data_bkg.keys())

    for tel_combo in sorted(common_combinations):

        logger.info(f'\nTraining event classifiers for the "{tel_combo}" type...')

        n_events_gamma = len(data_gamma[tel_combo].groupby(GROUP_INDEX).size())
        n_events_bkg = len(data_bkg[tel_combo].groupby(GROUP_INDEX).size())

        # Adjust the number of training samples:
        if n_events_gamma > n_events_bkg:
            logger.info(
                f"--> The number of gamma MC events is adjusted to {n_events_bkg}"
            )
            data_gamma[tel_combo] = get_events_at_random(
                data_gamma[tel_combo], n_events_bkg
            )

        elif n_events_bkg > n_events_gamma:
            logger.info(
                f"--> The number of background events is adjusted to {n_events_gamma}"
            )
            data_bkg[tel_combo] = get_events_at_random(
                data_bkg[tel_combo], n_events_gamma
            )

        data_train = data_gamma[tel_combo].append(data_bkg[tel_combo])
        event_classifier.fit(data_train)

        check_feature_importance(event_classifier)

        if use_unsigned_features:
            output_file = f"{output_dir}/event_classifiers_{tel_combo}_unsigned.joblib"
        else:
            output_file = f"{output_dir}/event_classifiers_{tel_combo}.joblib"

        event_classifier.save(output_file)
        output_files.append(output_file)

    logger.info("\nOutput file(s):")
    for output_file in output_files:
        logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file-gamma",
        "-g",
        dest="input_file_gamma",
        type=str,
        required=True,
        help="Path to an input DL1-stereo gamma MC data file.",
    )

    parser.add_argument(
        "--input-file-bkg",
        "-b",
        dest="input_file_bkg",
        type=str,
        help="Path to an input DL1-stereo background data file.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save trained RFs.",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config.yaml",
        help="Path to a yaml configuration file.",
    )

    parser.add_argument(
        "--train-energy",
        dest="train_energy",
        action="store_true",
        help="Train energy regressors.",
    )

    parser.add_argument(
        "--train-direction",
        dest="train_direction",
        action="store_true",
        help="Train direction regressors.",
    )

    parser.add_argument(
        "--train-classifier",
        dest="train_classifier",
        action="store_true",
        help="Train event classifiers.",
    )

    parser.add_argument(
        "--use-unsigned",
        dest="use_unsigned",
        action="store_true",
        help="Use unsigned features for training RFs.",
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Train RFs:
    if args.train_energy:
        train_energy_regressor(
            args.input_file_gamma, args.output_dir, config, args.use_unsigned
        )

    if args.train_direction:
        train_direction_regressor(
            args.input_file_gamma, args.output_dir, config, args.use_unsigned
        )

    if args.train_classifier:
        train_event_classifier(
            input_file_gamma=args.input_file_gamma,
            input_file_bkg=args.input_file_bkg,
            output_dir=args.output_dir,
            config=config,
            use_unsigned_features=args.use_unsigned,
        )

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
