#!/usr/bin/env python
# coding: utf-8

"""
This script trains energy, direction regressors and event classifiers
with DL1-stereo events. The RFs are currently trained per telescope
combination and per telescope type. When training event classifiers, the
number of gamma MC or proton MC events is adjusted so that the RFs are
trained with the same number of events.

Please specify the RF type that will be trained by using "--train-energy",
"--train-direction" or "--train-classifier" arguments.

If the "--use-unsigned" argument is given, the RFs will be trained with
unsigned features.

Since it allows only one input file for each event class, before running
this script it would be needed to merge DL1-stereo files with the script
"magicctapipe/scripts/lst1_magic/merge_hdf_files.py".

Usage:
$ python lst1_magic_train_rfs.py
--input-file-gamma ./dl1_stereo/dl1_stereo_gamma_40deg_90deg_run1_to_500.h5
--input-file-proton ./dl1_stereo/dl1_stereo_proton_40deg_90deg_run1_to_5000.h5
--output-dir ./rfs
--config-file ./config.yaml
(--train-energy)
(--train-direction)
(--train-classifier)
(--use-unsigned)
"""

import argparse
import logging
import time
from pathlib import Path

import yaml
from astropy import units as u
from ctapipe.instrument import SubarrayDescription
from magicctapipe.io import (
    check_feature_importance,
    get_events_at_random,
    load_train_data_file,
)
from magicctapipe.reco import DirectionRegressor, EnergyRegressor, EventClassifier

__all__ = [
    "train_energy_regressor",
    "train_direction_regressor",
    "train_event_classifier",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Use true Alt/Az directions for the pandas index in order to classify
# the events simulated by different telescope pointing directions but
# have the same observation ID
GROUP_INDEX = ["obs_id", "event_id", "true_alt", "true_az"]

EVENT_CLASS_GAMMA = 0
EVENT_CLASS_BKG = 1


def train_energy_regressor(input_file, output_dir, config, use_unsigned_features=False):
    """
    Trains energy regressors with gamma MC DL1-stereo events.

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

    offaxis_min = config_rf["gamma_offaxis"]["min"]
    offaxis_max = config_rf["gamma_offaxis"]["max"]

    if offaxis_min is not None:
        offaxis_min *= u.deg

    if offaxis_max is not None:
        offaxis_max *= u.deg

    # Load the input file
    logger.info(f"\nInput file:\n{input_file}")
    data_train = load_train_data_file(input_file, offaxis_min, offaxis_max)

    # Configure the energy regressor
    rf_settings = config_rf["settings"]
    features = config_rf["features"]

    logger.info("\nRF settings:")
    for key, value in rf_settings.items():
        logger.info(f"\t{key}: {value}")

    logger.info(f"\nFeatures:\n{features}")
    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    energy_regressor = EnergyRegressor(rf_settings, features, use_unsigned_features)

    # Train the regressors per telescope combination type
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
    Trains direction regressors with gamma MC DL1-stereo events.

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

    offaxis_min = config_rf["gamma_offaxis"]["min"]
    offaxis_max = config_rf["gamma_offaxis"]["max"]

    if offaxis_min is not None:
        offaxis_min *= u.deg

    if offaxis_max is not None:
        offaxis_max *= u.deg

    # Load the input file
    logger.info(f"\nInput file:\n{input_file}")
    data_train = load_train_data_file(input_file, offaxis_min, offaxis_max)

    subarray = SubarrayDescription.from_hdf(input_file)
    tel_descriptions = subarray.tel

    # Configure the direction regressor
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

    # Train the regressors per telescope combination type
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
    input_file_gamma, input_file_proton, output_dir, config, use_unsigned_features=False
):
    """
    Trains event classifiers with gamma and proton MC DL1-stereo events.

    Parameters
    ----------
    input_file_gamma: str
        Path to an input gamma MC DL1-stereo data file
    input_file_proton: str
        Path to an input proton MC DL1-stereo data file
    output_dir: str
        Path to a directory where to save trained RFs
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    use_unsigned_features: bool
        If `True`, it uses unsigned features for training RFs
    """

    config_rf = config["event_classifier"]

    offaxis_min = config_rf["gamma_offaxis"]["min"]
    offaxis_max = config_rf["gamma_offaxis"]["max"]

    if offaxis_min is not None:
        offaxis_min *= u.deg

    if offaxis_max is not None:
        offaxis_max *= u.deg

    # Load the input files
    logger.info(f"\nInput gamma MC data file:\n{input_file_gamma}")

    data_gamma = load_train_data_file(
        input_file_gamma, offaxis_min, offaxis_max, EVENT_CLASS_GAMMA
    )

    logger.info(f"\nInput proton MC data file:\n{input_file_proton}")

    data_proton = load_train_data_file(
        input_file_proton, true_event_class=EVENT_CLASS_BKG
    )

    # Configure the event classifier
    rf_settings = config_rf["settings"]
    features = config_rf["features"]

    logger.info("\nRF settings:")
    for key, value in rf_settings.items():
        logger.info(f"\t{key}: {value}")

    logger.info(f"\nFeatures:\n{features}")
    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    event_classifier = EventClassifier(rf_settings, features, use_unsigned_features)

    # Train the classifiers per telescope combination type
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    common_combinations = set(data_gamma.keys()) & set(data_proton.keys())

    for tel_combo in sorted(common_combinations):

        logger.info(f'\nTraining event classifiers for the "{tel_combo}" type...')

        n_events_gamma = len(data_gamma[tel_combo].groupby(GROUP_INDEX).size())
        n_events_proton = len(data_proton[tel_combo].groupby(GROUP_INDEX).size())

        # Adjust the number of training samples
        if n_events_gamma > n_events_proton:
            logger.info(
                f"--> The number of gamma MC events is adjusted to {n_events_proton}"
            )
            data_gamma[tel_combo] = get_events_at_random(
                data_gamma[tel_combo], n_events_proton
            )

        elif n_events_proton > n_events_gamma:
            logger.info(
                f"--> The number of proton MC events is adjusted to {n_events_gamma}"
            )
            data_proton[tel_combo] = get_events_at_random(
                data_proton[tel_combo], n_events_gamma
            )

        data_train = data_gamma[tel_combo].append(data_proton[tel_combo])
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
        "--input-file-proton",
        "-p",
        dest="input_file_proton",
        type=str,
        help="Path to an input DL1-stereo proton MC data file.",
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
            input_file_proton=args.input_file_proton,
            output_dir=args.output_dir,
            config=config,
            use_unsigned_features=args.use_unsigned,
        )

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
