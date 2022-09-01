#!/usr/bin/env python
# coding: utf-8

"""
This script trains energy, DISP regressors and event classifiers with
DL1-stereo events. The RFs are trained per telescope combination and per
telescope type. When training event classifiers, the number of gamma MC
or proton MC events is adjusted so that the RFs are trained with the
same number of events.

Please specify the RF type that will be trained by using
"--train-energy", "--train-disp" or "--train-classifier" arguments.

If the "--use-unsigned" argument is given, the RFs will be trained with
unsigned features.

Since it allows only one input file for each event class, before running
this script it would be needed to merge DL1-stereo files with the script
"magicctapipe/scripts/lst1_magic/merge_hdf_files.py".

Usage:
$ python lst1_magic_train_rfs.py
--input-file-gamma dl1_stereo/dl1_stereo_gamma_40deg_90deg.h5
--input-file-proton dl1_stereo/dl1_stereo_proton_40deg_90deg.h5
--output-dir rfs
--config-file config.yaml
(--train-energy)
(--train-disp)
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
from magicctapipe.io import TEL_NAMES, load_train_data_file
from magicctapipe.reco import DispRegressor, EnergyRegressor, EventClassifier

__all__ = ["train_energy_regressor", "train_disp_regressor", "train_event_classifier"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# True event class of gamma and proton MC events
EVENT_CLASS_GAMMA = 0
EVENT_CLASS_PROTON = 1


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

    # Load the input file
    logger.info(f"\nInput file:\n{input_file}")

    offaxis_min = config_rf["gamma_offaxis"]["min"]
    offaxis_max = config_rf["gamma_offaxis"]["max"]

    if offaxis_min is not None:
        offaxis_min = u.Quantity(offaxis_min)
        logger.info(f"Minimum off-axis angle allowed: {offaxis_min}")

    if offaxis_max is not None:
        offaxis_max = u.Quantity(offaxis_max)
        logger.info(f"Maximum off-axis angle allowed: {offaxis_max}")

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

    # Train the RFs per telescope combination type
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    for tel_combo in data_train.keys():

        logger.info(f"\nEnergy regressors for the '{tel_combo}' type:")
        energy_regressor.fit(data_train[tel_combo])

        # Check the feature importance
        for tel_id, telescope_rf in energy_regressor.telescope_rfs.items():

            logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
            importances = telescope_rf.feature_importances_

            for feature, importance in zip(features, importances):
                logger.info(f"\t{feature}: {importance.round(5)}")

        # Save the trained RFs to an output file
        if use_unsigned_features:
            output_file = f"{output_dir}/energy_regressors_{tel_combo}_unsigned.joblib"
        else:
            output_file = f"{output_dir}/energy_regressors_{tel_combo}.joblib"

        energy_regressor.save(output_file)
        output_files.append(output_file)

    logger.info("\nOutput file(s):")
    for output_file in output_files:
        logger.info(output_file)


def train_disp_regressor(input_file, output_dir, config, use_unsigned_features=False):
    """
    Trains DISP regressors with gamma MC DL1-stereo events.

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

    config_rf = config["disp_regressor"]

    # Load the input file
    logger.info(f"\nInput file:\n{input_file}")

    offaxis_min = config_rf["gamma_offaxis"]["min"]
    offaxis_max = config_rf["gamma_offaxis"]["max"]

    if offaxis_min is not None:
        offaxis_min = u.Quantity(offaxis_min)
        logger.info(f"Minimum off-axis angle allowed: {offaxis_min}")

    if offaxis_max is not None:
        offaxis_max = u.Quantity(offaxis_max)
        logger.info(f"Maximum off-axis angle allowed: {offaxis_max}")

    data_train = load_train_data_file(input_file, offaxis_min, offaxis_max)

    # Configure the DISP regressor
    rf_settings = config_rf["settings"]
    features = config_rf["features"]

    logger.info("\nRF settings:")
    for key, value in rf_settings.items():
        logger.info(f"\t{key}: {value}")

    logger.info(f"\nFeatures:\n{features}")
    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    disp_regressor = DispRegressor(rf_settings, features, use_unsigned_features)

    # Train the RFs per telescope combination type
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    for tel_combo in data_train.keys():

        logger.info(f"\nDISP regressors for the '{tel_combo}' type:")
        disp_regressor.fit(data_train[tel_combo])

        # Check the feature importance
        for tel_id, telescope_rf in disp_regressor.telescope_rfs.items():

            logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
            importances = telescope_rf.feature_importances_

            for feature, importance in zip(features, importances):
                logger.info(f"\t{feature}: {importance.round(5)}")

        # Save the trained RFs to an output file
        if use_unsigned_features:
            output_file = f"{output_dir}/disp_regressors_{tel_combo}_unsigned.joblib"
        else:
            output_file = f"{output_dir}/disp_regressors_{tel_combo}.joblib"

        disp_regressor.save(output_file)
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

    # Load the input gamma MC data file
    logger.info(f"\nInput gamma MC data file:\n{input_file_gamma}")

    offaxis_min = config_rf["gamma_offaxis"]["min"]
    offaxis_max = config_rf["gamma_offaxis"]["max"]

    if offaxis_min is not None:
        offaxis_min = u.Quantity(offaxis_min)
        logger.info(f"Minimum off-axis angle allowed: {offaxis_min}")

    if offaxis_max is not None:
        offaxis_max = u.Quantity(offaxis_max)
        logger.info(f"Maximum off-axis angle allowed: {offaxis_max}")

    data_gamma = load_train_data_file(
        input_file_gamma, offaxis_min, offaxis_max, EVENT_CLASS_GAMMA
    )

    # Load the input proton MC data file
    logger.info(f"\nInput proton MC data file:\n{input_file_proton}")

    data_proton = load_train_data_file(
        input_file_proton, true_event_class=EVENT_CLASS_PROTON
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

    # Train the RFs per telescope combination type
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    common_combinations = set(data_gamma.keys()) & set(data_proton.keys())

    for tel_combo in sorted(common_combinations):

        logger.info(f"\nEvent classifiers for the '{tel_combo}' type:")

        multi_indices_gamma = np.unique(data_gamma[tel_combo].index)
        multi_indices_proton = np.unique(data_proton[tel_combo].index)

        n_events_gamma = len(multi_indices_gamma)
        n_events_proton = len(multi_indices_proton)

        # Adjust the number of training samples
        if n_events_gamma > n_events_proton:
            logger.info(f"Extracting {n_events_proton} events from the gamma MCs...")

            multi_indices_random = pd.MultiIndex.from_tuples(
                tuples=random.sample(multi_indices_gamma.tolist(), n_events_proton),
                names=data_gamma[tel_combo].index.names,
            )
            data_gamma[tel_combo] = data_gamma[tel_combo].loc[multi_indices_random]
            data_gamma[tel_combo].sort_index(inplace=True)

        elif n_events_proton > n_events_gamma:
            logger.info(f"Extracting {n_events_gamma} events from the proton MCs...")

            multi_indices_random = pd.MultiIndex.from_tuples(
                tuples=random.sample(multi_indices_proton.tolist(), n_events_gamma),
                names=data_proton[tel_combo].index.names,
            )
            data_proton[tel_combo] = data_proton[tel_combo].loc[multi_indices_random]
            data_proton[tel_combo].sort_index(inplace=True)

        data_train = data_gamma[tel_combo].append(data_proton[tel_combo])
        event_classifier.fit(data_train)

        # Check the feature importance
        for tel_id, telescope_rf in event_classifier.telescope_rfs.items():

            logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
            importances = telescope_rf.feature_importances_

            for feature, importance in zip(features, importances):
                logger.info(f"\t{feature}: {importance.round(5)}")

        # Save the trained RFs to an output file
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
        "--train-disp",
        dest="train_disp",
        action="store_true",
        help="Train DISP regressors.",
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

    if args.train_disp:
        train_disp_regressor(
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
