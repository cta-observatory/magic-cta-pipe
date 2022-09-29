#!/usr/bin/env python
# coding: utf-8

"""
This script trains energy, DISP regressors and event classifiers with
DL1-stereo events. The RFs are trained per telescope combination type
and per telescope. When training event classifiers, gamma or proton MC
events are randomly extracted so that the RFs are trained with the same
number of events by both types of primary particles.

Please specify the RF type that will be trained by using
"--train-energy", "--train-disp" and "--train-classifier" arguments.

If the "--use-unsigned" argument is given, the RFs will be trained with
unsigned features.

Before running the script, it would be better to merge input MC files
per telescope pointing direction with the following script:
"magic-cta-pipe/magicctapipe/scripts/lst1_magic/merge_hdf_files.py"

Usage:
$ python lst1_magic_train_rfs.py
--input-dir-gamma dl1_stereo/gamma
(--input-dir-proton dl1_stereo/proton)
(--output-dir rfs)
(--config-file config.yaml)
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
from magicctapipe.io import load_train_data_files
from magicctapipe.io.io import TEL_NAMES
from magicctapipe.reco import DispRegressor, EnergyRegressor, EventClassifier

__all__ = ["train_energy_regressor", "train_disp_regressor", "train_event_classifier"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# True event class of gamma and proton MC events
EVENT_CLASS_GAMMA = 0
EVENT_CLASS_PROTON = 1

# Set the random seed
random.seed(1000)


def train_energy_regressor(input_dir, output_dir, config, use_unsigned_features=False):
    """
    Trains energy regressors with gamma MC DL1-stereo events.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input gamma MC data files are stored
    output_dir: str
        Path to a directory where to save trained RFs
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    use_unsigned_features: bool
        If `True`, it uses unsigned features for training RFs
    """

    config_rf = config["energy_regressor"]

    gamma_offaxis = config_rf["gamma_offaxis"]

    # Load the input files
    logger.info(f"\nInput directory:\n{input_dir}")

    event_data_train = load_train_data_files(
        input_dir, gamma_offaxis["min"], gamma_offaxis["max"]
    )

    # Configure the energy regressor
    logger.info("\nRF settings:")
    for key, value in config_rf["settings"].items():
        logger.info(f"\t{key}: {value}")

    logger.info("\nFeatures:")
    for i_param, feature in enumerate(config_rf["features"], start=1):
        logger.info(f"\t{i_param}: {feature}")

    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    energy_regressor = EnergyRegressor(
        config_rf["settings"], config_rf["features"], use_unsigned_features
    )

    # Loop over every telescope combination type
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    for tel_combo, df_train in event_data_train.items():

        logger.info(f"\nEnergy regressors for the '{tel_combo}' type:")

        # Train the RFs
        energy_regressor.fit(df_train)

        # Check the feature importance
        for tel_id, telescope_rf in energy_regressor.telescope_rfs.items():

            logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
            importances = telescope_rf.feature_importances_

            for feature, importance in zip(energy_regressor.features, importances):
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


def train_disp_regressor(input_dir, output_dir, config, use_unsigned_features=False):
    """
    Trains DISP regressors with gamma MC DL1-stereo events.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input gamma MC data files are stored
    output_dir: str
        Path to a directory where to save trained RFs
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    use_unsigned_features: bool
        If `True`, it uses unsigned features for training RFs
    """

    config_rf = config["disp_regressor"]

    gamma_offaxis = config_rf["gamma_offaxis"]

    # Load the input files
    logger.info(f"\nInput directory:\n{input_dir}")

    event_data_train = load_train_data_files(
        input_dir, gamma_offaxis["min"], gamma_offaxis["max"]
    )

    # Configure the DISP regressor
    logger.info("\nRF settings:")
    for key, value in config_rf["settings"].items():
        logger.info(f"\t{key}: {value}")

    logger.info("\nFeatures:")
    for i_param, feature in enumerate(config_rf["features"], start=1):
        logger.info(f"\t{i_param}: {feature}")

    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    disp_regressor = DispRegressor(
        config_rf["settings"], config_rf["features"], use_unsigned_features
    )

    # Loop over every telescope combination type
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    for tel_combo, df_train in event_data_train.items():

        logger.info(f"\nDISP regressors for the '{tel_combo}' type:")

        # Train the RFs
        disp_regressor.fit(df_train)

        # Check the feature importance
        for tel_id, telescope_rf in disp_regressor.telescope_rfs.items():

            logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
            importances = telescope_rf.feature_importances_

            for feature, importance in zip(disp_regressor.features, importances):
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
    input_dir_gamma, input_dir_proton, output_dir, config, use_unsigned_features=False
):
    """
    Trains event classifiers with gamma and proton MC DL1-stereo events.

    Parameters
    ----------
    input_dir_gamma: str
        Path to a directory where input gamma MC data files are stored
    input_dir_proton: str
        Path to a directory where input proton MC data files are stored
    output_dir: str
        Path to a directory where to save trained RFs
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    use_unsigned_features: bool
        If `True`, it uses unsigned features for training RFs
    """

    config_rf = config["event_classifier"]

    gamma_offaxis = config_rf["gamma_offaxis"]

    # Load the input gamma MC data files
    logger.info(f"\nInput gamma MC directory:\n{input_dir_gamma}")

    event_data_gamma = load_train_data_files(
        input_dir_gamma, gamma_offaxis["min"], gamma_offaxis["max"], EVENT_CLASS_GAMMA
    )

    # Load the input proton MC data files
    logger.info(f"\nInput proton MC directory:\n{input_dir_proton}")

    event_data_proton = load_train_data_files(
        input_dir_proton, true_event_class=EVENT_CLASS_PROTON
    )

    # Configure the event classifier
    logger.info("\nRF settings:")
    for key, value in config_rf["settings"].items():
        logger.info(f"\t{key}: {value}")

    logger.info("\nFeatures:")
    for i_param, feature in enumerate(config_rf["features"], start=1):
        logger.info(f"\t{i_param}: {feature}")

    logger.info(f"\n\nUse unsigned features: {use_unsigned_features}")

    event_classifier = EventClassifier(
        config_rf["settings"], config_rf["features"], use_unsigned_features
    )

    # Loop over every telescope combination type
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_files = []

    common_combinations = set(event_data_gamma.keys()) & set(event_data_proton.keys())

    for tel_combo in sorted(common_combinations):

        logger.info(f"\nEvent classifiers for the '{tel_combo}' type:")

        df_gamma = event_data_gamma[tel_combo]
        df_proton = event_data_proton[tel_combo]

        # Adjust the number of training samples
        multi_indices_gamma = np.unique(df_gamma.index).tolist()
        multi_indices_proton = np.unique(df_proton.index).tolist()

        n_events_gamma = len(multi_indices_gamma)
        n_events_proton = len(multi_indices_proton)

        if n_events_gamma > n_events_proton:
            logger.info(
                f"Extracting {n_events_proton} out of {n_events_gamma} gamma events..."
            )
            multi_indices_random = pd.MultiIndex.from_tuples(
                tuples=random.sample(multi_indices_gamma, n_events_proton),
                names=df_gamma.index.names,
            )
            df_gamma = df_gamma.loc[multi_indices_random]
            df_gamma.sort_index(inplace=True)

        elif n_events_proton > n_events_gamma:
            logger.info(
                f"Extracting {n_events_gamma} out of {n_events_proton} proton events..."
            )
            multi_indices_random = pd.MultiIndex.from_tuples(
                tuples=random.sample(multi_indices_proton, n_events_gamma),
                names=df_proton.index.names,
            )
            df_proton = df_proton.loc[multi_indices_random]
            df_proton.sort_index(inplace=True)

        df_train = df_gamma.append(df_proton)

        # Train the RFS
        event_classifier.fit(df_train)

        # Check the feature importance
        for tel_id, telescope_rf in event_classifier.telescope_rfs.items():

            logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
            importances = telescope_rf.feature_importances_

            for feature, importance in zip(event_classifier.features, importances):
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
        "--input-dir-gamma",
        "-g",
        dest="input_dir_gamma",
        type=str,
        required=True,
        help="Path to a directory where input gamma MC data files are stored",
    )

    parser.add_argument(
        "--input-dir-proton",
        "-p",
        dest="input_dir_proton",
        type=str,
        help="Path to a directory where input proton MC data files are stored",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save trained RFs",
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
        "--train-energy",
        dest="train_energy",
        action="store_true",
        help="Train energy regressors",
    )

    parser.add_argument(
        "--train-disp",
        dest="train_disp",
        action="store_true",
        help="Train DISP regressors",
    )

    parser.add_argument(
        "--train-classifier",
        dest="train_classifier",
        action="store_true",
        help="Train event classifiers",
    )

    parser.add_argument(
        "--use-unsigned",
        dest="use_unsigned",
        action="store_true",
        help="Use unsigned features for training RFs",
    )

    args = parser.parse_args()

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    # Train RFs
    if args.train_energy:
        train_energy_regressor(
            args.input_dir_gamma, args.output_dir, config, args.use_unsigned
        )

    if args.train_disp:
        train_disp_regressor(
            args.input_dir_gamma, args.output_dir, config, args.use_unsigned
        )

    if args.train_classifier:
        train_event_classifier(
            input_dir_gamma=args.input_dir_gamma,
            input_dir_proton=args.input_dir_proton,
            output_dir=args.output_dir,
            config=config,
            use_unsigned_features=args.use_unsigned,
        )

    if not any([args.train_energy, args.train_disp, args.train_classifier]):
        raise ValueError(
            "The RF type is not specified. Please see the usage with '--help' option."
        )

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
