#!/usr/bin/env python
# coding: utf-8

"""
This script trains energy, DISP regressors and event classifiers with
DL1-stereo events. The input events are separated by the telescope
combination types at first, and then telescope-wise RFs are trained for
every combination type. When training event classifiers, gamma or proton
MC events are randomly extracted so that the RFs are trained with the
same number of events by both types of primary particles.

Please specify the RF type that will be trained by using
`--train-energy`, `--train-disp` and `--train-classifier` arguments.

If the `--use-unsigned` argument is given, the RFs will be trained with
unsigned features.

Before running the script, it would be better to merge input MC files
per telescope pointing direction with the following script:
`magic-cta-pipe/magicctapipe/scripts/lst1_magic/merge_hdf_files.py`

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

Broader usage:
This script is called automatically from the script "RF.py".
If you want to analyse a target, this is the way to go. See this other script for more details.

"""

import argparse
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from magicctapipe.io import format_object, load_train_data_files_tel, telescope_combinations
from magicctapipe.io.io import GROUP_INDEX_TRAIN
from magicctapipe.reco import DispRegressor, EnergyRegressor, EventClassifier

__all__ = [
    "get_events_at_random",
    "train_energy_regressor",
    "train_disp_regressor",
    "train_event_classifier",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# True event class of gamma and proton MCs
EVENT_CLASS_GAMMA = 0
EVENT_CLASS_PROTON = 1

# Set the random seed
random.seed(1000)


def get_events_at_random(event_data, n_events_random):
    """
    Extracts a given number of shower events randomly.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Data frame of shower events
    n_events_random: int or float
        Number of events to be extracted randomly

    Returns
    -------
    event_data_selected: pandas.core.frame.DataFrame
        Data frame of the shower events extracted randomly
    """

    # Get the unique multi indices
    multi_indices_unique = np.unique(event_data.index).tolist()

    # Extract a given number of indices randomly
    multi_indices_random = pd.MultiIndex.from_tuples(
        tuples=random.sample(multi_indices_unique, n_events_random),
        names=event_data.index.names,
    )

    # Extract the events of the random indices
    event_data_selected = event_data.loc[multi_indices_random]
    event_data_selected.sort_index(inplace=True)

    return event_data_selected


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
        Configuration for the LST + MAGIC analysis
    use_unsigned_features: bool
        If `True`, it uses unsigned features for training RFs
    """

    config_rf = config["energy_regressor"]

    TEL_NAMES, _ = telescope_combinations(config)
    
    gamma_offaxis = config_rf["gamma_offaxis"]

    logger.info("\nGamma off-axis angles allowed:")
    logger.info(format_object(gamma_offaxis))

    # Load the input files
    logger.info(f"\nInput directory: {input_dir}")

    event_data_train = load_train_data_files_tel(
        input_dir, config, gamma_offaxis["min"], gamma_offaxis["max"]
    )

    # Configure the energy regressor
    logger.info("\nRF regressors:")
    logger.info(format_object(config_rf["settings"]))

    logger.info("\nFeatures:")
    logger.info(format_object(config_rf["features"]))

    logger.info(f"\nUse unsigned features: {use_unsigned_features}")
    
    logger.info(f"\nconfiguration file: {config}")
    logger.info(f'\nmc_tel_ids: {config["mc_tel_ids"]}')
    energy_regressor = EnergyRegressor(
        TEL_NAMES, config_rf["settings"], config_rf["features"], use_unsigned_features
    )

    # Create the output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Loop over every telescope combination type
    for tel_id, df_train in event_data_train.items():

        logger.info(f"\nEnergy regressors for the telescope ID '{tel_id}' :")

        # Train the RFs
        energy_regressor.fit(df_train)

        # Check the feature importance
        telescope_rf = energy_regressor.telescope_rfs[tel_id]

        importances = telescope_rf.feature_importances_.round(5)
        importances = dict(zip(energy_regressor.features, importances))

        logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
        logger.info(format_object(importances))

        # Save the trained RFs
        if use_unsigned_features:
            output_file = f"{output_dir}/energy_regressors_{tel_id}_unsigned.joblib"
        else:
            output_file = f"{output_dir}/energy_regressors_{tel_id}.joblib"

        energy_regressor.save(output_file)

        logger.info(f"\nOutput file: {output_file}")


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

    TEL_NAMES, _ = telescope_combinations(config)
    
    gamma_offaxis = config_rf["gamma_offaxis"]

    logger.info("\nGamma off-axis angles allowed:")
    logger.info(format_object(gamma_offaxis))

    # Load the input files
    logger.info(f"\nInput directory: {input_dir}")

    event_data_train = load_train_data_files_tel(
        input_dir, config, gamma_offaxis["min"], gamma_offaxis["max"]
    )

    # Configure the DISP regressor
    logger.info("\nRF regressors:")
    logger.info(format_object(config_rf["settings"]))

    logger.info("\nFeatures:")
    logger.info(format_object(config_rf["features"]))

    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    disp_regressor = DispRegressor(
        TEL_NAMES, config_rf["settings"], config_rf["features"], use_unsigned_features
    )

    # Create the output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Loop over every telescope combination type
    for tel_id, df_train in event_data_train.items():

        logger.info(f"\nDISP regressors for the telescope ID '{tel_id}':")

        # Train the RFs
        disp_regressor.fit(df_train)

        # Check the feature importance
        telescope_rf = disp_regressor.telescope_rfs[tel_id]

        importances = telescope_rf.feature_importances_.round(5)
        importances = dict(zip(disp_regressor.features, importances))

        logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
        logger.info(format_object(importances))

        # Save the trained RFs to an output file
        if use_unsigned_features:
            output_file = f"{output_dir}/disp_regressors_{tel_id}_unsigned.joblib"
        else:
            output_file = f"{output_dir}/disp_regressors_{tel_id}.joblib"

        disp_regressor.save(output_file)

        logger.info(f"\nOutput file: {output_file}")


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

    TEL_NAMES, _ = telescope_combinations(config)
    
    gamma_offaxis = config_rf["gamma_offaxis"]

    logger.info("\nGamma off-axis angles allowed:")
    logger.info(format_object(gamma_offaxis))

    # Load the input gamma MC data files
    logger.info(f"\nInput gamma MC directory: {input_dir_gamma}")

    event_data_gamma = load_train_data_files_tel(
        input_dir_gamma, config, gamma_offaxis["min"], gamma_offaxis["max"], EVENT_CLASS_GAMMA
    )

    # Load the input proton MC data files
    logger.info(f"\nInput proton MC directory: {input_dir_proton}")

    event_data_proton = load_train_data_files_tel(
        input_dir_proton, config, true_event_class=EVENT_CLASS_PROTON
    )

    # Configure the event classifier
    logger.info("\nRF classifiers:")
    logger.info(format_object(config_rf["settings"]))

    logger.info("\nFeatures:")
    logger.info(format_object(config_rf["features"]))

    logger.info(f"\nUse unsigned features: {use_unsigned_features}")

    event_classifier = EventClassifier(
        TEL_NAMES, config_rf["settings"], config_rf["features"], use_unsigned_features
    )

    # Create the output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Loop over every telescope combination type
    common_combinations = set(event_data_gamma.keys()) & set(event_data_proton.keys())

    for tel_id in sorted(common_combinations):

        logger.info(f"\nEvent classifiers for the telescope ID '{tel_id}':")

        df_gamma = event_data_gamma[tel_id]
        df_proton = event_data_proton[tel_id]

        # Adjust the number of training samples
        n_events_gamma = len(df_gamma.groupby(GROUP_INDEX_TRAIN).size())
        n_events_proton = len(df_proton.groupby(GROUP_INDEX_TRAIN).size())

        if n_events_gamma > n_events_proton:
            logger.info(f"Extracting {n_events_proton} gamma MC events...")
            df_gamma = get_events_at_random(df_gamma, n_events_proton)

        elif n_events_proton > n_events_gamma:
            logger.info(f"Extracting {n_events_gamma} proton MC events...")
            df_proton = get_events_at_random(df_proton, n_events_gamma)

        df_train = pd.concat([df_gamma, df_proton])

        # Train the RFs
        event_classifier.fit(df_train)

        # Check the feature importance
        telescope_rf = event_classifier.telescope_rfs[tel_id]

        importances = telescope_rf.feature_importances_.round(5)
        importances = dict(zip(event_classifier.features, importances))

        logger.info(f"\n{TEL_NAMES[tel_id]} feature importance:")
        logger.info(format_object(importances))

        # Save the trained RFs to an output file
        if use_unsigned_features:
            output_file = f"{output_dir}/event_classifiers_{tel_id}_unsigned.joblib"
        else:
            output_file = f"{output_dir}/event_classifiers_{tel_id}.joblib"

        event_classifier.save(output_file)

        logger.info(f"\nOutput file: {output_file}")


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
            "The RF type is not specified. Please see the usage with `--help`."
        )

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
