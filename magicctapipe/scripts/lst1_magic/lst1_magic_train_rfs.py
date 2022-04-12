#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script trains energy, direction regressors and event classifiers with input DL1-stereo data.
The RFs are currently trained per telescope combination and per telescope type.
When training event classifiers, the number of gamma MC or background samples is automatically adjusted
so that the same number of gamma MC and background samples is used for training.

Since it requires one HDF file for each input, please merge all HDF files of training samples with "merge_hdf_files.py" in advance.
When running the script, please specify the type of RFs that you want to train by giving
"--train-energy", "--train-direction" or "--train-classifier" arguments.

Usage:
$ python lst1_magic_train_rfs.py
--input-file-gamma ./data/dl1_stereo/dl1_stereo_gamma_40deg_90deg_off0.4deg_LST-1_MAGIC_run1_to_run400.h5
--input-file-bkg ./data/dl1_stereo/dl1_stereo_proton_40deg_90deg_LST-1_MAGIC_run1_to_run4000.h5
--output-dir ./data/rfs
--config-file ./config.yaml
(--train-energy)
(--train-direction)
(--train-classifier)
"""

import time
import yaml
import random
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from ctapipe.instrument import SubarrayDescription
from magicctapipe.reco import (
    EnergyRegressor,
    DirectionRegressor,
    EventClassifier,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

tel_combinations = {
    'm1_m2': [2, 3],
    'lst1_m1': [1, 2],
    'lst1_m2': [1, 3],
    'lst1_m1_m2': [1, 2, 3],
}

event_class_gamma = 0
event_class_bkg = 1

__all__ = [
    'load_train_data_file',
    'check_importance',
    'get_events_at_random',
    'train_energy_regressor',
    'train_direction_regressor',
    'train_event_classifier',
]


def load_train_data_file(input_file, features, true_event_class=None):
    """
    Loads an input DL1-stereo data file for training samples
    and separates the events per telescope combination.

    Parameters
    ----------
    input_file: str
        Path to an input DL1-stereo data file
    features: list
        Parameters used for training RFs
    true_event_class: int
        True event class of input events

    Returns
    -------
    data_train: dict
        Pandas data frames of training samples
        per telescope combination
    """

    event_data = pd.read_hdf(input_file, key='events/parameters')
    event_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    event_data.sort_index(inplace=True)

    if true_event_class is not None:
        event_data['true_event_class'] = true_event_class

    # Here event weights are set to 1, meaning no weights.
    # ToBeChecked: what weights are best for training RFs?
    event_data['event_weight'] = 1

    data_train = {}

    for tel_combo, tel_ids in tel_combinations.items():

        df_events = event_data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})').copy()
        df_events.dropna(subset=features, inplace=True)

        df_events['multiplicity'] = df_events.groupby(['obs_id', 'event_id']).size()
        df_events.query(f'multiplicity == {len(tel_ids)}', inplace=True)

        n_events = len(df_events.groupby(['obs_id', 'event_id']).size())
        logger.info(f'{tel_combo}: {n_events} events')

        if n_events > 0:
            data_train[tel_combo] = df_events

    return data_train


def check_importance(estimator):
    """
    Checks the parameter importance of trained RFs:

    Parameters
    ----------
    estimator: magicctapipe.reco.estimator
        Trained regressor or classifier
    """

    tel_ids = estimator.telescope_rfs.keys()

    for tel_id in tel_ids:

        logger.info(f'Telescope {tel_id}')

        # Sort the parameters by the importance:
        importances = estimator.telescope_rfs[tel_id].feature_importances_
        importances_sort = np.sort(importances)[::-1]

        indices = np.argsort(importances)[::-1]
        params_sort = np.array(estimator.features)[indices]

        for param, importance in zip(params_sort, importances_sort):
            logger.info(f'\t{param}: {importance}')


def get_events_at_random(event_data, n_events):
    """
    Extracts a given number of events at random.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of training samples
    n_events:
        The number of events to be extracted at random

    Returns
    -------
    data_selected: pandas.core.frame.DataFrame
        Pandas data frame of the events randomly extracted
    """

    group_size = event_data.groupby(['obs_id', 'event_id']).size()
    indices = random.sample(range(len(group_size)), n_events)

    data_selected = pd.DataFrame()
    tel_ids = np.unique(event_data.index.get_level_values('tel_id'))

    for tel_id in tel_ids:

        df_events = event_data.query(f'tel_id == {tel_id}').copy()
        df_events = df_events.iloc[indices]

        data_selected = data_selected.append(df_events)

    data_selected.sort_index(inplace=True)

    return data_selected


def train_energy_regressor(input_file, output_dir, config):
    """
    Trains energy regressors with input gamma MC DL1-stereo data.

    Parameters
    ----------
    input_file: str
        Path to an input gamma MC DL1-stereo data file
    output_dir: str
        Path to a directory where to save trained regressors
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_rf = config['energy_regressor']

    logger.info('\nConfiguration for training energy regressors:')
    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    # Load the input file:
    logger.info('\nLoading the input file:')
    logger.info(input_file)

    data_train = load_train_data_file(input_file, config_rf['features'])

    # Configure the energy regressor:
    energy_regressor = EnergyRegressor(config_rf['features'], config_rf['settings'])

    # Train the regressors per telescope combination:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for tel_combo in data_train.keys():

        logger.info(f'\nTraining energy regressors for "{tel_combo}" events...')
        energy_regressor.fit(data_train[tel_combo])

        logger.info('\nParameter importance:')
        check_importance(energy_regressor)

        output_file = f'{output_dir}/energy_regressors_{tel_combo}.joblib'
        energy_regressor.save(output_file)

        logger.info('\nOutput file:')
        logger.info(output_file)


def train_direction_regressor(input_file, output_dir, config):
    """
    Trains direction regressors with input gamma MC DL1-stereo data.

    Parameters
    ----------
    input_file: str
        Path to an input gamma MC DL1-stereo data file
    output_dir: str
        Path to a directory where to save trained regressors
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_rf = config['direction_regressor']

    logger.info('\nConfiguration for training direction regressors:')
    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    # Load the input file:
    logger.info('\nLoading the input file:')
    logger.info(input_file)

    data_train = load_train_data_file(input_file, config_rf['features'])

    subarray = SubarrayDescription.from_hdf(input_file)
    tel_descriptions = subarray.tel

    # Configure the direction regressor:
    direction_regressor = DirectionRegressor(config_rf['features'], config_rf['settings'])

    # Train the regressors per telescope combination:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for tel_combo in data_train.keys():

        logger.info(f'\nTraining direction regressors for "{tel_combo}" events...')
        direction_regressor.fit(data_train[tel_combo], tel_descriptions)

        logger.info('\nParameter importance:')
        check_importance(direction_regressor)

        output_file = f'{output_dir}/direction_regressors_{tel_combo}.joblib'
        direction_regressor.save(output_file)

        logger.info('\nOutput file:')
        logger.info(output_file)


def train_event_classifier(input_file_gamma, input_file_bkg, output_dir, config):
    """
    Trains event classifiers with input gamma MC and background DL1-stereo data.

    Parameters
    ----------
    input_file_gamma: str
        Path to an input gamma MC DL1-stereo data file
    input_file_bkg: str
        Path to an input background DL1-stereo data file
    output_dir: str
        Path to a directory where to save trained classifiers
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_rf = config['event_classifier']

    logger.info('\nConfiguration for training event classifiers:')
    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    # Load the input files:
    logger.info('\nLoading the input gamma MC data file:')
    logger.info(input_file_gamma)

    data_gamma = load_train_data_file(input_file_gamma, config_rf['features'], event_class_gamma)

    logger.info('\nLoading the input background data file:')
    logger.info(input_file_bkg)

    data_bkg = load_train_data_file(input_file_bkg, config_rf['features'], event_class_bkg)

    # Configure the event classifier:
    event_classifier = EventClassifier(config_rf['features'], config_rf['settings'])

    # Train the classifiers per telescope combination:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for tel_combo in sorted(set(data_gamma.keys()) & set(data_bkg.keys())):

        logger.info(f'\nTraining event classifiers for "{tel_combo}" events...')

        n_events_gamma = len(data_gamma[tel_combo].groupby(['obs_id', 'event_id']).size())
        n_events_bkg = len(data_bkg[tel_combo].groupby(['obs_id', 'event_id']).size())

        # Adjust the number of training samples:
        if n_events_gamma > n_events_bkg:
            data_gamma[tel_combo] = get_events_at_random(data_gamma[tel_combo], n_events_bkg)
            n_events_gamma = len(data_gamma[tel_combo].groupby(['obs_id', 'event_id']).size())

        elif n_events_bkg > n_events_gamma:
            data_bkg[tel_combo] = get_events_at_random(data_bkg[tel_combo], n_events_gamma)
            n_events_bkg = len(data_bkg[tel_combo].groupby(['obs_id', 'event_id']).size())

        logger.info(f'--> n_events_gamma = {n_events_gamma}, n_events_bkg = {n_events_bkg}')

        data_train = data_gamma[tel_combo].append(data_bkg[tel_combo])
        event_classifier.fit(data_train)

        logger.info('\nParameter importance:')
        check_importance(event_classifier)

        output_file = f'{output_dir}/event_classifiers_{tel_combo}.joblib'
        event_classifier.save(output_file)

        logger.info('\nOutput file:')
        logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file-gamma', '-g', dest='input_file_gamma', type=str, required=True,
        help='Path to an input DL1-stereo gamma MC data file.',
    )

    parser.add_argument(
        '--input-file-bkg', '-b', dest='input_file_bkg', type=str, default=None,
        help='Path to an input DL1-stereo background data file.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save trained RFs.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.',
    )

    parser.add_argument(
        '--train-energy', dest='train_energy', action='store_true',
        help='Train energy regressors.',
    )

    parser.add_argument(
        '--train-direction', dest='train_direction', action='store_true',
        help='Train direction regressors.',
    )

    parser.add_argument(
        '--train-classifier', dest='train_classifier', action='store_true',
        help='Train event classifiers.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # Train RFs:
    if args.train_energy:
        train_energy_regressor(args.input_file_gamma, args.output_dir, config)

    if args.train_direction:
        train_direction_regressor(args.input_file_gamma, args.output_dir, config)

    if args.train_classifier:
        train_event_classifier(args.input_file_gamma, args.input_file_bkg, args.output_dir, config)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
