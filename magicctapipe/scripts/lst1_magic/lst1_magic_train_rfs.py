#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script trains energy, direction regressors and event classifiers with DL1-stereo data samples.
So far, the estimators are trained per telescope combination and per telescope type.
The number of gamma and background training samples is automatically adjusted to the same number when training the classifiers.

Usage:
$ python lst1_magic_train_rfs.py
--input-file-gamma ./data/dl1_stereo/dl1_stereo_lst1_magic_gamma_40deg_90deg_off0.4_run1_to_400.h5
--input-file-bkg ./data/dl1_stereo/dl1_stereo_lst1_magic_proton_40deg_90deg_run1_to_4000.h5
--output-dir ./data/rfs
--config-file ./config.yaml
--train-energy
--train-direction
--train-classifier
"""

import time
import yaml
import random
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from magicctapipe.reco import (
    EnergyRegressor,
    DirectionRegressor,
    EventClassifier,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

tel_combinations = {
    'm1_m2': [2, 3],
    'lst1_m1': [1, 2],
    'lst1_m2': [1, 3],
    'lst1_m1_m2': [1, 2, 3],
}

__all__ = [
    'train_rf_regressor',
    'train_rf_classifier',
]


def load_data(input_file, feature_names, event_class=None):
    """
    Loads an input DL1-stere file and separates the data
    telescope combination wise.

    Parameters
    ----------
    input_file: str
        Path to an input DL1-stereo data file
    feature_names: list
        Parameters used for training estimators
    event_class: int
        True event class of an input data,
        0 for gamma MC, 1 for backgrounds

    Returns
    -------
    data_return: dict
        pandas data frames separated by
        the telescope combinations
    """

    data = pd.read_hdf(input_file, key='events/params')
    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

    # Exclude the events whose training parameters are NaN:
    data.dropna(subset=feature_names, inplace=True)
    data.sort_index(inplace=True)

    # Set the true event class:
    if event_class is not None:
        data['event_class'] = event_class

    # Now the event weight is set to 1, meaning no weights:
    data['event_weight'] = 1

    # Separate the data telescope combination wise:
    data_return = {}

    for tel_combo, tel_ids in tel_combinations.items():

        df = data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        df['multiplicity'] = df.groupby(['obs_id', 'event_id']).size()
        df.query(f'multiplicity == {len(tel_ids)}', inplace=True)

        n_events = len(df.groupby(['obs_id', 'event_id']).size())
        logger.info(f'{tel_combo}: {n_events} events')

        if n_events > 0:
            data_return[tel_combo] = df

    return data_return


def check_importances(estimator):
    """
    Checks the parameter importances of trained estimators:

    estimator: EnergyEstimator, DirectionEstimator or EventClassifier
        trained estimators
    """

    telescope_ids = estimator.telescope_rfs.keys()

    for tel_id in telescope_ids:

        logger.info(f'\nTelescope {tel_id}')

        # Sort the parameters by the importances:
        importances = estimator.telescope_rfs[tel_id].feature_importances_
        importances_sort = np.sort(importances)[::-1]

        indices = np.argsort(importances)[::-1]
        params_sort = np.array(estimator.feature_names)[indices]

        for param, importance in zip(params_sort, importances_sort):
            logger.info(f'{param}: {importance}')


def get_events_at_random(data, n_events):
    """
    Extracts specified number of events at random.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        pandas data frame
    n_events:
        the number of events to extract at random

    Returns
    -------
    data_return: pandas.core.frame.DataFrame
        pandas data frame of the events randomly selected
    """

    group = data.groupby(['obs_id', 'event_id']).size()
    indices = random.sample(range(len(group)), n_events)

    data_return = pd.DataFrame()
    telescope_ids = np.unique(data.index.get_level_values('tel_id'))

    for tel_id in telescope_ids:
        df = data.query(f'tel_id == {tel_id}')
        df = df.iloc[indices]
        data_return = data_return.append(df)

    data_return.sort_index(inplace=True)

    return data_return


def train_rf_regressor(
    input_file,
    output_dir,
    config,
    rf_type,
):
    """
    Trains RF regressors with input gamma MC samples.

    Parameters
    ----------
    input_file: str
        Path to an input gamma MC DL1-stereo data file
    output_dir: str
        Path to a directory where to save output trained estimators
    config: dict
        Configuration for LST-1 + MAGIC analysis
    rf_type: str
        Type of estimators, "energy" or "direction"
    """

    config_rf = config[f'{rf_type}_regressor']

    logger.info(f'\nConfiguration for training the {rf_type} RF regressors:')
    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    # Load the input file:
    logger.info(f'\nLoading the input data file:\n{input_file}')
    data_train = load_data(input_file, config_rf['features'])

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Train the estimators per telescope combination:
    for tel_combo in data_train.keys():

        logger.info(f'\nTraining the {rf_type} RF regressors for "{tel_combo}" events...')

        if rf_type == 'energy':
            regressor = EnergyRegressor(config_rf['features'], config_rf['settings'])

        elif rf_type == 'direction':
            regressor = DirectionRegressor(config_rf['features'], config_rf['settings'])

        # Train the estimators:
        regressor.fit(data_train[tel_combo])

        logger.info('\nParameter importances:')
        check_importances(regressor)

        output_file = f'{output_dir}/{rf_type}_regressors_{tel_combo}.joblib'
        regressor.save(output_file)

        logger.info(f'\nOutput file:\n{output_file}')

    logger.info('\nDone.')


def train_rf_classifier(
    input_file_gamma,
    input_file_bkg,
    output_dir,
    config,
):
    """
    Trains RF classifiers with input gamma MC and
    background DL1-stere data files.

    Parameters
    ----------
    input_file_gamma: str
        Path to an input gamma MC DL1-stereo data file
    input_file_bkg: str
        Path to an input background DL1-stereo data file
    output_dir: str
        Path to a directory where to save output trained estimators
    config: dict
        Configuration for LST-1 + MAGIC analysis
    """

    config_rf = config['event_classifier']

    logger.info(f'\nConfiguration for training the event classifiers:')
    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    # Load the input files:
    logger.info(f'\nLoading the input gamma MC data file:\n{input_file_gamma}')
    data_gamma = load_data(input_file_gamma, config_rf['features'], event_class=0)

    logger.info(f'\nLoading the input background data file:\n{input_file_bkg}')
    data_bkg = load_data(input_file_bkg, config_rf['features'], event_class=1)

    # Check the telescope combinations common to both gamma and background samples:
    tel_combinations = set(data_gamma.keys()) & set(data_bkg.keys())

    # Train the estimators per telescope combination:
    for tel_combo in tel_combinations:

        logger.info(f'\nTraining the event classifiers for "{tel_combo}" events...')

        n_events_gamma = len(data_gamma[tel_combo].groupby(['obs_id', 'event_id']).size())
        n_events_bkg = len(data_bkg[tel_combo].groupby(['obs_id', 'event_id']).size())

        # Adjust the number of samples:
        if n_events_gamma > n_events_bkg:
            data_gamma[tel_combo] = get_events_at_random(data_gamma[tel_combo], n_events_bkg)
            n_events_gamma = len(data_gamma[tel_combo].groupby(['obs_id', 'event_id']).size())

        elif n_events_bkg > n_events_gamma:
            data_bkg[tel_combo] = get_events_at_random(data_bkg[tel_combo], n_events_gamma)
            n_events_bkg = len(data_bkg[tel_combo].groupby(['obs_id', 'event_id']).size())

        logger.info(f'--> n_events_gamma = {n_events_gamma}, n_events_bkg = {n_events_bkg}')
        data_train = data_gamma[tel_combo].append(data_bkg[tel_combo])

        classifier = EventClassifier(config_rf['features'], config_rf['settings'])

        # Train the classifiers:
        classifier.fit(data_train)

        logger.info('\nParameter importances:')
        check_importances(classifier)

        output_file = f'{output_dir}/event_classifiers_{tel_combo}.joblib'
        classifier.save(output_file)

        logger.info(f'\nOutput file:\n{output_file}')

    logger.info('\nDone.')


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
        help='Path to a directory where to save output trained estimators.',
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

    if args.train_energy:
        train_rf_regressor(
            input_file=args.input_file_gamma,
            output_dir=args.output_dir,
            config=config,
            rf_type='energy',
        )

    if args.train_direction:
        train_rf_regressor(
            input_file=args.input_file_gamma,
            output_dir=args.output_dir,
            config=config,
            rf_type='direction',
        )

    if args.train_classifier:
        train_rf_classifier(
            input_file_gamma=args.input_file_gamma,
            input_file_bkg=args.input_file_bkg,
            output_dir=args.output_dir,
            config=config,
        )

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
