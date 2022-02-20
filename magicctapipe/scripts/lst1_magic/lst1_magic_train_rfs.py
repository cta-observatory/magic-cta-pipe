#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script trains energy, direction regressors and event classifiers with DL1-stereo data samples.
So far, they are trained per telescope combination and per telescope type.
When training the classifiers, the number of gamma and background training samples is automatically adjusted to the same number.

Usage:
$ python lst1_magic_train_rfs.py
--input-file-gamma ./data/dl1_stereo/dl1_stereo_gamma_40deg_90deg_off0.4deg_LST-1_MAGIC_run1_to_run400.h5
--input-file-bkg ./data/dl1_stereo/dl1_stereo_proton_40deg_90deg_LST-1_MAGIC_run1_to_run4000.h5
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
from ctapipe.instrument import SubarrayDescription
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

event_class_gamma = 0
event_class_bkg = 1

__all__ = [
    'train_rf_regressor',
    'train_rf_classifier',
]


def load_data(input_file, feature_names, event_class=None):
    """
    Loads an input DL1-stereo data file and separates
    the data telescope combination wise.

    Parameters
    ----------
    input_file: str
        Path to an input DL1-stereo data file
    feature_names: list
        Parameters used for training RFs
    event_class: int
        True event class of an input data

    Returns
    -------
    data_return: dict
        Pandas data frames separated by
        the telescope combinations
    """

    data = pd.read_hdf(input_file, key='events/params')
    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

    data.dropna(subset=feature_names, inplace=True)
    data.sort_index(inplace=True)

    if event_class is not None:
        data['event_class'] = event_class

    # So far, the event weight is set to 1, meaning no weights:
    data['event_weight'] = 1

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
    Checks the parameter importances of trained RFs:

    Parameters
    ----------
    estimator: EnergyRegressor, DirectionRegressor or EventClassifier
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
    Extracts a given number of events at random.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing training samples
    n_events:
        The number of events to be extracted at random

    Returns
    -------
    data_return: pandas.core.frame.DataFrame
        Pandas data frame of the events randomly extracted
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


def train_rf_regressor(input_file, output_dir, config, rf_type):
    """
    Trains RF regressors with input gamma MC samples.

    Parameters
    ----------
    input_file: str
        Path to an input gamma MC DL1-stereo data file
    output_dir: str
        Path to a directory where to save output trained regressors
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    rf_type: str
        Type of regressors, "energy" or "direction"
    """

    config_rf = config[f'{rf_type}_regressor']

    logger.info(f'\nConfiguration for training the {rf_type} RF regressors:')
    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    # Load the input file:
    logger.info('\nLoading the input file:')
    logger.info(input_file)

    data_train = load_data(input_file, config_rf['features'])
    subarray = SubarrayDescription.from_hdf(input_file)

    if rf_type == 'energy':
        regressor = EnergyRegressor(
            feature_names=config_rf['features'],
            rf_settings=config_rf['settings'],
        )

    elif rf_type == 'direction':
        regressor = DirectionRegressor(
            feature_names=config_rf['features'],
            rf_settings=config_rf['settings'],
            tel_descriptions=subarray.tel,
        )

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Train the regressors per telescope combination:
    for tel_combo in data_train.keys():

        logger.info(f'\nTraining the regressors for "{tel_combo}" events...')
        regressor.fit(data_train[tel_combo])

        logger.info('\nParameter importances:')
        check_importances(regressor)

        output_file = f'{output_dir}/{rf_type}_regressors_{tel_combo}.joblib'
        regressor.save(output_file)

        logger.info(f'Output file:')
        logger.info(output_file)

    logger.info('\nDone.')


def train_rf_classifier(input_file_gamma, input_file_bkg, output_dir, config):
    """
    Trains RF classifiers with input gamma MC and
    background DL1-stereo samples.

    Parameters
    ----------
    input_file_gamma: str
        Path to an input gamma MC DL1-stereo data file
    input_file_bkg: str
        Path to an input background DL1-stereo data file
    output_dir: str
        Path to a directory where to save output trained classifiers
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_rf = config['event_classifier']

    logger.info(f'\nConfiguration for training the event classifiers:')
    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    classifier = EventClassifier(config_rf['features'], config_rf['settings'])

    # Load the input files:
    logger.info('\nLoading the input gamma MC data file:')
    logger.info(input_file_gamma)

    data_gamma = load_data(input_file_gamma, config_rf['features'], event_class=event_class_gamma)

    logger.info('\nLoading the input background data file:')
    logger.info(input_file_bkg)

    data_bkg = load_data(input_file_bkg, config_rf['features'], event_class=event_class_bkg)

    # Check the telescope combinations common to both gamma and background samples:
    tel_combinations = set(data_gamma.keys()) & set(data_bkg.keys())

    # Train the classifiers per telescope combination:
    for tel_combo in sorted(tel_combinations):

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
        classifier.fit(data_train)

        logger.info('\nParameter importances:')
        check_importances(classifier)

        output_file = f'{output_dir}/event_classifiers_{tel_combo}.joblib'
        classifier.save(output_file)

        logger.info('\nOutput file:')
        logger.info(output_file)

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
        help='Path to a directory where to save output trained RFs.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.',
    )

    parser.add_argument(
        '--train-energy', dest='train_energy', action='store_true',
        help='Trains energy regressors.',
    )

    parser.add_argument(
        '--train-direction', dest='train_direction', action='store_true',
        help='Trains direction regressors.',
    )

    parser.add_argument(
        '--train-classifier', dest='train_classifier', action='store_true',
        help='Trains event classifiers.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    if args.train_energy:
        train_rf_regressor(args.input_file_gamma, args.output_dir, config, 'energy')

    if args.train_direction:
        train_rf_regressor(args.input_file_gamma, args.output_dir, config, 'direction')

    if args.train_classifier:
        train_rf_classifier(args.input_file_gamma, args.input_file_bkg, args.output_dir, config)

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
