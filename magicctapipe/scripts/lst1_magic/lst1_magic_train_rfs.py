#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

Train the energy, direction and classifier RFs with the DL1-stereo data samples.
The RFs will be trained per telescope combination and per telescope type.
The number of gamma MC and background training samples will be automatically
adjusted to the same value when training the classifer RFs.

Usage:
$ python lst1_magic_train_rfs.py
--type-rf "classifier"
--input-file-gamma "./data/dl1_stereo/dl1_stereo_lst1_magic_gamma_40deg_90deg_off0.4_run1_to_400.h5"
--input-file-bkg "./data/dl1_stereo/dl1_stereo_lst1_magic_proton_40deg_90deg_run1_to_4000.h5"
--output-dir "./data/rfs"
--config-file "./config.yaml"
"""

import sys
import time
import yaml
import random
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from magicctapipe.utils import (
    EnergyRegressor,
    DirectionRegressor,
    EventClassifier
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

__all__ = [
    'train_rf_regressor',
    'train_rf_classifier',
]


def load_data(input_file, feature_names, event_class=None):

    tel_combinations = {
        'm1_m2': [2, 3],
        'lst1_m1': [1, 2],
        'lst1_m2': [1, 3],
        'lst1_m1_m2': [1, 2, 3]
    }

    data = pd.read_hdf(input_file, key='events/params')
    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data.dropna(subset=feature_names, inplace=True)
    data.sort_index(inplace=True)

    if event_class is not None:
        data['event_class'] = event_class

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

    telescope_ids = estimator.telescope_rfs.keys()

    for tel_id in telescope_ids:

        logger.info(f'\nTelescope {tel_id}')

        importances = estimator.telescope_rfs[tel_id].feature_importances_
        importances_sort = np.sort(importances)[::-1]

        indices = np.argsort(importances)[::-1]
        params_sort = np.array(estimator.feature_names)[indices]

        for param, importance in zip(params_sort, importances_sort):
            logger.info(f'{param}: {importance}')


def get_events_at_random(data, n_events):

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


def train_rf_regressor(type_rf, input_file, output_dir, config):

    config_rf = config[f'{type_rf}_regressor']
    logger.info(f'\nConfiguration for training the {type_rf} RF regressors:')

    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    logger.info(f'\nLoading the input data file:\n{input_file}')
    data_train = load_data(input_file, config_rf['features'])

    for tel_combo in data_train.keys():

        logger.info(f'\nTraining the {type_rf} RF regressors for "{tel_combo}" events...')
        data_train[tel_combo]['event_weight'] = 1

        if type_rf == 'energy':
            regressor = EnergyRegressor(config_rf['features'], config_rf['settings'])

        elif type_rf == 'direction':
            regressor = DirectionRegressor(config_rf['features'], config_rf['settings'])

        regressor.fit(data_train[tel_combo])

        logger.info('\nParameter importances:')
        check_importances(regressor)

        output_file = f'{output_dir}/{type_rf}_regressors_{tel_combo}.joblib'
        regressor.save(output_file)

    logger.info(f'\nOutput directory: {output_dir}')
    logger.info('\nDone.')


def train_rf_classifier(input_file_gamma, input_file_bkg, output_dir, config):

    config_rf = config['event_classifier']
    logger.info(f'\nConfiguration for training the event classifiers:')

    for key, value in config_rf.items():
        logger.info(f'{key}: {value}')

    logger.info(f'\nLoading the input gamma MC data file:\n{input_file_gamma}')
    data_gamma = load_data(input_file_gamma, config_rf['features'], event_class=0)

    logger.info(f'\nLoading the input background data file:\n{input_file_bkg}')
    data_bkg = load_data(input_file_bkg, config_rf['features'], event_class=1)

    tel_combinations = set(data_gamma.keys()) & set(data_bkg.keys())

    for tel_combo in sorted(tel_combinations, key=['m1_m2', 'lst1_m1', 'lst1_m2', 'lst1_m1_m2'].index):

        logger.info(f'\nTraining the event classifiers for "{tel_combo}" events...')

        n_events_gamma = len(data_gamma[tel_combo].groupby(['obs_id', 'event_id']).size())
        n_events_bkg = len(data_bkg[tel_combo].groupby(['obs_id', 'event_id']).size())

        if n_events_gamma > n_events_bkg:
            data_gamma[tel_combo] = get_events_at_random(data_gamma[tel_combo], n_events_bkg)
            n_events_gamma = len(data_gamma[tel_combo].groupby(['obs_id', 'event_id']).size())

        elif n_events_bkg > n_events_gamma:
            data_bkg[tel_combo] = get_events_at_random(data_bkg[tel_combo], n_events_gamma)
            n_events_bkg = len(data_bkg[tel_combo].groupby(['obs_id', 'event_id']).size())

        logger.info(f'--> n_events_gamma = {n_events_gamma}, n_events_bkg = {n_events_bkg}')

        data_gamma[tel_combo]['event_weight'] = 1
        data_bkg[tel_combo]['event_weight'] = 1

        data_train = data_gamma[tel_combo].append(data_bkg[tel_combo])

        classifier = EventClassifier(config_rf['features'], config_rf['settings'])
        classifier.fit(data_train)

        logger.info('\nParameter importances:')
        check_importances(classifier)

        output_file = f'{output_dir}/event_classifiers_{tel_combo}.joblib'
        classifier.save(output_file)

    logger.info(f'\nOutput directory: {output_dir}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--type-rf', '-t', dest='type_rf', type=str,
        help='Type of RF that will be trained, "energy", "direction" or "classifier".'
    )

    parser.add_argument(
        '--input-file-gamma', '-g', dest='input_file_gamma', type=str,
        help='Path to an input DL1-stereo gamma MC data file.'
    )

    parser.add_argument(
        '--input-file-bkg', '-b', dest='input_file_bkg', type=str, default=None,
        help='Path to an input DL1-stereo background data file.'
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='.',
        help='Path to a directory where the output RFs are saved.'
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a yaml configuration file.'
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    if np.any(args.type_rf == np.array(['energy', 'direction'])):
        train_rf_regressor(args.type_rf, args.input_file_gamma, args.output_dir, config)

    elif args.type_rf == 'classifier':
        train_rf_classifier(args.input_file_gamma, args.input_file_bkg, args.output_dir, config)

    else:
        logger.error(f'Unknown RF type "{args.type_rf}". Input "energy", "direction", "classifier". Exiting.\n')
        sys.exit()

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
