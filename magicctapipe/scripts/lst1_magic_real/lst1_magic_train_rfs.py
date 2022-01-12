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
--type-rf "all"
--input-file-gamma "./data/dl1_stereo/gamma_off0.4deg/merged/dl1_stereo_lst1_magic_gamma_40deg_90deg_off0.4_run1_to_400.h5"
--input-file-bkg "./data/dl1_stereo/proton/merged/dl1_stereo_lst1_magic_proton_40deg_90deg_run1_to_4000.h5"
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
    EnergyEstimator,
    DirectionEstimator,
    EventClassifier
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

__all__ = [
    'train_energy_rfs',
    'train_direction_rfs',
    'train_classifier_rfs'
]


def load_data(input_file, feature_names, event_class=None):

    tel_combinations = {
        'm1_m2': [2, 3], 
        'lst1_m1': [1, 2], 
        'lst1_m2': [1, 3],  
        'lst1_m1_m2': [1, 2, 3]   
    }

    data = pd.read_hdf(input_file, key='events/params')
    data.sort_index(inplace=True)

    if event_class != None:
        data['event_class'] = event_class

    data_return = {}

    for tel_combo, tel_ids in tel_combinations.items():

        df = data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        df.dropna(subset=feature_names, inplace=True)

        df['multiplicity'] = df.groupby(['obs_id', 'event_id']).size()
        df.query(f'multiplicity == {len(tel_ids)}', inplace=True)

        n_events = len(df.groupby(['obs_id', 'event_id']).size())
        logger.info(f'{tel_combo}: {n_events} events')

        if n_events > 0:
            data_return[tel_combo] = df
        
    return data_return


def check_importances(telescope_rfs, features):

    telescope_ids = telescope_rfs.keys()

    for tel_id in telescope_ids:

        logger.info(f'\nTelescope {tel_id}')

        importances = telescope_rfs[tel_id].feature_importances_
        importances_sort = np.sort(importances)[::-1]

        indices = np.argsort(importances)[::-1]
        params_sort = np.array(features)[indices]

        for param, importance in zip(params_sort, importances_sort):
            logger.info(f'{param}: {importance}')


def get_events_at_random(data, n_events):

    group = data.groupby(['obs_id', 'event_id']).size()
    indices = random.sample(range(len(group)), n_events)

    telescope_ids = np.unique(data.index.get_level_values('tel_id'))
    data_return = pd.DataFrame()

    for tel_id in telescope_ids: 

        df = data.query(f'tel_id == {tel_id}')
        df = df.iloc[indices]

        data_return = data_return.append(df)

    data_return.sort_index(inplace=True)

    return data_return


def train_energy_rfs(input_file, output_dir, config):

    config_rf = config['energy_rf']
    logger.info(f'\nConfiguration for training the energy RFs:\n{config_rf}')

    # --- load the input data file ---
    logger.info(f'\nLoading the input data file:\n{input_file}')
    data_train = load_data(input_file, config_rf['features'])

    # --- train the energy RFs ---
    for tel_combo in data_train.keys():

        logger.info(f'\nTraining the energy RFs for "{tel_combo}" events...')

        data_train[tel_combo]['event_weight'] = 1 

        energy_estimator = EnergyEstimator(config_rf['features'], config_rf['settings'])
        energy_estimator.fit(data_train[tel_combo])

        logger.info('\nParameter importances:')
        check_importances(energy_estimator.telescope_rfs, config_rf['features'])

        # --- save the trained RFs ---
        output_file = f'{output_dir}/energy_rfs_{tel_combo}.joblib'
        energy_estimator.save(output_file)

    logger.info(f'\nOutput directory: {output_dir}')
    logger.info('\nDone.')


def train_direction_rfs(input_file, output_dir, config):

    config_rf = config['direction_rf']
    logger.info(f'\nConfiguration for training the direction RF:\n{config_rf}')

    # --- load the input data file ---
    logger.info(f'\nLoading the input data file:\n{input_file}')
    data_train = load_data(input_file, config_rf['features'])

    # --- train the direction RFs ---
    subarray = pd.read_pickle(config['stereo_reco']['subarray'])

    for tel_combo in data_train.keys():

        logger.info(f'\nTraining the direction RF for "{tel_combo}" events...')

        data_train[tel_combo]['event_weight'] = 1 

        direction_estimator = DirectionEstimator(config_rf['features'], subarray.tels, config_rf['settings'])
        direction_estimator.fit(data_train[tel_combo])

        logger.info('\nParameter importances:')
        check_importances(direction_estimator.telescope_rfs, config_rf['features'])

        # --- save the trained RFs ---
        output_file = f'{output_dir}/direction_rfs_{tel_combo}.joblib'
        direction_estimator.save(output_file)

    logger.info(f'\nOutput directory: {output_dir}')
    logger.info('\nDone.')


def train_classifier_rfs(input_file_gamma, input_file_bkg, output_dir, config):

    config_rf = config['classifier_rf']
    logger.info(f'\nConfiguration for training the classifier RF:\n{config_rf}')

    # --- load the input data file ---
    logger.info(f'\nLoading the input gamma MC data file:\n{input_file_gamma}')
    data_gamma = load_data(input_file_gamma, config_rf['features'], event_class=0)

    logger.info(f'\nLoading the input background data file:\n{input_file_bkg}')
    data_bkg = load_data(input_file_bkg, config_rf['features'], event_class=1)

    # --- train the classifier RFs ---
    tel_combinations = set(data_gamma.keys()) & set(data_bkg.keys())

    for tel_combo in sorted(tel_combinations, key=['m1_m2', 'lst1_m1', 'lst1_m2', 'lst1_m1_m2'].index):

        logger.info(f'\nTraining the classifier RF for "{tel_combo}" events...')

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

        event_classifier = EventClassifier(config_rf['features'], config_rf['settings'])
        event_classifier.fit(data_train)

        logger.info('\nParameter importances:')
        check_importances(event_classifier.telescope_rfs, config_rf['features'])
        
        # --- save the trained RFs ---
        output_file = f'{output_dir}/classifier_rfs_{tel_combo}.joblib'
        event_classifier.save(output_file)

    logger.info(f'\nOutput directory: {output_dir}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--type-rf', '-t', dest='type_rf', type=str, 
        help='Type of RF that will be trained, "energy", "direction", "classifier" or "all".'  
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

    config = yaml.safe_load(open(args.config_file, 'r'))

    if args.type_rf == 'energy':
        train_energy_rfs(args.input_file_gamma, args.output_dir, config)

    elif args.type_rf == 'direction':
        train_direction_rfs(args.input_file_gamma, args.output_dir, config)

    elif args.type_rf == 'classifier':
        train_classifier_rfs(args.input_file_gamma, args.input_file_bkg, args.output_dir, config)

    elif args.type_rf == 'all':
        train_energy_rfs(args.input_file_gamma, args.output_dir, config)
        train_direction_rfs(args.input_file_gamma, args.output_dir, config)
        train_classifier_rfs(args.input_file_gamma, args.input_file_bkg, args.output_dir, config)

    else:
        logger.error(f'Unknown type of RF "{args.type_rf}". Input "energy", "direction", "classifier" or "all". Exiting.\n')
        sys.exit()

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
