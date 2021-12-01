#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import sys
import time
import yaml
import glob
import warnings
import argparse
import pandas as pd
import numpy as np
from event_processing import EventClassifierPandas
from event_processing import DirectionEstimatorPandas
from event_processing import EnergyEstimatorPandas
from utils import merge_hdf_files


warnings.simplefilter('ignore')

__all__ = [
    'train_energy_rf',
    'train_direction_rf',
    'train_classifier_rf'
]


def get_weights(data):

    sin_edges = np.linspace(0, 1, num=51)
    alt_edges = np.lib.scimath.arcsin(sin_edges)
    intensity_edges = np.logspace(1, 6, num=51)

    hist, _, _ = np.histogram2d(data['alt_tel'], data['intensity'], bins=[alt_edges, intensity_edges])
    availability_hist = np.clip(hist, 0, 1)

    bins_alt = np.digitize(data['alt_tel'], alt_edges) - 1
    bins_intensity = np.digitize(data['intensity'], intensity_edges) - 1

    # --- treating the out-of-range events ---
    bins_alt[bins_alt == len(alt_edges) - 1] = len(alt_edges) - 2
    bins_intensity[bins_intensity == len(intensity_edges) - 1] = len(intensity_edges) - 2

    weights = 1 / hist[bins_alt, bins_intensity]
    weights *= availability_hist[bins_alt, bins_intensity]

    return weights


def get_weights_classifier(data_gamma, data_bkg):

    sin_edges = np.linspace(0, 1, num=51)
    alt_edges = np.lib.scimath.arcsin(sin_edges)
    intensity_edges = np.logspace(1, 6, num=51)
    
    hist_gamma, _, _ = np.histogram2d(data_gamma['alt_tel'], data_gamma['intensity'], bins=[alt_edges, intensity_edges])
    hist_bkg, _, _ = np.histogram2d(data_bkg['alt_tel'], data_bkg['intensity'], bins=[alt_edges, intensity_edges])

    availability_hist = np.clip(hist_gamma, 0, 1) * np.clip(hist_bkg, 0, 1)

    # --- weights for gamma-ray samples ---
    bins_alt_gamma = np.digitize(data_gamma['alt_tel'], alt_edges) - 1
    bins_intensity_gamma = np.digitize(data_gamma['intensity'], intensity_edges) - 1

    bins_alt_gamma[bins_alt_gamma == len(alt_edges) - 1] = len(alt_edges) - 2
    bins_intensity_gamma[bins_intensity_gamma == len(intensity_edges) - 1] = len(intensity_edges) - 2

    weights_gamma = 1 / hist_gamma[bins_alt_gamma, bins_intensity_gamma]
    weights_gamma *= availability_hist[bins_alt_gamma, bins_intensity_gamma]

    # --- weights for background samples ---
    bins_alt_bkg = np.digitize(data_bkg['alt_tel'], alt_edges) - 1
    bins_intensity_bkg = np.digitize(data_bkg['intensity'], intensity_edges) - 1

    bins_alt_bkg[bins_alt_bkg == len(alt_edges) - 1] = len(alt_edges) - 2
    bins_intensity_bkg[bins_intensity_bkg == len(intensity_edges) - 1] = len(intensity_edges) - 2

    weights_bkg = 1 / hist_bkg[bins_alt_bkg, bins_intensity_bkg]
    weights_bkg *= availability_hist[bins_alt_bkg, bins_intensity_bkg]

    return weights_gamma, weights_bkg


def load_data(data_path, event_class):

    paths_list = glob.glob(data_path)
    paths_list.sort()

    if len(paths_list) == 1:
        data = pd.read_hdf(data_path, key='events/params')

    elif len(paths_list) > 1:
        data = merge_hdf_files(data_path)
    
    data.sort_index(inplace=True)

    data['true_event_class'] = event_class

    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):
        n_events = len(data.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}: {n_events} events')

    return data


def train_energy_rf(data_path, config):

    print('\nConfiguration for training energy RF:\n{}'.format(config))

    # --- load the input gamma-ray data ---
    print(f'\nLoading the input data file: {data_path}')
    
    data_train = load_data(data_path, event_class=0)

    # --- get the event weights ---
    weights = get_weights(data_train)

    data_train['event_weight'] = weights

    # --- train RF ---
    energy_estimator = EnergyEstimatorPandas(config['features'], **config['settings'])

    print('\nTraining the energy RF...')

    energy_estimator.fit(data_train)

    # --- check the parameter importances ---
    print('\nParameter importances:')

    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):

        print(f'  {tel_name}:')

        importances = energy_estimator.telescope_regressors[tel_id].feature_importances_
        indices = np.argsort(importances)[::-1]

        importances_sort = np.sort(importances)[::-1]
        params_sort = np.array(config['features'])[indices]

        for i_par, param in enumerate(params_sort):
            print(f'    {param}: {importances_sort[i_par]}')

    return energy_estimator


def train_direction_rf(data_path, tel_discriptions, config):

    print('\nConfiguration for training direction RF:\n{}'.format(config))

    # --- load the input gamma-ray data ---
    print(f'\nLoading the input data file: {data_path}')
    
    data_train = load_data(data_path, event_class=0)

    # --- get the event weights ---
    weights = get_weights(data_train)

    data_train['event_weight'] = weights

    # --- train RF ---
    direction_estimator = DirectionEstimatorPandas(config['features'], tel_discriptions, **config['settings'])

    print('\nTraining the direction RF...')

    direction_estimator.fit(data_train)

    # --- check the parameter importances ---
    print('\nParameter importances (disp):')

    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):

        print(f'  {tel_name}:')

        importances = direction_estimator.telescope_rfs['disp'][tel_id].feature_importances_
        indices = np.argsort(importances)[::-1]

        importances_sort = np.sort(importances)[::-1]
        params_sort = np.array(config['features']['disp'])[indices]

        for i_par, param in enumerate(params_sort):
            print(f'    {param}: {importances_sort[i_par]}')

    return direction_estimator


def train_classifier_rf(data_path_gamma, data_path_bkg, config):

    print('\nConfiguration for training classifier RF:\n{}'.format(config))

    # --- load the input gamma-ray data ---
    print(f'\nLoading the input MC gamma-ray data file: {data_path_gamma}')
    
    data_gamma = load_data(data_path_gamma, event_class=0)

    # --- load the input background data ---
    print(f'\nLoading the input background data file: {data_path_bkg}')

    data_bkg = load_data(data_path_bkg, event_class=1)

    # --- get the event weights ---
    weights_gamma, weights_bkg = get_weights_classifier(data_gamma, data_bkg)

    data_gamma['event_weight'] = weights_gamma
    data_bkg['event_weight'] = weights_bkg

    # --- train RF ---
    data_train = data_gamma.append(data_bkg)

    class_estimator = EventClassifierPandas(config['features'], **config['settings'])

    print('\nTraining the classifier RF...')

    class_estimator.fit(data_train)

    # --- check the parameter importances ---
    print('\nParameter importances:')

    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):

        print(f'  {tel_name}:')

        importances = class_estimator.telescope_classifiers[tel_id].feature_importances_
        indices = np.argsort(importances)[::-1]

        importances_sort = np.sort(importances)[::-1]
        params_sort = np.array(config['features'])[indices]

        for i_par, param in enumerate(params_sort):
            print(f'    {param}: {importances_sort[i_par]}')

    return class_estimator


def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--type-rf', '-t', dest='type_rf', type=str, 
        help='Type of RF which will be trained, "energy", "direction" or "classifier"'  
    )

    arg_parser.add_argument(
        '--input-data-gamma', '-g', dest='input_data_gamma', type=str, 
        help='Path to input DL1+stereo gamma-ray data file(s) for training RF, e.g., dl1_stereo_gamma_run*.h5'
    )

    arg_parser.add_argument(
        '--input-data-bkg', '-b', dest='input_data_bkg', type=str, default=None,
        help='Path to  input DL1+stereo background data file(s) for training classifier RF, e.g., dl1_stereo_proton_run*.h5'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, 
        help='Path and name of an output data file of trained RF with joblib format, e.g., classifier_rf.joblib'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a config file with yaml format, e.g., config.yaml'
    )

    args = arg_parser.parse_args()

    # --- train RF ---
    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    if args.type_rf == 'energy':

        rf = train_energy_rf(args.input_data_gamma, config_lst1_magic['energy_rf'])

    elif args.type_rf == 'direction':

        subarray = pd.read_pickle(config_lst1_magic['stereo_reco']['subarray'])
        tel_discriptions = subarray.tels

        rf = train_direction_rf(args.input_data_gamma, tel_discriptions, config_lst1_magic['direction_rf'])

    elif args.type_rf == 'classifier':

        rf = train_classifier_rf(args.input_data_gamma, args.input_data_bkg, config_lst1_magic['classifier_rf'])

    else:
        
        print(f'Unknown type of RF "{args.type_rf}". Should be "energy", "direction" or "classifier". Exiting')
        sys.exit()

    # --- store the trained RF ---
    rf.save(args.output_data)

    print(f'\nOutput file: {args.output_data}')

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()