#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import time
import yaml
import warnings
import argparse
import pandas as pd
import numpy as np
from event_processing import EnergyEstimatorPandas

warnings.simplefilter('ignore')

start_time = time.time()

def get_weights(mc_data, alt_edges, intensity_edges):
    mc_hist, _, _ = np.histogram2d(mc_data['alt_tel'], mc_data['intensity'], bins=[alt_edges, intensity_edges])
    availability_hist = np.clip(mc_hist, 0, 1)

    # --- MC weights ---
    mc_alt_bins = np.digitize(mc_data['alt_tel'], alt_edges) - 1
    mc_intensity_bins = np.digitize(mc_data['intensity'], intensity_edges) - 1

    # Treating the out-of-range events
    mc_alt_bins[mc_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    mc_intensity_bins[mc_intensity_bins == len(intensity_edges) - 1] = len(intensity_edges) - 2

    mc_weights = 1 / mc_hist[mc_alt_bins, mc_intensity_bins]
    mc_weights *= availability_hist[mc_alt_bins, mc_intensity_bins]

    # --- Storing to a data frame ---
    mc_weight_df = pd.DataFrame(data={'event_weight': mc_weights},index=mc_data.index)

    return mc_weight_df

# ===================================================
# === Get the argument and load the configuration ===
# ===================================================

parser = argparse.ArgumentParser()

parser.add_argument('--input-file', '-i', dest='input_file', type=str, 
    help='Path to the input training samples of MC gamma rays with HDF5 format')

parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./energy_rf.joblib', 
    help='Path and name of the trained RF')

parser.add_argument('--config-file', '-c', dest='config_file', type=str, default='./config.yaml', help='Path to the config file')

args = parser.parse_args()

config = yaml.safe_load(open(args.config_file, "r"))

print('\nTelescope IDs: {}'.format(config['tel_ids']))

# ======================================
# === Load the Gamma training sample ===
# ======================================

print(f'\nLoading the Gamma training samples: {args.input_file}')

mc_train = pd.read_hdf(args.input_file, key='events/params')
mc_train.sort_index(inplace=True)

mc_train['multiplicity'] = mc_train.groupby(['obs_id', 'event_id']).size()
mc_train = mc_train.query('multiplicity == 3')

for tel_name in config['tel_ids']:
    n_events = len(mc_train.query('tel_id == {}'.format(config['tel_ids'][tel_name])))
    print(f'{tel_name}: {n_events} events')

# ===================================
# === Calculate the event weights ===
# ===================================

sin_edges = np.linspace(0, 1, num=51)
alt_edges = np.lib.scimath.arcsin(sin_edges)
intensity_edges = np.logspace(1, 6, num=51)

mc_train['event_weight'] = get_weights(mc_train, alt_edges, intensity_edges)

# ==================================
# === Training the Random Forest ===
# ==================================

print('\nTraining configuration:\n {}'.format(config['energy_rf']))

energy_estimator = EnergyEstimatorPandas(config['energy_rf']['features'], **config['energy_rf']['settings'])

print('\nTraining the Random Forest...')

energy_estimator.fit(mc_train)
energy_estimator.save(args.output_file)

print('\nParameter importances:')

for tel_name in config['tel_ids']:

    print(f'  {tel_name}:')
    tel_id = config['tel_ids'][tel_name]

    for i_par, param in enumerate(config['energy_rf']['features']):
        print(f'    {param}: {energy_estimator.telescope_regressors[tel_id].feature_importances_[i_par]}')

end_time = time.time()

print(f'\nDone. Elapsed time = {end_time - start_time:.2f} [sec]')
