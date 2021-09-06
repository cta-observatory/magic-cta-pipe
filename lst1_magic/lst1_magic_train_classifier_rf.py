#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import time
import yaml
import warnings
import argparse
import pandas as pd
import numpy as np
from event_processing import EventClassifierPandas

warnings.simplefilter('ignore')

start_time = time.time()

def get_weights(mc_data, bkg_data, alt_edges, intensity_edges):
    mc_hist, _, _ = np.histogram2d(mc_data['alt_tel'], mc_data['intensity'], bins=[alt_edges, intensity_edges])
    bkg_hist, _, _ = np.histogram2d(bkg_data['alt_tel'],bkg_data['intensity'],bins=[alt_edges, intensity_edges])

    availability_hist = np.clip(mc_hist, 0, 1) * np.clip(bkg_hist, 0, 1)

    # --- MC weights ---
    mc_alt_bins = np.digitize(mc_data['alt_tel'], alt_edges) - 1
    mc_intensity_bins = np.digitize(mc_data['intensity'], intensity_edges) - 1

    # Treating the out-of-range events
    mc_alt_bins[mc_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    mc_intensity_bins[mc_intensity_bins == len(intensity_edges) - 1] = len(intensity_edges) - 2

    mc_weights = 1 / mc_hist[mc_alt_bins, mc_intensity_bins]
    mc_weights *= availability_hist[mc_alt_bins, mc_intensity_bins]

    # --- Bkg weights ---
    bkg_alt_bins = np.digitize(bkg_data['alt_tel'], alt_edges) - 1
    bkg_intensity_bins = np.digitize(bkg_data['intensity'], intensity_edges) - 1

    # Treating the out-of-range events
    bkg_alt_bins[bkg_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    bkg_intensity_bins[bkg_intensity_bins == len(intensity_edges) - 1] = len(intensity_edges) - 2

    bkg_weights = 1 / bkg_hist[bkg_alt_bins, bkg_intensity_bins]
    bkg_weights *= availability_hist[bkg_alt_bins, bkg_intensity_bins]

    # --- Storing to a data frame ---
    mc_weight_df = pd.DataFrame(data={'event_weight': mc_weights},index=mc_data.index)
    bkg_weight_df = pd.DataFrame(data={'event_weight': bkg_weights},index=bkg_data.index)

    return mc_weight_df, bkg_weight_df

# ===================================================
# === Get the argument and load the configuration ===
# ===================================================

parser = argparse.ArgumentParser()

parser.add_argument('--input-file-gamma', '-ig', dest='input_file_gamma', type=str, 
    help='Path to the input training samples of MC gamma rays with HDF5 format')

parser.add_argument('--input-file-proton', '-ip', dest='input_file_proton', type=str, 
    help='Path to the input training samples of MC proton with HDF5 format')

parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./classifier_rf.joblib', 
    help='Path and name of the trained RF')

parser.add_argument('--config-file', '-c', dest='config_file', type=str, default='./config.yaml', help='Path to the config file')

args = parser.parse_args()

config = yaml.safe_load(open(args.config_file, "r"))

print('\nTelescope IDs: {}'.format(config['tel_ids']))

# ======================================
# === Load the Gamma training sample ===
# ======================================

print(f'\nLoading the Gamma training samples: {args.input_file_gamma}')

mc_train_gamma = pd.read_hdf(args.input_file_gamma, key='events/params')
mc_train_gamma.sort_index(inplace=True)

mc_train_gamma['multiplicity'] = mc_train_gamma.groupby(['obs_id', 'event_id']).size()
mc_train_gamma = mc_train_gamma.query('multiplicity == 3')

mc_train_gamma['true_event_class'] = 0

for tel_name in config['tel_ids']:
    n_events = len(mc_train_gamma.query('tel_id == {}'.format(config['tel_ids'][tel_name])))
    print(f'{tel_name}: {n_events} events')

# =======================================
# === Load the Proton training sample ===
# =======================================

print(f'\nLoading the Proton training samples: {args.input_file_proton}')

mc_train_proton = pd.read_hdf(args.input_file_proton, key='events/params')
mc_train_proton.sort_index(inplace=True)

mc_train_proton['multiplicity'] = mc_train_proton.groupby(['obs_id', 'event_id']).size()
mc_train_proton = mc_train_proton.query('multiplicity == 3')

mc_train_proton['true_event_class'] = 1

for tel_name in config['tel_ids']:
    n_events = len(mc_train_proton.query('tel_id == {}'.format(config['tel_ids'][tel_name])))
    print(f'{tel_name}: {n_events} events')

# ==================================
# === Training the event weights ===
# ==================================

sin_edges = np.linspace(0, 1, num=51)
alt_edges = np.lib.scimath.arcsin(sin_edges)
intensity_edges = np.logspace(1, 6, num=51)

weights_gamma, weights_proton = get_weights(mc_train_gamma, mc_train_proton, alt_edges, intensity_edges)

mc_train_gamma['event_weight'] = weights_gamma
mc_train_proton['event_weight'] = weights_proton

# ===================================
# === Calculate the Random Forest ===
# ===================================

mc_train = mc_train_gamma.append(mc_train_proton)

print('\nTraining configuration:\n {}'.format(config['classifier_rf']))

class_estimator = EventClassifierPandas(config['classifier_rf']['features'], **config['classifier_rf']['settings'])

print('\nTraining the Random Forest...')

class_estimator.fit(mc_train)
class_estimator.save(args.output_file)

print('\nParameter importances:')

for tel_name in config['tel_ids']:

    print(f'  {tel_name}:')
    tel_id = config['tel_ids'][tel_name]

    for i_par, param in enumerate(config['classifier_rf']['features']):
        print(f'    {param}: {class_estimator.telescope_classifiers[tel_id].feature_importances_[i_par]}')

end_time = time.time()

print(f'\nDone. Elapsed time = {end_time - start_time:.2f} [sec]')
