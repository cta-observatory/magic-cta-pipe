#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import time
import yaml
import glob
import warnings
import argparse
import pandas as pd
import numpy as np
from ctapipe.io import event_source
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, Angle, EarthLocation
from event_processing import EnergyEstimatorPandas, DirectionEstimatorPandas, EventClassifierPandas

warnings.simplefilter('ignore')

start_time = time.time()

# ===================================================
# === Get the argument and load the configuration ===
# ===================================================

parser = argparse.ArgumentParser()

parser.add_argument('--input-file', '-i', dest='input_file', type=str, 
    help='Path to the input file for applying the trained RFs')

parser.add_argument('--input-energy-rf', '-re', dest='input_energy_rf', type=str, default=None,
    help='Path to the input directory that contains the trained RFs')

parser.add_argument('--input-direction-rf', '-rd', dest='input_direction_rf', type=str, default=None,
    help='Path to the input directory that contains the trained RFs')

parser.add_argument('--input-classifier-rf', '-rc', dest='input_classifier_rf', type=str, default=None,
    help='Path to the input directory that contains the trained RFs')    

parser.add_argument('--output-file', '-o', dest='output_file', type=str, default='./dl2_data.h5', 
    help='Path and name of the output file')

parser.add_argument('--config-file', '-c', dest='config_file', type=str, default='./config.yaml', help='Path to the config file')

args = parser.parse_args()

config = yaml.safe_load(open(args.config_file, "r"))

print('\nTelescope IDs: {}'.format(config['tel_ids']))

# ============================
# === Load the input files ===
# ============================

print(f'\nLoading the input files: {args.input_file}')

test_data = pd.read_hdf(args.input_file, key='events/params')
test_data.sort_index(inplace=True)

test_data['multiplicity'] = test_data.groupby(['obs_id', 'event_id']).size()
test_data = test_data.query('multiplicity == 3')

for tel_name in config['tel_ids']:
    n_events = len(test_data.query('tel_id == {}'.format(config['tel_ids'][tel_name])))
    print(f'{tel_name}: {n_events} events')

# ========================
# === Applying the RFs ===
# ========================

if args.input_energy_rf != None:

    print('\nApplying the energy RF...')

    energy_estimator = EnergyEstimatorPandas(config['energy_rf']['features'], **config['energy_rf']['settings'])
    energy_estimator.load(args.input_energy_rf)

    energy_reco = energy_estimator.predict(test_data)
    test_data = test_data.join(energy_reco)

if args.input_direction_rf != None:

    print('\nApplying the direction RF...')

    source = event_source(config['subarray']['simtel_path'])
    telescope_discriptions = source.subarray.tel

    direction_estimator = DirectionEstimatorPandas(config['direction_rf']['features'], telescope_discriptions, **config['direction_rf']['settings'])
    direction_estimator.load(args.input_direction_rf)

    direction_reco = direction_estimator.predict(test_data)
    test_data = test_data.join(direction_reco)

    ts_type = config['coincidence']['timestamp_lst']

    if ts_type in test_data.columns:

        # === convert the "Alt/Az" to "RA/Dec" === 
        print(f'\nTransforming "Alt/Az" to "RA/Dec" direction...')

        config_loc = config['obs_location']
        location = EarthLocation.from_geodetic(lat=config_loc['lat']*u.deg, lon=config_loc['lon']*u.deg, height=config_loc['height']*u.m)

        df_tel = test_data.query('tel_id == {}'.format(config['tel_ids']['LST-1']))

        timestamps = Time(df_tel[ts_type].values, format='unix', scale='utc')
        horizon_frames = AltAz(location=location, obstime=timestamps)

        event_coords = SkyCoord(alt=df_tel['alt_reco_mean'].values, az=df_tel['az_reco_mean'].values, unit='rad', frame=horizon_frames)
        event_coords = event_coords.transform_to('fk5')

        for tel_name in config['tel_ids']:
            test_data.loc[(slice(None), slice(None), config['tel_ids'][tel_name]), 'ra_reco_mean'] = event_coords.ra.value
            test_data.loc[(slice(None), slice(None), config['tel_ids'][tel_name]), 'dec_reco_mean'] = event_coords.dec.value

if args.input_classifier_rf != None:

    print('\nApplying the classifier RF...')

    class_estimator = EventClassifierPandas(config['classifier_rf']['features'], **config['classifier_rf']['settings'])
    class_estimator.load(args.input_classifier_rf)

    class_reco = class_estimator.predict(test_data)
    test_data = test_data.join(class_reco)

test_data.to_hdf(args.output_file, key='events/params')

end_time = time.time()

print(f'\nDone. Elapsed time = {end_time - start_time:.2f} [sec]')
