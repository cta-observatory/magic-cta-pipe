#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import time
import yaml
import warnings
import argparse
import pandas as pd
from astropy import units as u
from astropy.time import Time
from event_processing import EnergyEstimatorPandas, DirectionEstimatorPandas, EventClassifierPandas
from utils import transform_to_radec

warnings.simplefilter('ignore')

__all__ = ['dl1_to_dl2']


def reco_energy(data, energy_rf, config):

    energy_estimator = EnergyEstimatorPandas(config['features'], **config['settings'])
    energy_estimator.load(energy_rf)

    print('\nReconstructing energy...')

    energy_reco = energy_estimator.predict(data)

    for param in energy_reco.columns:
        data[param] = energy_reco[param]

    return data


def reco_direction(data, direction_rf, tel_discriptions, config):

    direction_estimator = DirectionEstimatorPandas(config['features'], tel_discriptions, **config['settings'])
    direction_estimator.load(direction_rf)

    print('\nReconstructing direction...')

    direction_reco = direction_estimator.predict(data)

    for param in direction_reco.columns:
        data[param] = direction_reco[param]

    return data


def reco_gammaness(data, classifier_rf, config):

    class_estimator = EventClassifierPandas(config['features'], **config['settings'])
    class_estimator.load(classifier_rf)

    print('\nReconstructing gammaness...')

    class_reco = class_estimator.predict(data)

    for param in class_reco.columns:
        data[param] = class_reco[param]

    return data


def dl1_to_dl2(data_path, config, energy_rf=None, direction_rf=None, classifier_rf=None):

    print(f'\nLoading the input data file: {data_path}')

    data_stereo = pd.read_hdf(data_path, key='events/params')
    data_stereo.sort_index(inplace=True)

    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):
        n_events = len(data_stereo.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}: {n_events} events')

    # --- reconstruct energy ---
    if energy_rf != None:

        data_stereo = reco_energy(data_stereo, energy_rf, config['energy_rf'])

    # --- reconstruct arrival direction ---
    if direction_rf != None:

        subarray = pd.read_pickle(config['stereo_reco']['subarray'])
        tel_discriptions = subarray.tels

        data_stereo = reco_direction(data_stereo, direction_rf, tel_discriptions, config['direction_rf'])

        # --- transform Alt/Az to RA/Dec ---
        df_lst = data_stereo.query('tel_id == 1')

        type_lst_time = config['event_coincidence']['type_lst_time']

        ra, dec = transform_to_radec(
            alt=u.Quantity(df_lst['alt_reco_mean'].values, u.rad), 
            az=u.Quantity(df_lst['az_reco_mean'].values, u.rad),
            timestamp=Time(df_lst[type_lst_time].values, format='unix', scale='utc')
        ) 

        for tel_id in [1, 2, 3]:
            data_stereo.loc[(slice(None), slice(None), tel_id), 'ra_reco_mean'] = ra
            data_stereo.loc[(slice(None), slice(None), tel_id), 'dec_reco_mean'] = dec


    # --- reconstruct gammaness ---
    if classifier_rf != None:

        data_stereo = reco_gammaness(data_stereo, classifier_rf, config['classifier_rf'])

    return data_stereo


def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str, 
        help='Path to an input DL1+stereo data file, e.g., dl1_stereo_lst1_magic_Run02923.0040.h5'
    )

    arg_parser.add_argument(
        '--energy-rf', '-re', dest='energy_rf', type=str, default=None,
        help='Path to a trained energy RF, e.g., energy_rf.joblib'
    )

    arg_parser.add_argument(
        '--direction-rf', '-rd', dest='direction_rf', type=str, default=None,
        help='Path to a trained direction RF, e.g., direction_rf.joblib'
    )

    arg_parser.add_argument(
        '--classifier-rf', '-rc', dest='classifier_rf', type=str, default=None,
        help='Path to a trained classifier RF, e.g., classifier_rf.joblib'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, 
        help='Path and name of an output data file with HDF5 format, e.g., dl2_lst1_magic.h5'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str,
       help='Path to a config file with yaml format, e.g., config.yaml'
    )

    args = arg_parser.parse_args()

    # --- process the DL1+stereo data to DL2 ---
    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    data_stereo = dl1_to_dl2(
        args.input_data, config_lst1_magic, args.energy_rf, args.direction_rf, args.classifier_rf
    )

    # --- store the DL1+stereo data file ---
    data_stereo.to_hdf(args.output_data, key='events/params')

    print(f'\nOutput data file: {args.output_data}')

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')    


if __name__ == '__main__':
    main()