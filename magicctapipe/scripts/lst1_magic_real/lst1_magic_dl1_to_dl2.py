#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import os
import time
import yaml
import warnings
import argparse
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.time import Time
from magicctapipe.train.event_processing import EnergyEstimatorPandas
from magicctapipe.train.event_processing import DirectionEstimatorPandas
from magicctapipe.train.event_processing import EventClassifierPandas
from magicctapipe.utils import transform_to_radec

warnings.simplefilter('ignore')

__all__ = ['dl1_to_dl2']


def dl1_to_dl2(input_data, output_data, config):

    print(f'\nLoading the input data file: {input_data}')

    data_stereo = pd.read_hdf(input_data, key='events/params')
    data_stereo.sort_index(inplace=True)

    data_type = 'mc' if ('mc_energy' in data_stereo.columns) else 'real'

    if config['energy_rf']['path_to_file'] != None:

        # --- reconstruct energy ---
        print('\nReconstructing energy...')

        config_rf = config['energy_rf']

        energy_estimator = EnergyEstimatorPandas(config_rf['features'], **config_rf['settings'])
        energy_estimator.load(config_rf['path_to_file'])

        energy_reco = energy_estimator.predict(data_stereo)

        for param in energy_reco.columns:
            data_stereo[param] = energy_reco[param]

    
    if config['direction_rf']['path_to_file'] != None:

        # --- reconstruct arrival direction ---
        print('\nReconstructing arrival direction...')

        subarray = pd.read_pickle(config['stereo_reco']['subarray'])

        config_rf = config['direction_rf'] 

        direction_estimator = DirectionEstimatorPandas(config_rf['features'], subarray.tels, **config_rf['settings'])
        direction_estimator.load(config_rf['path_to_file'])

        direction_reco = direction_estimator.predict(data_stereo)

        for param in direction_reco.columns:
            data_stereo[param] = direction_reco[param]

        if data_type == 'real':

            print('Transforming Alt/Az to RA/Dec coordinates...')

            # --- pointing Alt/Az to RA/Dec ---
            ra_tel, dec_tel = transform_to_radec(
                alt=u.Quantity(data_stereo['alt_tel'].values, u.rad),
                az=u.Quantity(data_stereo['az_tel'].values, u.rad),
                timestamp=Time(data_stereo['timestamp'].values, format='unix', scale='utc')
            )

            data_stereo['ra_tel'] = ra_tel.to(u.deg).value
            data_stereo['dec_tel'] = dec_tel.to(u.deg).value
            
            # --- reconstructed arrival Alt/Az to RA/Dec ---
            ra_reco, dec_reco = transform_to_radec(
                alt=u.Quantity(data_stereo['alt_reco'].values, u.rad), 
                az=u.Quantity(data_stereo['az_reco'].values, u.rad),
                timestamp=Time(data_stereo['timestamp'].values, format='unix', scale='utc')
            ) 

            ra_reco_mean, dec_reco_mean = transform_to_radec(
                alt=u.Quantity(data_stereo['alt_reco_mean'].values, u.rad), 
                az=u.Quantity(data_stereo['az_reco_mean'].values, u.rad),
                timestamp=Time(data_stereo['timestamp'].values, format='unix', scale='utc')
            ) 

            data_stereo['ra_reco'] = ra_reco.to(u.deg).value
            data_stereo['dec_reco'] = dec_reco.to(u.deg).value  
            data_stereo['ra_reco_mean'] = ra_reco_mean.to(u.deg).value
            data_stereo['dec_reco_mean'] = dec_reco_mean.to(u.deg).value

    
    if config['classifier_rf']['path_to_file'] != None:

        # --- reconstruct gammaness ---
        print('\nReconstructing gammaness...')

        config_rf = config['classifier_rf'] 

        class_estimator = EventClassifierPandas(config_rf['features'], **config_rf['settings'])
        class_estimator.load(config_rf['path_to_file'])

        class_reco = class_estimator.predict(data_stereo)

        for param in class_reco.columns:
            data_stereo[param] = class_reco[param]


    # --- store the data ---
    output_dir = str(Path(output_data).parent)
    os.makedirs(output_dir, exist_ok=True)

    data_stereo.to_hdf(output_data, key='events/params')

    print(f'\nOutput data file: {output_data}')


def main():

    start_time = time.time()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str, 
        help='Path to a DL1 stereo data file.'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, default='./dl2_lst1_magic.h5',
        help='Path to an output data file with h5 extention.'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
       help='Path to a configuration file with yaml extention.'
    )

    args = arg_parser.parse_args()

    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    data_stereo = dl1_to_dl2(args.input_data, args.output_data, config_lst1_magic)

    print('\nDone.')
    print(f'\nelapsed time = {time.time() - start_time:.0f} [sec]\n')  


if __name__ == '__main__':
    main()