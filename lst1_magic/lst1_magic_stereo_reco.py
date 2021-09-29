#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import sys
import time
import yaml
import argparse
import warnings
import numpy as np 
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz, Angle
from astropy.coordinates.angle_utilities import angular_separation
from ctapipe.reco import HillasReconstructor
from ctapipe.containers import HillasParametersContainer, ReconstructedShowerContainer
from ctapipe.reco.reco_algorithms import InvalidWidthException
from utils import calc_impact

warnings.simplefilter('ignore')

__all__ = ['stereo_reco']


def check_pointings(data_stereo):

    theta_lim = 2/60    # [deg]

    print('\nChecking the telescope pointings...')

    df_lst = data_stereo.query('tel_id == 1')
    df_magic = data_stereo.query('tel_id == 2')
    
    theta = angular_separation(
        lon1=df_lst['az_tel'].values*u.rad, lat1=df_lst['alt_tel'].values*u.rad,
        lon2=df_magic['az_tel'].values*u.rad, lat2=df_magic['alt_tel'].values*u.rad
    )

    theta = theta.to(u.deg).value

    n_events = len(theta)
    condition = (theta > theta_lim)

    if np.sum(condition) == n_events:
        print(f'--> All the events are taken with the angular separation larger than {theta_lim*60} arcmin. ' \
            'The events may be taken with different pointing directions. Please check your input data. Exiting.')
        sys.exit()

    elif np.sum(condition) > 0:
        print(f'--> {np.sum(condition)} events are found with the angular separation larger than {theta_lim*60} arcmin. ' \
            'There may be an issue of the pointings. Please check your input data. Exiting.') 
        sys.exit()

    else:
        print(f'--> All the events are taken with the angular separation less than {theta_lim*60} arcmin. Continuing.')

 
def stereo_reco(data_path, config):

    print(f'Configuration for the stereo reconstruction:\n{config}')

    subarray = pd.read_pickle(config['subarray'])
    positions = subarray.positions

    print(f'\nSubarray configuration:\n{subarray.tels}')
    print(f'\nTelescope positions:\n{positions}')

    # --- load the input data ---
    print(f'\nLoading the input data file: {data_path}')

    data_stereo = pd.read_hdf(data_path, key='events/params')

    if type(data_stereo.index) == pd.Int64Index:   # in case default MC dl1 data file is loaded
        data_stereo.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

    data_stereo.sort_index(inplace=True)

    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):
        n_events = len(data_stereo.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}: {n_events} events')

    is_data = ('mc_energy' not in data_stereo.columns)

    if is_data:
        check_pointings(data_stereo)

    # --- reconstruct the stereo parameters ---
    print('\nReconstructing the stereo parameters...')

    hillas_reconstructor = HillasReconstructor()

    allowed_tels_num = config['allowed_tels_num']

    obs_id = np.unique(data_stereo.index.get_level_values('obs_id').values)
    event_ids_list = np.unique(data_stereo.index.get_level_values('event_id').values)
        
    for i_ev, event_id in enumerate(event_ids_list):
    
        if i_ev%100 == 0:
            print(f'{i_ev} events')
        
        df_ev = data_stereo.query(f'event_id == {event_id}')

        tel_ids_list = df_ev.index.get_level_values('tel_id')
        num_tels = len(tel_ids_list)

        if num_tels < allowed_tels_num:
            print(f'--> {i_ev} event (event ID = {event_id}): the number of triggered telescopes = {num_tels} '
                   f'is less than the required number = {allowed_tels_num}. Skipping.')
            data_stereo = data_stereo.drop(index=event_id, level=1)
            continue

        array_pointing = SkyCoord(
            alt=u.Quantity(df_ev['alt_tel'].values[0], u.rad),
            az=u.Quantity(df_ev['az_tel'].values[0], u.rad), 
            frame=AltAz()
        )
        
        hillas_params = {}
        
        for tel_id in tel_ids_list:    
        
            df_tel = df_ev.query(f'tel_id == {tel_id}')

            hillas_params[tel_id] = HillasParametersContainer()
            hillas_params[tel_id].intensity = float(df_tel['intensity'].values[0])
            hillas_params[tel_id].x = u.Quantity(df_tel['x'].values[0], u.m)
            hillas_params[tel_id].y = u.Quantity(df_tel['y'].values[0], u.m)
            hillas_params[tel_id].r = u.Quantity(df_tel['r'].values[0], u.m)
            hillas_params[tel_id].phi = Angle(df_tel['phi'].values[0], u.deg)
            hillas_params[tel_id].length = u.Quantity(df_tel['length'].values[0], u.m)
            hillas_params[tel_id].width = u.Quantity(df_tel['width'].values[0], u.m)
            hillas_params[tel_id].psi = Angle(df_tel['psi'].values[0], u.deg)
            hillas_params[tel_id].skewness = float(df_tel['skewness'].values[0])
            hillas_params[tel_id].kurtosis = float(df_tel['kurtosis'].values[0])
        
        try:
            stereo_params = hillas_reconstructor.predict(hillas_params, subarray, array_pointing)

        except InvalidWidthException:
            print(f'--> {i_ev} event (event ID = {event_id}): HillasContainer contains width = 0 or nan. ' \
                   'Stereo parameter reconstruction failed. Skipped.')
            data_stereo = data_stereo.drop(index=event_id, level=1)
            continue 

        if stereo_params.az < 0:
                stereo_params.az = stereo_params.az + u.Quantity(2*np.pi, u.rad)

        for tel_id in tel_ids_list:

            # --- calculate the impact parameter ---
            impact = calc_impact(
                stereo_params.core_x, stereo_params.core_y, stereo_params.az, stereo_params.alt,
                positions[tel_id][0], positions[tel_id][1], positions[tel_id][2],
            )

            # --- write the reconstructed parameters ---
            data_stereo.loc[(obs_id, event_id, tel_id), 'alt'] = stereo_params.alt.to(u.rad).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'alt_uncert'] = stereo_params.alt_uncert.to(u.rad).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'az'] = stereo_params.az.to(u.rad).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'az_uncert'] = stereo_params.az_uncert.to(u.rad).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'core_x'] = stereo_params.core_x.to(u.m).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'core_y'] = stereo_params.core_y.to(u.m).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'core_uncert'] = stereo_params.core_uncert.to(u.m).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'impact'] = impact.to(u.m).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'h_max'] = stereo_params.h_max.to(u.m).value
            data_stereo.loc[(obs_id, event_id, tel_id), 'h_max_uncert'] = stereo_params.h_max_uncert.to(u.m).value
            
    print(f'{i_ev+1} events')

    print('\nThe number of reconstructed events:')
    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):
        n_events = len(data_stereo.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}: {n_events} events')

    return data_stereo


def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str, 
        help='Path to an input LST-1 + MAGIC DL1 file, e.g., dl1_lst1_magic_Run02923.0040.h5'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, 
        help='Path and name of an output data file with HDF5 format, e.g., dl1_stereo_lst1_magic.h5'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str,
       help='Path to a config file with yaml format, e.g., config.yaml'
    )

    args = arg_parser.parse_args()

    # --- perform the stereo reconstruction ---
    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    data_stereo = stereo_reco(args.input_data, config_lst1_magic['stereo_reco'])

    # --- store the DL1+stereo data file ---
    data_stereo.to_hdf(args.output_data, key='events/params')

    print(f'\nOutput data file: {args.output_data}')

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')    


if __name__ == '__main__':
    main()
