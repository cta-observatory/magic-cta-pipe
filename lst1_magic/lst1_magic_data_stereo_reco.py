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
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates.angle_utilities import angular_separation
from ctapipe.reco import HillasReconstructor
from ctapipe.containers import HillasParametersContainer, ReconstructedShowerContainer
from ctapipe.reco.reco_algorithms import InvalidWidthException
from utils import calc_impact

warnings.simplefilter('ignore')

__all__ = ['stereo_reco']
 
def stereo_reco(data_path, subarray):

    theta_lim = 2/60    # [deg]

    print(f'\nSubarray configuration:\n{subarray.tels}')
    print(f'\nTelescope positions:\n{subarray.positions}')

    # --- load the input data ---
    print(f'\nLoading the input data file: {data_path}')

    data_stereo = pd.read_hdf(data_path, key='events/params')
    data_stereo.sort_index(inplace=True)

    for tel_id, tel_name in zip([1, 2, 3], ['LST-1', 'MAGIC-I', 'MAGIC-II']):
        n_events = len(data_stereo.query(f'tel_id == {tel_id}'))
        print(f'{tel_name}: {n_events} events')

    # --- check the pointing directions --- 
    print('\nChecking the telescope pointings...')

    df_lst = data_stereo.query('tel_id == 1')
    df_magic = data_stereo.query('tel_id == 2')

    theta = angular_separation(
        lon1=df_lst['az_tel'].values*u.rad, lat1=df_lst['alt_tel'].values*u.rad,
        lon2=df_magic['az_tel'].values*u.rad, lat2=df_magic['alt_tel'].values*u.rad
    )

    theta = theta.to(u.deg).value
    condition = theta > theta_lim

    if np.sum(condition) == len(condition):
        print(f'--> All the events are taken with the angular separation larger than {theta_lim*60} arcmin. ' \
            'The events may be taken with different pointing directions. Please check your input data. Exiting.')
        sys.exit()

    elif np.sum(condition) > 0:
        n_events = np.sum(condition)
        print(f'--> {n_events} events are found with the angular separation larger than {theta_lim*60} arcmin. ' \
            'There may be an issue of the pointings. Please check your input data. Exiting.') 
        sys.exit()

    else:
        print(f'--> All the events are taken with the angular separation less than {theta_lim*60} arcmin. Continuing.')

    # --- reconstruct the stereo parameters ---
    print('\nReconstructing the stereo parameters...')

    hillas_reconstructor = HillasReconstructor()
    container = ReconstructedShowerContainer()

    for param in container.keys():
        container[param] = []

    event_ids = np.unique(data_stereo.index.get_level_values('event_id').values)

    event_ids_drop = []
        
    for i_ev, event_id in enumerate(event_ids):
        
        if i_ev%100 == 0:
            print(f'{i_ev} events')
        
        df_ev = data_stereo.query(f'event_id == {event_id}')
        
        array_pointing = SkyCoord(
            alt=u.Quantity(df_ev['alt_tel'].values[0], u.rad),
            az=u.Quantity(df_ev['az_tel'].values[0], u.rad), 
            frame=AltAz()
        )
        
        hillas_params = {}
        
        for tel_id in [1, 2, 3]:    
        
            df_tel = df_ev.query(f'tel_id == {tel_id}')

            hillas_params[tel_id] = HillasParametersContainer()
            hillas_params[tel_id].intensity = float(df_tel['intensity'].values[0])
            hillas_params[tel_id].x = u.Quantity(df_tel['x'].values[0], u.m)
            hillas_params[tel_id].y = u.Quantity(df_tel['y'].values[0], u.m)
            hillas_params[tel_id].r = u.Quantity(df_tel['r'].values[0], u.m)
            hillas_params[tel_id].phi = u.Quantity(df_tel['phi'].values[0], u.deg)
            hillas_params[tel_id].length = u.Quantity(df_tel['length'].values[0], u.m)
            hillas_params[tel_id].width = u.Quantity(df_tel['width'].values[0], u.m)
            hillas_params[tel_id].psi = u.Quantity(df_tel['psi'].values[0], u.deg)
            hillas_params[tel_id].skewness = float(df_tel['skewness'].values[0])
            hillas_params[tel_id].kurtosis = float(df_tel['kurtosis'].values[0])
        
        try:
            stereo_params = hillas_reconstructor.predict(hillas_params, subarray, array_pointing)

        except InvalidWidthException:
            print(f'--> event ID {event_id}: HillasContainer contains width = 0 or nan. Stereo parameter calculation skipped.')
            event_ids_drop.append(event_id)
            continue 

        for param in stereo_params.keys():

            if param == 'tel_ids':
                continue
                
            elif 'astropy' in str(type(stereo_params[param])):
                container[param].append(stereo_params[param].value)

            else:
                container[param].append(stereo_params[param])

    print(f'{i_ev+1} events')

    for param in container.keys():
        container[param] = np.array(container[param])

    container['az'][container['az'] < 0] += 2*np.pi
    
    data_stereo = data_stereo.drop(index=event_ids_drop, level=1)

    for tel_id in [1, 2, 3]:

        tel_position = subarray.positions[tel_id]

        # --- calculate the impact parameter ---
        impact = calc_impact(
            container['core_x'], container['core_y'], container['az'], container['alt'],
            tel_position[0].value, tel_position[1].value, tel_position[2].value
        )

        # --- write the stereo parameters in the data frame ---
        data_stereo.loc[(slice(None), slice(None), tel_id), 'impact'] = impact

        for param in container.keys():

            if param == 'tel_ids':
                continue

            else: 
                data_stereo.loc[(slice(None), slice(None), tel_id), param] = container[param]

    return data_stereo

# ============
# === Main ===
# ============

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
    config_lst1_magic = yaml.safe_load(open(args.config_file, "r"))

    subarray = pd.read_pickle(config_lst1_magic['stereo_reco']['subarray'])

    data_stereo = stereo_reco(args.input_data, subarray)

    # --- store the DL1 stereo data file ---
    print(f'\nOutput data file: {args.output_data}')
    data_stereo.to_hdf(args.output_data, key='events/params')

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')    

if __name__ == '__main__':
    main()
