#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

Reconstruct the stereo parameters of the events containing more that two telescopes information.
The quality cuts specified in the configuration file will be applied before the reconstruction.
If real data is input and it contains both LST-1 and MAGIC events, the angular separation between 
the telescope systems are checked, and the script stops if they are separated by more than 2 arcmin. 

Usage:
$ python lst1_magic_stereo_reco.py 
--input-file "./data/dl1_coincidence/dl1_lst1_magic_Run02923.0040.h5"
--output-file "./data/dl1_stereo/dl1_stereo_lst1_magic_Run02923.0040.h5"
--config-file "./config.yaml"
"""

import sys
import h5py
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
from ctapipe.containers import HillasParametersContainer
from utils import calc_impact

warnings.simplefilter('ignore')

__all__ = ['stereo_reco']


def calc_tel_mean_pointing(data):

    x_coords = np.cos(np.deg2rad(data['alt_tel'])) * np.cos(np.deg2rad(data['az_tel']))
    y_coords = np.cos(np.deg2rad(data['alt_tel'])) * np.sin(np.deg2rad(data['az_tel']))
    z_coords = np.sin(np.deg2rad(data['alt_tel']))

    coord_mean = SkyCoord(
        x=x_coords.groupby(['obs_id', 'event_id']).mean().values,
        y=y_coords.groupby(['obs_id', 'event_id']).mean().values,
        z=z_coords.groupby(['obs_id', 'event_id']).mean().values,
        representation_type='cartesian'
    )
        
    df_mean = pd.DataFrame(
        data={'alt_tel_mean': coord_mean.spherical.lat.to(u.deg).value,
              'az_tel_mean': coord_mean.spherical.lon.to(u.deg).value},
        index=data.groupby(['obs_id', 'event_id']).mean().index
    )

    return df_mean

 
def stereo_reco(input_file, output_file, config):

    tel_combinations = {
        'm1_m2': [2, 3],
        'lst1_m1': [1, 2],
        'lst1_m2': [1, 3],
        'lst1_m1_m2': [1, 2, 3]
    }

    config_sterec = config['stereo_reco']

    print(f'\nConfiguration for the stereo reconstruction:\n{config_sterec}')

    subarray = pd.read_pickle(config_sterec['subarray'])
    tel_positions = subarray.positions

    print(f'\nSubarray configuration:\n{subarray.tels}')
    print(f'\nTelescope positions:\n{tel_positions}')

    # --- load the input data file ---
    print(f'\nLoading the input data file:\n{input_file}')

    data_joint = pd.read_hdf(input_file, key='events/params')
    data_type = 'mc' if ('mc_energy' in data_joint.columns) else 'real'

    if data_type == 'mc':

        data_joint.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
        data_joint.sort_index(inplace=True)

        for param in ['slope_err', 'intercept_err']:
            if param in data_joint.columns:
                data_joint.drop(param, axis=1, inplace=True)

        data_joint['multiplicity'] = data_joint.groupby(['obs_id', 'event_id']).size()
        data_joint.query('multiplicity > 1', inplace=True)

    n_events_total = len(data_joint.groupby(['obs_id', 'event_id']).size())
    print(f'\nIn total {n_events_total} stereo events:') 

    for tel_combo, tel_ids in zip(tel_combinations.keys(), tel_combinations.values()):
        
        df = data_joint.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        n_events = np.sum(df.groupby(['obs_id', 'event_id']).size() == len(tel_ids))
        print(f'{tel_combo}: {n_events} events ({n_events/n_events_total*100:.1f}%)')

    # --- check the angular separation ---
    tel_id_lst = 1
    telescope_ids = np.unique(data_joint.index.get_level_values('tel_id'))

    if (data_type == 'real') and (tel_id_lst in telescope_ids):

        print('\nChecking the angular separation of LST-1 and MAGIC pointing directions...')

        theta_lim = u.Quantity(2, u.arcmin)

        df_lst = data_joint.query('tel_id == 1')
        n_events_joint = len(df_lst)

        obs_ids_joint = list(df_lst.index.get_level_values('obs_id'))
        event_ids_joint = list(df_lst.index.get_level_values('event_id'))

        multi_indices = pd.MultiIndex.from_arrays([obs_ids_joint, event_ids_joint], names=['obs_id', 'event_id'])

        df_magic = data_joint.query('tel_id == [2, 3]')
        df_magic.reset_index(level='tel_id', inplace=True)
        df_magic = df_magic.loc[multi_indices]
        df_magic.sort_index(inplace=True)
        
        df_magic_pointing = calc_tel_mean_pointing(df_magic)

        theta = angular_separation(
            lon1=u.Quantity(df_lst['az_tel'].values, u.deg), 
            lat1=u.Quantity(df_lst['alt_tel'].values, u.deg),
            lon2=u.Quantity(df_magic_pointing['az_tel_mean'].values, u.deg), 
            lat2=u.Quantity(df_magic_pointing['alt_tel_mean'].values, u.deg)
        )

        n_events_sep = np.sum(theta > theta_lim)

        if n_events_sep == n_events_joint:
            print(f'--> All the events are separeted more than {theta_lim.value} arcmin. ' \
                  'The data would be taken by different wobble offsets. Please check the input data. Exiting.\n')
            sys.exit()

        elif n_events_sep > 0:
            print(f'--> {n_events_sep} events are separeted more than {theta_lim.value} arcmin. ' \
                  'There would be a pointing issue. Please check the input data carefully. Exiting.\n')
            sys.exit()

        else: 
            print(f'--> All the events are separated less than {theta_lim.value} arcmin.')

    # --- apply the quality cuts ---
    print('\nApplying the quality cuts...')

    data_joint.query(config_sterec['quality_cuts'], inplace=True)

    data_joint['multiplicity'] = data_joint.groupby(['obs_id', 'event_id']).size()
    data_joint.query('multiplicity > 1', inplace=True)

    n_events_total = len(data_joint.groupby(['obs_id', 'event_id']).size()) 
    print(f'--> {n_events_total} stereo events are survived:') 

    for tel_combo, tel_ids in zip(tel_combinations.keys(), tel_combinations.values()):
        
        df = data_joint.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        n_events = np.sum(df.groupby(['obs_id', 'event_id']).size() == len(tel_ids))
        print(f'{tel_combo}: {n_events} events ({n_events/n_events_total*100:.1f}%)')

    # --- reconstruct the stereo parameters ---
    print('\nReconstructing the stereo parameters...')

    hillas_reconstructor = HillasReconstructor()

    df_mean_pointing = calc_tel_mean_pointing(data_joint)
    data_joint = data_joint.join(df_mean_pointing)

    group = data_joint.groupby(['obs_id', 'event_id']).size()

    observation_ids = group.index.get_level_values('obs_id').values
    event_ids = group.index.get_level_values('event_id').values

    for i_ev, (obs_id, ev_id) in enumerate(zip(observation_ids, event_ids)):
    
        if (i_ev % 100) == 0:
            print(f'{i_ev} events')
        
        df_ev = data_joint.query(f'(obs_id == {obs_id}) & (event_id == {ev_id})')
        telescope_ids = df_ev.index.get_level_values('tel_id')

        array_pointing = SkyCoord(
            alt=u.Quantity(df_ev['alt_tel_mean'].values[0], u.deg),
            az=u.Quantity(df_ev['az_tel_mean'].values[0], u.deg),
            frame=AltAz()
        )
        
        hillas_params = {}
        
        for tel_id in telescope_ids:    
        
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
        
        stereo_params = hillas_reconstructor.predict(hillas_params, subarray, array_pointing)

        if stereo_params.az < 0:
                stereo_params.az += u.Quantity(360, u.deg)

        for tel_id in telescope_ids:

            # --- calculate the impact parameter ---
            impact = calc_impact(
                core_x=stereo_params.core_x, core_y=stereo_params.core_y, az=stereo_params.az, alt=stereo_params.alt,
                tel_pos_x=tel_positions[tel_id][0], tel_pos_y=tel_positions[tel_id][1], tel_pos_z=tel_positions[tel_id][2],
            )

            # --- save the reconstructed parameters ---
            data_joint.loc[(obs_id, ev_id, tel_id), 'h_max'] = stereo_params.h_max.to(u.m).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'alt'] = stereo_params.alt.to(u.deg).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'az'] = stereo_params.az.to(u.deg).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'core_x'] = stereo_params.core_x.to(u.m).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'core_y'] = stereo_params.core_y.to(u.m).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'impact'] = impact.to(u.m).value
            
    print(f'{i_ev+1} events processed.')

    # --- save the data frame ---
    data_joint.to_hdf(output_file, key='events/params')

    with h5py.File(input_file, 'r') as f:
        if 'simulation' in f.keys():
            df = pd.read_hdf(input_file, key='simulation/params')
            df.to_hdf(output_file, key='simulation/params', mode='a')

    print(f'\nOutput data file: {output_file}')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str, 
        help='Path to an input DL1 data file containing events triggering multiple telescopes.' 
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./dl1_stereo.h5',
        help='Path to an output DL1-stereo data file.'
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
       help='Path to a configuration file.'
    )

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file, 'r'))

    stereo_reco(args.input_file, args.output_file, config)

    print('\nDone.')

    end_time = time.time()
    print(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
