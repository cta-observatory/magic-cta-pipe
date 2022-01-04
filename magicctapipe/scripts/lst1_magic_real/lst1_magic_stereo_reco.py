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
from ctapipe.containers import CameraHillasParametersContainer
from magicctapipe.utils import calc_impact

warnings.simplefilter('ignore')

__all__ = ['stereo_reco']


def stereo_reco(input_data, output_data, config):

    print(f'\nConfiguration for the stereo reconstruction:\n{config}')

    subarray = pd.read_pickle(config['subarray'])
    positions = subarray.positions

    print(f'\nSubarray configuration:\n{subarray.tels}')
    print(f'\nTelescope positions:\n{positions}')

    # --- load the input data ---
    print(f'\nLoading the input data: {input_data}')

    data_stereo = pd.read_hdf(input_data, key='events/params')

    data_type = 'mc' if ('mc_energy' in data_stereo.columns) else 'real'

    if data_type == 'mc':

        data_stereo.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
        data_stereo.sort_index(inplace=True)

        data_stereo['multiplicity'] = data_stereo.groupby(['obs_id', 'event_id']).size()
        data_stereo.query('multiplicity > 1', inplace=True)

    n_events = len(data_stereo.groupby(['obs_id', 'event_id']).size())
    print(f'Number of stereo events = {n_events}')

    if data_type == 'real':

        # --- check the pointing directions ---
        print('\nChecking the angular separation of LST-1 and MAGIC pointing directions...')

        theta_lim = 2/60 * u.deg

        event_ids = data_stereo.query('tel_id == 1').index.get_level_values('event_id')
        n_events_lst = len(event_ids)

        df = data_stereo.query(f'event_id == {list(event_ids)}')

        df_lst = df.query('tel_id == 1')
        df_magic = df.query('tel_id == [2, 3]').groupby(['obs_id', 'event_id']).mean()

        theta = angular_separation(
            lon1=df_lst['az_tel'].values*u.rad, lat1=df_lst['alt_tel'].values*u.rad,
            lon2=df_magic['az_tel'].values*u.rad, lat2=df_magic['alt_tel'].values*u.rad
        )

        n_events_sep = np.sum(theta.to(u.deg) > theta_lim)

        if n_events_sep > 0:
            print(f'--> {n_events_sep}/{n_events_lst} events are taken with the angular separation '
                    f'larger than {theta_lim*60} arcmin. Exiting.\n')
            sys.exit()

        else:
            print(f'--> All the events are taken with the angular separation less than {theta_lim*60} arcmin. Continue.')

    # --- apply the quality cuts ---
    print('\nApplying the quality cuts...')

    data_stereo.query(config['quality_cuts'], inplace=True)

    data_stereo['multiplicity'] = data_stereo.groupby(['obs_id', 'event_id']).size()
    data_stereo.query('multiplicity > 1', inplace=True)

    groupby = data_stereo.groupby(['obs_id', 'event_id']).size()

    # --- check the number of events ---
    n_events_total = len(data_stereo.groupby(['obs_id', 'event_id']).size())
    print(f'\nIn total {n_events_total} stereo events are found.')

    print('\nEvents with 2 tels info:')

    tel_ids_dict = {
        'LST-1 + MAGIC-I': [1, 2],
        'LST-1 + MAGIC-II': [1, 3],
        'MAGIC-I + MAGIC-II': [2, 3]
    }

    for tel_name, tel_ids, in zip(tel_ids_dict.keys(), tel_ids_dict.values()):
        
        df = data_stereo.query(f'(tel_id == {list(tel_ids)}) & (multiplicity == 2)')
        n_events = np.sum(df.groupby(['obs_id', 'event_id']).size().values == 2)
        print(f'{tel_name}: {n_events} events ({n_events/n_events_total*100:.1f}%)')

    print('\nEvents with 3 tels info:')

    n_events = len(data_stereo.query(f'multiplicity == 3').groupby(['obs_id', 'event_id']).size())
    print(f'LST-1 + MAGIC-I + MAGIC-II: {n_events:.0f} events ({n_events/n_events_total*100:.1f}%)')

    # --- reconstruct the stereo parameters ---
    print('\nReconstructing the stereo parameters...')

    hillas_reconstructor = HillasReconstructor()

    obs_ids_list = groupby.index.get_level_values('obs_id').values
    event_ids_list = groupby.index.get_level_values('event_id').values

    for i_ev, (obs_id, event_id) in enumerate(zip(obs_ids_list, event_ids_list)):

        if i_ev % 100 == 0:
            print(f'{i_ev} events')

        df_ev = data_stereo.query(f'(obs_id == {obs_id}) & (event_id == {event_id})')
        tel_ids_list = df_ev.index.get_level_values('tel_id')

        array_pointing = SkyCoord(
            alt=u.Quantity(np.mean(df_ev['alt_tel'].values), u.rad),
            az=u.Quantity(np.mean(df_ev['az_tel'].values), u.rad),
            frame=AltAz()
        )

        hillas_params = {}

        for tel_id in tel_ids_list:

            df_tel = df_ev.query(f'tel_id == {tel_id}')

            hillas_params[tel_id] = CameraHillasParametersContainer(
                intensity=float(df_tel['intensity'].values[0]),
                x=u.Quantity(df_tel['x'].values[0], u.m),
                y=u.Quantity(df_tel['y'].values[0], u.m),
                r=u.Quantity(df_tel['r'].values[0], u.m),
                phi=Angle(df_tel['phi'].values[0], u.deg),
                length=u.Quantity(df_tel['length'].values[0], u.m),
                width=u.Quantity(df_tel['width'].values[0], u.m),
                psi=Angle(df_tel['psi'].values[0], u.deg),
                skewness=float(df_tel['skewness'].values[0]),
                kurtosis=float(df_tel['kurtosis'].values[0]),
            )

        stereo_params = hillas_reconstructor._predict(
            hillas_params,
            subarray,
            array_pointing
        )

        if stereo_params.az < 0:
            stereo_params.az = stereo_params.az + u.Quantity(2*np.pi, u.rad)

        for tel_id in tel_ids_list:

            # --- calculate the impact parameter ---
            impact = calc_impact(
                stereo_params.core_x, stereo_params.core_y, stereo_params.az, stereo_params.alt,
                positions[tel_id][0], positions[tel_id][1], positions[tel_id][2],
            )

            # --- save the reconstructed parameters ---
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

    print(f'{i_ev+1} events processed.')

    # --- save the data frame ---
    data_stereo.to_hdf(output_data, key='events/params')

    print(f'\nOutput data: {output_data}')


def main():

    start_time = time.time()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str,
        help='Path to a DL1 coincidence data file.'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, default='./dl1_stereo_lst1_magic.h5',
        help='Path to an output data file with h5 extention.'
    )

    arg_parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
        help='Path to a configuration file with yaml extention.'
    )

    args = arg_parser.parse_args()

    config_lst1_magic = yaml.safe_load(open(args.config_file, 'r'))

    stereo_reco(args.input_data, args.output_data, config_lst1_magic['stereo_reco'])

    print('\nDone.')
    print(f'\nelapsed time = {time.time() - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
