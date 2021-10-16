#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import re
import glob
import time
import uproot
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot

pyplot.rcParams['figure.figsize'] = (12, 9)
pyplot.rcParams['font.size'] = 15
pyplot.rcParams['grid.linestyle'] = ':'

warnings.simplefilter('ignore')
c_cycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']

__all__ = ['check_pointings']


def check_pointings(data_path_lst, data_path_magic, output_file=None):

    pyplot.figure()
    pyplot.xlabel('Azimuth [deg]')
    pyplot.ylabel('Altitude [deg]')

    print('\nChecking the observation IDs of LST-1 input data...')

    paths_list = glob.glob(data_path_lst)
    paths_list.sort()

    obs_ids_list = []

    for path in paths_list:

        obs_id = re.findall('dl1_LST-1\.Run(\d+)\..*', path)[0]
        obs_ids_list.append(obs_id)

    obs_ids_list = np.unique(obs_ids_list)

    print(f'--> {obs_ids_list}')

    print('\nLoading the LST-1 input data files:')

    input_dir = str(Path(data_path_lst).parent)

    for i_obs, obs_id in enumerate(obs_ids_list):

        print(f'Run{obs_id}:')

        azimuth = np.array([])
        altitude = np.array([])
        
        data_path = input_dir + f'/*Run{obs_id}*.h5'

        paths_list = glob.glob(data_path)
        paths_list.sort()

        for i_path, path in enumerate(paths_list):

            if (i_path % 20 == 0) or (path == paths_list[-1]): 

                print(path)

                df = pd.read_hdf(path, key='dl1/event/telescope/parameters/LST_LSTCam')
                df.set_index(['obs_id', 'event_id'], inplace=True)
                df.sort_index(inplace=True)

                azimuth = np.append(azimuth, np.rad2deg(df['az_tel'].values))
                altitude = np.append(altitude, np.rad2deg(df['alt_tel'].values))

        pyplot.plot(
            azimuth, altitude, color=c_cycle[i_obs%10], label=f'LST-1 ID {obs_id}'
        )

    print('\nLoading MAGIC data files:')

    paths_list = glob.glob(data_path_magic)
    paths_list.sort()

    for i_path, path in enumerate(paths_list):

        print(path)

        obs_id = re.findall('.*_(\d+)_S_.*', path)[0]

        with uproot.open(path) as input_data:

            azimuth = np.array(input_data['Events'][f'MPointingPos_1.fAz'].array())
            altitude = np.array(90 - np.array(input_data['Events'][f'MPointingPos_1.fZd'].array()))
            
            pyplot.plot(
                azimuth, altitude, linewidth=10, alpha=0.5, 
                color=c_cycle[i_path%10], label=f'MAGIC ID {obs_id}'
            )

    pyplot.grid()
    pyplot.legend()

    if output_file != None:

        pyplot.savefig(output_file)
        print(f'\nOutput file: {output_file}')
        

def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser() 

    arg_parser.add_argument(
        '--input-data-lst', '-l', dest='input_data_lst', type=str, 
        help='Path to input LST-1 DL1 data files, e.g., dl1_LST-1.Run*.h5'
    )

    arg_parser.add_argument(
        '--input-data-magic', '-m', dest='input_data_magic', type=str, 
        help='Path to input MAGIC SuperStar data files, e.g., *_S_*.root'
    )

    arg_parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default=None,
        help='Path and name of an ouput PDF file, e.g., pointings.pdf' 
    )

    args = arg_parser.parse_args()

    # --- check the pointing directions ---
    check_pointings(
        args.input_data_lst, args.input_data_magic, args.output_file
    )

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
