#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import os
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
color_cycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']

__all__ = ['check_pointing']


def check_pointing(input_data_mask_lst, input_data_mask_magic, output_file):

    print(f'\nInput LST-1 data: {input_data_mask_lst}')
    print(f'Input MAGIC data: {input_data_mask_magic}')

    pyplot.figure()
    pyplot.xlabel('Azimuth [deg]')
    pyplot.ylabel('Zenith [deg]')

    paths_list = glob.glob(input_data_mask_lst)
    paths_list.sort()

    obs_ids_lst = []

    for path in paths_list:

        obs_id = re.findall('.*dl1_LST-1\.Run(\d+)\..*', path)[0]
        obs_ids_lst.append(obs_id)

    obs_ids_lst = np.unique(obs_ids_lst)

    print(f'\nLST observation IDs: {obs_ids_lst}')

    parent_dir = str(Path(input_data_mask_lst).parent)

    print('\nLoading the LST-1 input data files:')

    for i_obs, obs_id in enumerate(obs_ids_lst):

        print(f'Run{obs_id}:')

        azimuths = np.array([])
        zeniths = np.array([])
        
        data_mask = f'{parent_dir}/*Run{obs_id}*.h5'

        paths_list = glob.glob(data_mask)
        paths_list.sort()

        for i_path, path in enumerate(paths_list):

            if (i_path % 20 == 0) or (path == paths_list[-1]): 

                print(path)

                df = pd.read_hdf(path, key='dl1/event/telescope/parameters/LST_LSTCam')
                df.set_index(['obs_id', 'event_id'], inplace=True)
                df.sort_index(inplace=True)

                azimuths = np.append(azimuths, np.rad2deg(df['az_tel'].values))
                zeniths = np.append(zeniths, 90 - np.rad2deg(df['alt_tel'].values))

        pyplot.plot(
            azimuths, zeniths, color=color_cycle[i_obs%10], label=f'LST ID {int(obs_id)}'
        )

    print('\nLoading the MAGIC input data files:')

    paths_list = glob.glob(input_data_mask_magic)
    paths_list.sort()

    for i_path, path in enumerate(paths_list):

        print(path)

        with uproot.open(path) as input_data:

            obs_id = np.array(input_data['RunHeaders']['MRawRunHeader_1./MRawRunHeader_1.fRunNumber'].array())[0]
            azimuths = np.array(input_data['Events'][f'MPointingPos_1.fAz'].array())
            zeniths = np.array(np.array(input_data['Events'][f'MPointingPos_1.fZd'].array()))
            
        pyplot.plot(
            azimuths, zeniths, linewidth=10, alpha=0.5, 
            color=color_cycle[i_path%10], label=f'MAGIC ID {obs_id}'
        )

    pyplot.grid()
    pyplot.legend(fontsize=10)

    # --- save the figure ---
    output_dir = str(Path(output_file).parent)
    os.makedirs(output_dir, exist_ok=True)

    pyplot.savefig(output_file)

    print(f'\nOutput file: {output_file}')
        

def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser() 

    arg_parser.add_argument(
        '--input-data-lst', '-l', dest='input_data_lst', type=str, 
        help='Path to LST-1 DL1 data files.'
    )

    arg_parser.add_argument(
        '--input-data-magic', '-m', dest='input_data_magic', type=str, 
        help='Path to MAGIC SuperStar data files.'
    )

    arg_parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str,
        help='Path to an output PDF file. Directory will be created if it does not exist.'
    )

    args = arg_parser.parse_args()

    # --- check the pointing directions ---
    check_pointing(
        args.input_data_lst, args.input_data_magic, args.output_file
    )

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
