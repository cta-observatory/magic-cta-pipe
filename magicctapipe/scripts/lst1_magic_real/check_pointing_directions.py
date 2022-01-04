#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import glob
import time
import uproot
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from utils import get_obs_ids_from_name

pyplot.rcParams['figure.figsize'] = (12, 9)
pyplot.rcParams['font.size'] = 15
pyplot.rcParams['grid.linestyle'] = ':'

warnings.simplefilter('ignore')
color_cycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']

__all__ = ['check_pointing_directions']


def check_pointing_directions(input_data_mask_lst, input_data_mask_magic, output_file):

    pyplot.figure()
    pyplot.xlabel('Azimuth angle [deg]')
    pyplot.ylabel('Zenith angle [deg]')

    # --- get LST observation IDs ---
    obs_ids_lst = get_obs_ids_from_name(input_data_mask_lst)

    print(f'\nFound the following LST observation IDs:\n{obs_ids_lst}')

    # --- check LST-1 pointing directions ---
    parent_dir = str(Path(input_data_mask_lst).parent)

    print('\nLoading the LST-1 input data:')

    for i_obs, obs_id in enumerate(obs_ids_lst):

        print(f'\nRun{obs_id}:')

        azimuths = np.array([])
        zeniths = np.array([])

        data_mask = f'{parent_dir}/*Run{obs_id}*.h5'

        paths_list = glob.glob(data_mask)
        paths_list.sort()

        for i_path, path in enumerate(paths_list):

            if (i_path % 10 == 0) or (path == paths_list[-1]):

                print(path)

                df = pd.read_hdf(path, key='dl1/event/telescope/parameters/LST_LSTCam')

                azimuths = np.append(azimuths, np.rad2deg(df['az_tel'].values))
                zeniths = np.append(zeniths, 90 - np.rad2deg(df['alt_tel'].values))

        pyplot.plot(
            azimuths, zeniths, color=color_cycle[i_obs%10], label=f'LST ID {int(obs_id)}'
        )

    # --- check MAGIC pointing directions ---
    print('\nLoading the MAGIC input data:')

    paths_list = glob.glob(input_data_mask_magic)
    paths_list.sort()

    for i_path, path in enumerate(paths_list):

        print(path)

        with uproot.open(path) as input_data:

            obs_id = int(input_data['RunHeaders']['MRawRunHeader_1./MRawRunHeader_1.fRunNumber'].array()[0])
            azimuths = np.array(input_data['Events'][f'MPointingPos_1.fAz'].array())
            zeniths = np.array(np.array(input_data['Events'][f'MPointingPos_1.fZd'].array()))

        pyplot.plot(
            azimuths, zeniths, linewidth=10, alpha=0.5,
            color=color_cycle[i_path%10], label=f'MAGIC ID {obs_id}'
        )

    pyplot.grid()
    pyplot.legend(fontsize=10)

    # --- save the figure ---
    pyplot.savefig(output_file)

    print(f'\nOutput file: {output_file}')


def main():

    start_time = time.time()

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
        '--output-file', '-o', dest='output_file', type=str, default='./pointing_directions.pdf',
        help='Path to an output file.'
    )

    args = arg_parser.parse_args()

    check_pointing_directions(args.input_data_lst, args.input_data_magic, args.output_file)

    print('\nDone.')
    print(f'\nelapsed time = {time.time() - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
