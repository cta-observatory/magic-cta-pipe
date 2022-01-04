#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

import re
import time
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from utils import get_obs_ids_from_name

pyplot.rcParams['figure.figsize'] = (20, 7)
pyplot.rcParams['font.size'] = 20
pyplot.rcParams['grid.linestyle'] = ':'

color_cycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']


def check_coincidence_overview(input_data_mask, output_file, get_profile='False'):

    obs_ids_list = get_obs_ids_from_name(input_data_mask)

    print(f'\nFound the following observation IDs: {obs_ids_list}')

    # --- check the overview of the coincidence ---
    parent_dir = str(Path(input_data_mask).parent)
    output_dir = str(Path(output_file).parent)

    container = {}

    print('\nLoading the following input data files:')

    for obs_id in obs_ids_list:

        print(f'\nRun{obs_id}:')

        data_mask = f'{parent_dir}/*Run{obs_id}.*.h5'

        paths_list = glob.glob(data_mask)
        paths_list.sort()

        subset = 0
        previous_subrun_id = -10

        container[obs_id] = {}

        for path in paths_list:

            print(path)

            subrun_id = re.findall(f'.*Run{obs_id}\.(\d+)\.h5', path)[0]

            if int(subrun_id) != (previous_subrun_id + 1):

                subset += 1
                container[obs_id][subset] = pd.DataFrame()

            previous_subrun_id = int(subrun_id)

            # --- get the feature values ---
            df_features = pd.read_hdf(path, key='coincidence/features')
            container[obs_id][subset] = container[obs_id][subset].append(df_features, ignore_index=True)

            if get_profile == "True":

                # --- plot the profile of the coincidence scan ---
                df_profile = pd.read_hdf(path, key='coincidence/profile')
                offsets_avg = df_features['offset_avg_us'].values

                for i_tel, tel_type in enumerate(['m1', 'm2']):

                    pyplot.figure()
                    pyplot.xlabel('Offset [us]')
                    pyplot.ylabel('Number of coincident events')

                    pyplot.scatter(
                        df_profile['offset_us'].values, df_profile[f'n_coincidence_{tel_type}'].values
                    )

                    pyplot.step(
                        df_profile['offset_us'].values, df_profile[f'n_coincidence_btwn_{tel_type}'].values, where='post'
                    )
                    
                    ylim = pyplot.ylim()

                    pyplot.plot(
                        (offsets_avg[i_tel], offsets_avg[i_tel]), (0, ylim[1]),
                        linestyle='--', label=f'Averaged offset = {offsets_avg[i_tel]:.3f} [us]', color='grey'
                    )

                    pyplot.grid()
                    pyplot.legend()

                    pyplot.savefig(f'{output_dir}/profile_{tel_type}_Run{obs_id}.{subrun_id}.pdf')
                    pyplot.close()

    print('\nMaking an overview plot...')

    pyplot.figure(figsize=(50, 40))
    grid = (4, 2)

    # --- zenith angles ---
    pyplot.subplot2grid(grid, (0, 0))
    pyplot.title('narrow line = LST-1, broad line = MAGIC')
    pyplot.xlabel('Unix time [sec]')
    pyplot.ylabel('Zenith angle [deg]')

    for i, obs_id in enumerate(obs_ids_list):

        subsets_list = list(container[obs_id].keys())

        for subset in subsets_list:

            df = container[obs_id][subset].query('tel_name == "MAGIC-I"')

            label = f'ID {int(obs_id)}' if (subset == subsets_list[-1]) else None

            pyplot.plot(
                df['mean_time_unix'].values, 90 - df['mean_alt_lst'].values,
                label=label, color=color_cycle[i%10],
            )

            pyplot.plot(
                df['mean_time_unix'].values, 90 - df['mean_alt_magic'].values,
                linewidth=7, alpha=0.5, color=color_cycle[i % 10],
            )

    pyplot.grid()
    pyplot.legend(fontsize=15)

    # --- azimuth angles ---
    pyplot.subplot2grid(grid, (1, 0))
    pyplot.title('narrow line = LST-1, broad line = MAGIC')
    pyplot.xlabel('Unix time [sec]')
    pyplot.ylabel('Azimuth angle [deg]')

    for i, obs_id in enumerate(obs_ids_list):

        subsets_list = list(container[obs_id].keys())

        for subset in subsets_list:

            df = container[obs_id][subset].query('tel_name == "MAGIC-I"')

            label = f'ID {int(obs_id)}' if (subset == subsets_list[-1]) else None

            pyplot.plot(
                df['mean_time_unix'].values, df['mean_az_lst'].values,
                label=label, color=color_cycle[i % 10],
            )

            pyplot.plot(
                df['mean_time_unix'].values, df['mean_az_magic'].values,
                linewidth=7, alpha=0.5, color=color_cycle[i % 10],
            )

    pyplot.grid()
    pyplot.legend(fontsize=15)

    # --- averaged offset distribution ---
    pyplot.subplot2grid(grid, (0, 1))
    pyplot.title('solid line = (MAGIC-I - LST-1), dotted line = (MAGIC-II - LST-1)')
    pyplot.xlabel('Unix time [sec]')
    pyplot.ylabel('Averaged offset [us]')

    for i, obs_id in enumerate(obs_ids_list):

        subsets_list = list(container[obs_id].keys())

        for subset in subsets_list:

            df_m1 = container[obs_id][subset].query('tel_name == "MAGIC-I"')
            df_m2 = container[obs_id][subset].query('tel_name == "MAGIC-II"')

            label = f'ID {int(obs_id)}' if (subset == subsets_list[-1]) else None

            pyplot.plot(
                df_m1['mean_time_unix'].values, df_m1['offset_avg_us'].values,
                marker='o', markersize=2, linewidth=1, label=label, color=color_cycle[i % 10]
            )

            pyplot.plot(
                df_m2['mean_time_unix'].values, df_m2['offset_avg_us'].values,
                marker='o', markersize=2, linewidth=1, linestyle=':', color=color_cycle[i % 10]
            )

    pyplot.grid()
    pyplot.legend(fontsize=15)

    # --- difference of the averaged offset ---
    pyplot.subplot2grid(grid, (1, 1))
    pyplot.title('(MAGIC-II - LST-1) - (MAGIC-I - LST-1) = MAGIC-II - MAGIC-I')
    pyplot.xlabel('Unix time [sec]')
    pyplot.ylabel('Difference of the averaged offsets [us]')
    pyplot.ylim(-0.2, 0.2)

    for i, obs_id in enumerate(obs_ids_list):

        subsets_list = list(container[obs_id].keys())

        for subset in subsets_list:

            df_m1 = container[obs_id][subset].query('tel_name == "MAGIC-I"')
            df_m2 = container[obs_id][subset].query('tel_name == "MAGIC-II"')

            label = f'ID {int(obs_id)}' if (subset == subsets_list[-1]) else None

            pyplot.plot(
                df_m1['mean_time_unix'].values, df_m2['offset_avg_us'].values - df_m1['offset_avg_us'].values,
                marker='o', markersize=2, linewidth=1, label=label, color=color_cycle[i % 10]
            )

    pyplot.grid()
    pyplot.legend(fontsize=15)

    for i_tel, tel_name in enumerate(['MAGIC-I', 'MAGIC-II']):

        # --- number of coincident events ---
        pyplot.subplot2grid(grid, (2, i_tel))
        pyplot.title(f'LST-1 & {tel_name}')
        pyplot.xlabel('Unix time [sec]')
        pyplot.ylabel('Number of events')

        for i, obs_id in enumerate(obs_ids_list):

            subsets_list = list(container[obs_id].keys())

            for subset in subsets_list:

                df = container[obs_id][subset].query(f'tel_name == "{tel_name}"')

                label = f'ID {int(obs_id)}' if (subset == subsets_list[-1]) else None
                label_magic = tel_name if (obs_id == obs_ids_list[-1]) and (subset == subsets_list[-1]) else None

                pyplot.plot(
                    df['mean_time_unix'].values, df['n_coincidence'].values,
                    marker='o', markersize=3, linewidth=1, label=label, color=color_cycle[i % 10]
                )

                pyplot.plot(
                    df['mean_time_unix'].values, df['n_magic'].values, color='silver',
                    marker='o', markersize=3, linewidth=1, alpha=0.8, label=label_magic
                )

        pyplot.grid()
        pyplot.legend(fontsize=15)

        # --- ratio of the coincidence ---
        pyplot.subplot2grid(grid, (3, i_tel))
        pyplot.title(f'LST-1 & {tel_name}')
        pyplot.xlabel('Unix time [sec]')
        pyplot.ylabel('$N_\mathrm{coincidence}/N_\mathrm{magic}$')
        pyplot.ylim(0, 1)

        for i, obs_id in enumerate(obs_ids_list):

            subsets_list = list(container[obs_id].keys())

            for subset in subsets_list:

                df = container[obs_id][subset].query(f'tel_name == "{tel_name}"')
                
                label = f'ID {int(obs_id)}' if (subset == subsets_list[-1]) else None

                pyplot.plot(
                    df['mean_time_unix'].values, df['ratio'].values,
                    marker='o', markersize=3, linewidth=1, label=label, color=color_cycle[i % 10]
                )

        pyplot.grid()
        pyplot.legend(fontsize=15)

    pyplot.savefig(output_file)

    print(f'\nOutput directory: {output_dir}')


def main():

    start_time = time.time()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str,
        help='Path to input DL1 coincidence data files.'
    )

    arg_parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./overview.pdf',
        help='Path to an output overview file.'
    )

    arg_parser.add_argument(
        '--get-profile', '-p', dest='get_profile', type=str, default='False',
        help='If "True" the profiles of the coincidence scan will be also output.'
    )

    args = arg_parser.parse_args()

    check_coincidence_overview(args.input_data, args.output_file, args.get_profile)

    print('\nDone.')
    print(f'\nelapsed time = {time.time() - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
