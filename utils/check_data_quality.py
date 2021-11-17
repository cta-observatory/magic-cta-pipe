#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import re
import os
import time
import glob
import uproot
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot
from datetime import datetime, timedelta

pyplot.rcParams['figure.figsize'] = (20, 15)
pyplot.rcParams['font.size'] = 15
pyplot.rcParams['grid.linestyle'] = ':'


def plot_runsum(x, y, pos, ylabel, ylim, cut_value=None):
    
    pyplot.subplot2grid((2, 2), pos)
    pyplot.xlabel('Run number')
    pyplot.ylabel(ylabel)
    pyplot.ylim(ylim)
    
    if type(y) == dict:
        pyplot.plot(x, y[1], marker='o', linewidth=1, markersize=10, label='MAGIC-I')
        pyplot.plot(x, y[2], marker='o', linewidth=1, markersize=10, label='MAGIC-II')
        pyplot.legend()
        
    else:
        pyplot.plot(x, y, marker='o', linewidth=1, markersize=10)
        
    if cut_value is not None:
                
        pyplot.plot(
            pyplot.xlim(), (cut_value, cut_value), 
            linewidth=2, linestyle='--', color='grey', label=f'cut = {cut_value}'
        )
        
        pyplot.legend()
        
    pyplot.grid()


def check_data_quality(input_data_mask):

    quality_cuts = {
        'transmission': 0.7,
        'nstar': 15,
        'rate': 100
    }

    print(f'\nQuality cuts:\n{quality_cuts}')

    paths_list = glob.glob(input_data_mask)
    paths_list.sort()

    bad_obs_ids = np.array([])

    print('\nChecking the Out.root files:\n')

    for path in paths_list:

        print(path)

        parent_dir = str(Path(path).parent)

        with uproot.open(path) as f:

            obs_ids = np.array(f['RunSum;1']['MRunSummary_1/fRunNumber'].array())
            zeniths = np.array(f['RunSum;1']['MRunSummary_1/fZd'].array())
            transmissions = np.array(f['RunSum;1']['MRunSummary_1/fAerosolTrans9km'].array())
            rates = np.array(f['RunSum;1']['MRunSummary_1/fL3Rate'].array())
            
            nstars = {
                1: np.array(f['RunSum;1']['MRunSummary_1/fNumStars'].array()),
                2: np.array(f['RunSum;1']['MRunSummary_2/fNumStars'].array())
            }

        pyplot.figure()

        plot_runsum(obs_ids, zeniths, (0, 0), 'Zenith angle [deg]', (0, 90))
        plot_runsum(obs_ids, transmissions, (1, 0), 'Transmission @ 9km', (0, 1), quality_cuts['transmission'])
        plot_runsum(obs_ids, rates, (0, 1), 'L3 rate [Hz]', (0, 600), quality_cuts['rate'])
        plot_runsum(obs_ids, nstars, (1, 1), 'Number of stars', (0, 60), quality_cuts['nstar'])

        pyplot.savefig(f'{parent_dir}/data_quality.pdf')
        pyplot.close()

        condition_trans = (transmissions <= 0.1) | (transmissions >= quality_cuts['transmission'])
        condition_nstars = (nstars[1] >= quality_cuts['nstar']) & (nstars[2] >= quality_cuts['nstar'])
        condition_rates = (rates >= quality_cuts['rate'])

        condition = (condition_trans & condition_nstars & condition_rates)

        print(f'Observation IDs: {obs_ids}')
        print(f'--> {condition}')
        print(f'good runs = {np.sum(condition)}, bad runs = {np.sum(~condition)}\n')

        bad_obs_ids = np.append(bad_obs_ids, obs_ids[~condition])

    bad_obs_ids = bad_obs_ids.astype(int)

    print('\n=== Summary ===')

    print(f'\nObservation IDs of bad quality data:')
    print(*bad_obs_ids)


def main():

    start_time = time.time()

    # --- get the arguments ---
    arg_parser = argparse.ArgumentParser() 

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str, 
        help='Path to Out.root data files produced by MARS "quate" program.'
    )

    args = arg_parser.parse_args()

    # --- check data quality ---
    check_data_quality(args.input_data)

    print('\nDone.')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()