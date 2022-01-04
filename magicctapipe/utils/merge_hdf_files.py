#!/usr/bin/env python
# coding: utf-8

# Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

import os
import re 
import time
import glob
import argparse
import pandas as pd 
import numpy as np
from pathlib import Path

__all__ = [
    'merge_hdf_files'
]


def merge_hdf_files(input_data_mask, n_files=50, output_data=None):

    paths_list = glob.glob(input_data_mask)
    paths_list.sort()

    # --- merge the input data to subsets ---
    print('\nMerging the input data to subsets...')

    dir_tmp = str(Path(paths_list[0]).parent)

    data_merged = pd.DataFrame()
    subsets_list = []

    for i_file, path in enumerate(paths_list):

        print(path)

        if len(data_merged) == 0:
            filename_start = re.findall('(\S+).h5', path.split('/')[-1])[0]

        df = pd.read_hdf(path, key='events/params')
        data_merged = data_merged.append(df)

        if ( (i_file+1) % n_files == 0 ) or ( path == paths_list[-1] ):

            filename_end = re.findall('(\S+).h5', path.split('/')[-1])[0]
            output_data_subset = f'{dir_tmp}/subset_{filename_start}_to_{filename_end}.h5'

            data_merged.to_hdf(output_data_subset, key='events/params')
            subsets_list.append(output_data_subset)

            data_merged = pd.DataFrame()

            print(f'--> {output_data_subset}\n')
            

    # --- merge the subset data ---
    print('Merging the subset data:')

    for path in subsets_list:

        print(path)

        df = pd.read_hdf(path, key='events/params')
        data_merged = data_merged.append(df)

        os.remove(path)

    if output_data != None:
        
        # --- save the data frame ---
        data_merged.to_hdf(output_data, key='events/params')

        print(f'\nOutput data: {output_data}')

    return data_merged


def main():

    start_time = time.time()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str,
        help='Path to input data files with h5 extention.'
    )

    arg_parser.add_argument(
        '--n-files', '-n', dest='n_files', type=int, default=50,
        help='Number of data files merged to a subset data file.'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, default='./merged_data.h5',
        help='Path to an output data file with h5 extention.'
    )

    args = arg_parser.parse_args()

    merge_hdf_files(args.input_data, args.n_files, args.output_data)

    print('\nDone.')
    print(f'\nelapsed time = {time.time() - start_time:.0f} [sec]\n')  


if __name__ == '__main__':
    main()