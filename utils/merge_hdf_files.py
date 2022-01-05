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

__all__ = ['merge_hdf_files']


def merge_hdf_files(input_files, output_file=None, n_files=50):

    file_paths = glob.glob(input_files)
    file_paths.sort()

    # --- merge the input data to subsets ---
    print('\nMerging the input data to subsets:')

    output_dir_tmp = str(Path(file_paths[0]).parent)

    subset_paths = []
    data_merged = pd.DataFrame()

    for i_file, path in enumerate(file_paths):

        print(path)

        if len(data_merged) == 0:
            file_name_start = re.findall('(\S+).h5', Path(path).name)[0]

        df = pd.read_hdf(path, key='events/params')
        data_merged = data_merged.append(df)

        if ( (i_file+1) % n_files == 0 ) or ( path == file_paths[-1] ):

            file_name_end = re.findall('(\S+).h5', Path(path).name)[0]
            subset_file = f'{output_dir_tmp}/subset_{file_name_start}_to_{file_name_end}.h5'

            subset_paths.append(subset_file)

            data_merged.to_hdf(subset_file, key='events/params')
            data_merged = pd.DataFrame()
            
            print(f'--> {subset_file}\n')

    # --- merge the subsets ---
    print('Merging the subsets:')

    for path in subset_paths:

        print(Path(path).name)

        df = pd.read_hdf(path, key='events/params')
        data_merged = data_merged.append(df)

        os.remove(path)

    if output_file != None:
        
        # --- save the data frame ---
        output_dir = str(Path(output_file).parent)
        os.makedirs(output_dir, exist_ok=True)

        data_merged.to_hdf(output_file, key='events/params')

        print(f'\nOutput data file: {output_file}')

    return data_merged


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-files', '-i', dest='input_files', type=str,
        help='Path to input HDF data files.'
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./merged_data.h5',
        help='Path to an output HDF data file. The output directory will be created if it does NOT exist.'
    )

    parser.add_argument(
        '--n-files', '-n', dest='n_files', type=int, default=50,
        help='Number of data files merged to a subset.'
    )

    args = parser.parse_args()

    merge_hdf_files(args.input_files, args.output_file, args.n_files) 

    print('\nDone.')

    end_time = time.time()
    print(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()