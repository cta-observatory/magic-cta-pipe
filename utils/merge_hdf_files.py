import re
import os
import time
import glob
import argparse
import pandas as pd 
import numpy as np
from pathlib import Path

__all__ = [
    'merge_hdf_files'
]


def merge_hdf_files(data_path, n_files_subsets=50):

    input_dir = str(Path(data_path).parent)

    paths_list = glob.glob(data_path)
    paths_list.sort()

    # --- merge the files to subsets ---
    print('\nMerging the input data files to subsets...')

    i_subset = 1
    data_subset = pd.DataFrame()

    print(f'\nSubset {i_subset}')

    for i_file, path in enumerate(paths_list):

        print(path)

        df = pd.read_hdf(path, key='events/params')
        data_subset = data_subset.append(df)

        if (i_file+1) % n_files_subsets == 0:

            data_subset.to_hdf(input_dir+f'/subset{i_subset}.h5', key='events/params')

            i_subset += 1
            data_subset = pd.DataFrame()

            print(f'\nSubset {i_subset}')

        elif path == paths_list[-1]:

            data_subset.to_hdf(input_dir+f'/subset{i_subset}.h5', key='events/params')

    # --- merge the subsets ---
    paths_list = glob.glob(input_dir+'/subset*.h5')
    paths_list.sort()

    data_merged = pd.DataFrame()

    print('\nMerging the subset files...')

    for path in paths_list:

        print(path)

        df = pd.read_hdf(path, key='events/params')
        data_merged = data_merged.append(df)

        os.remove(path)

    return data_merged


def main():

    start_time = time.time()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str,
        help='Path to input HDF files to be combined, e.g., dl1_stereo_lst1_magic_run*.h5'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str,  
        help='Path and name of an output data file with HDF5 format, e.g., dl1_stereo_lst1_magic_merged.h5'
    )

    arg_parser.add_argument(
        '--n-files-subsets', '-n', dest='n_files_subsets', type=int, default=50, 
        help='The number of files for each subset, default is 50' 
    )
    
    args = arg_parser.parse_args()

    # --- merge the input HDF data files ---
    data_merged = merge_hdf_files(args.input_data, args.n_files_subsets)

    # --- store the merged HDF data ---
    data_merged.to_hdf(args.output_data, key='events/params')

    print(f'\nOutput data file: {args.output_data}')

    print('\nDone.\n')

    end_time = time.time()
    print(f'\nelapsed time = {end_time - start_time:.0f} [sec]\n')  


if __name__ == '__main__':
    main()