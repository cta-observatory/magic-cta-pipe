import re 
import os
import time
import glob
import argparse
import pandas as pd 
import numpy as np
from pathlib import Path
from utils import get_obs_ids_from_name

__all__ = [
    'merge_hdf_files'
]


def merge_hdf_files(input_data_mask, n_files=50, output_data=None):

    paths_list = glob.glob(input_data_mask)
    paths_list.sort()

    dir_tmp = str(Path(paths_list[0]).parent)

    # --- merge the files to subset files ---
    print('\nMerging the input data files to subsets...')

    i_subset = 1
    data_subset = pd.DataFrame()

    print(f'Subset {i_subset}')

    for i_file, path in enumerate(paths_list):

        print(path)

        df = pd.read_hdf(path, key='events/params')
        data_subset = data_subset.append(df)

        if path == paths_list[-1]:
            data_subset.to_hdf(f'{dir_tmp}/subset{i_subset}.h5', key='events/params')

        elif (i_file+1) % n_files == 0:

            data_subset.to_hdf(f'{dir_tmp}/subset{i_subset}.h5', key='events/params')

            i_subset += 1
            data_subset = pd.DataFrame()

            print(f'Subset {i_subset}')

    # --- merge the subset files ---
    paths_list = glob.glob(f'{dir_tmp}/subset*.h5')
    paths_list.sort()

    data_merged = pd.DataFrame()

    print('\nMerging the subset files:')

    for path in paths_list:

        print(path)

        df = pd.read_hdf(path, key='events/params')
        data_merged = data_merged.append(df)

        os.remove(path)

    if output_data != None:
        
        output_dir = str(Path(output_data).parent)
        os.makedirs(output_dir, exist_ok=True)

        data_merged.to_hdf(output_data, key='events/params')

        print(f'\nOutput data file: {output_data}')

    return data_merged


def merge_hdf_files_run_wise(input_data_mask, n_files):

    # --- get observation IDs ---
    obs_ids_list = get_obs_ids_from_name(input_data_mask)

    print(f'\nFound the following observation IDs: {obs_ids_list}')

    # --- merge the input data run-wise ---
    parent_dir = str(Path(input_data_mask).parent)
    output_dir = f'{parent_dir}/merged'

    print(f'\nOutput directory: {output_dir}')

    for obs_id in obs_ids_list:

        print(f'\nRun{obs_id}:')

        data_mask = f'{parent_dir}/*Run{obs_id}*'

        file_name = re.findall('(\w+)_Run.*', glob.glob(data_mask)[0])[0]

        output_data = f'{output_dir}/{file_name}_Run{obs_id}.h5'

        merge_hdf_files(data_mask, n_files, output_data)


def main():

    start_time = time.time()

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-data', '-i', dest='input_data', type=str,
        help='Path to input data files.'
    )

    arg_parser.add_argument(
        '--n-files', '-n', dest='n_files', type=int, default=50,
        help='Number of data files merged to a subset file.'
    )

    arg_parser.add_argument(
        '--run-wise', '-r', dest='run_wise', type=str, default='False',
        help='Whether the input data files are merged run-wise or not, "True" or "False"'
    )

    arg_parser.add_argument(
        '--output-data', '-o', dest='output_data', type=str, default=None,
        help='Path to an output data file. The output directory will be created if it does not exist.'
    )

    args = arg_parser.parse_args()

    if args.run_wise == 'True':
        merge_hdf_files_run_wise(args.input_data, args.n_files)

    elif args.run_wise == 'False':
        merge_hdf_files(args.input_data, args.n_files, args.output_data)

    print('\nDone.')
    print(f'\nelapsed time = {time.time() - start_time:.0f} [sec]\n')  


if __name__ == '__main__':
    main()