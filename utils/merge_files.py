import re
import os
import glob
import argparse
import pandas as pd 
import numpy as np

__all__ = [
    'merge_data_subrun_files',
    'merge_mc_run_files'
]


def merge_data_subrun_files(input_dir):

    data_path = input_dir + '/subrun-wise/*.h5'

    paths_list = glob.glob(data_path)
    paths_list.sort()

    # --- check the run_ids ---
    run_ids_list = []

    for path in paths_list:
        re_parser = re.findall('(\w+)_Run(\d+)\.(\d+).h5', path)[0]
        run_ids_list.append(re_parser[1])

    file_name = re_parser[0]
    run_ids_list = np.unique(run_ids_list)

    outputs_list = []

    # --- merge the files ---
    for run_id in run_ids_list:

        print(f'\n--- Merge Run{run_id} ---')
        data_path = input_dir + f'/subrun-wise/*_Run{run_id}.*.h5'
        paths_list = glob.glob(data_path)
        paths_list.sort()

        data = pd.DataFrame()

        for path in paths_list:

            print(f'{path}')
            df = pd.read_hdf(path, key='events/params')
            data = data.append(df)

        output_data = input_dir + f'/{file_name}_Run{run_id}.h5'
        outputs_list.append(output_data)

        data.to_hdf(output_data, key='events/params')

    print('\nOutput data:')

    for path in outputs_list:
        print(path)


def merge_mc_run_files(input_dir, n_subsets):

    data_path = input_dir + '/run-wise/*.h5'
    
    paths_list = glob.glob(data_path)
    paths_list.sort()

    file_name = re.findall('(.*)_run(\d+)\.h5', paths_list[0].split('/')[-1])[0][0]

    # --- make subsets ---
    paths_list = np.reshape(paths_list, (n_subsets, int(len(paths_list)/n_subsets)))

    for i_sub in range(len(paths_list)):

        print(f'\n--- subset{i_sub} ---')
        data = pd.DataFrame()
        
        for path in paths_list[i_sub]:

            print(f'{path}')
            df = pd.read_hdf(path, key='events/params')
            data = data.append(df)

        output_data = input_dir + f'/subset{i_sub}.h5'
        data.to_hdf(output_data, key='events/params')

    # --- merge the subsets ---
    print('\nCombine the subsets...')

    data_path = input_dir + '/subset*.h5'
    paths_list = glob.glob(data_path)
    paths_list.sort()

    data = pd.DataFrame()
        
    for path in paths_list:

        print(f'{path}')
        df = pd.read_hdf(path, key='events/params')
        data = data.append(df)

    output_data = input_dir + f'/{file_name}.h5'

    data.to_hdf(output_data, key='events/params')

    for path in paths_list:
        os.remove(path)

    print(f'\nOutput data: {output_data}')


def main():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '--input-dir', '-i', dest='input_dir', type=str,
        help='Path to an input directory'
    )

    arg_parser.add_argument(
        '--n-subsets', '-n', dest='n_subsets', type=int, default=20, 
        help='The number of subsets that will be used for merging the MC data' 
    )
    
    args = arg_parser.parse_args()

    dirs = os.listdir(args.input_dir)

    if 'subrun-wise' in dirs: 
        merge_data_subrun_files(args.input_dir)

    elif 'run-wise' in dirs:
        merge_mc_run_files(args.input_dir, args.n_subsets)

    print('\nDone.\n')


if __name__ == '__main__':
    main()