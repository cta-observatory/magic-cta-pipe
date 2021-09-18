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

    path_list = glob.glob(data_path)
    path_list.sort()

    # --- check the run_ids ---
    run_ids = []

    for path in path_list:
        re_parser = re.findall('(\w+)_Run(\d+)\.(\d+).h5', path)[0]
        run_ids.append(re_parser[1])

    file_name = re_parser[0]
    run_ids = np.unique(run_ids)

    output_list = []

    # --- merge the files ---
    for run_id in run_ids:

        print(f'\n--- Merge Run{run_id} ---')
        data_path = input_dir + f'/subrun-wise/*_Run{run_id}.*.h5'
        path_list = glob.glob(data_path)
        path_list.sort()

        data_stereo = pd.DataFrame()

        for path in path_list:

            print(f'{path}')
            df = pd.read_hdf(path, key='events/params')
            data_stereo = pd.concat([data_stereo, df])

        output_data = input_dir + f'/{file_name}_Run{run_id}.h5'
        output_list.append(output_data)

        data_stereo.to_hdf(output_data, key='events/params')

    print('\nOutput data:')
    for path in output_list:
        print(path)

def merge_mc_run_files(input_dir, n_subsets):
    
    print(f'\nChecking the input directory {args.input_dir}')

    input_mask = args.input_dir + '/*run-wise/*.h5'
    data_paths = glob.glob(input_mask)
    data_paths.sort()

    print(f'--> {len(data_paths)} input files\n')

    re_parser = re.findall('(.*)_run(\d+)\.h5', data_paths[0].split('/')[-1])[0]
    file_name = re_parser[0]

    data_paths = np.reshape(data_paths, (args.num_subsets, int(len(data_paths)/args.num_subsets)))

    for i_col in range(len(data_paths)):
        print(f'=== subset {i_col} ===')
        data = pd.DataFrame()
        
        for path in data_paths[i_col]:
            print(f'Combine {path}...')
            df = pd.read_hdf(path, key='events/params')
            data = pd.concat([data, df])
            del df

        output_file = args.input_dir + f'/subset{i_col}.h5'
        data.to_hdf(output_file, key='events/params')

    print('=== Combine the subsets ===')

    input_mask = args.input_dir + '/subset*.h5'
    data_paths = glob.glob(input_mask)
    data_paths.sort()

    data = pd.DataFrame()
        
    for path in data_paths:
        print(f'Combine {path}...')
        df = pd.read_hdf(path, key='events/params')
        data = pd.concat([data, df])
        del df

    if args.suffix != None:
        output_path = args.input_dir + f'/{file_name}_{args.suffix}.h5'
    else:
        output_path = args.input_dir + f'/{file_name}.h5'

    data.to_hdf(output_path, key='events/params')

    for path in data_paths:
        os.remove(path)

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