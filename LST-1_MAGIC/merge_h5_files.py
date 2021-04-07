import re
import os
import sys
import glob
import argparse
import pandas as pd 
import numpy as np

# ========================
# === Get the argument === 
# ========================

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--input-dir', '-i', dest='input_dir', type=str, default=None, help='Path to the input directory')
arg_parser.add_argument('--output-file', '-o', dest='output_file', type=str, default=None, help='Name of the output file')

args = arg_parser.parse_args()

# ============
# === Main === 
# ============

print(f'\nChecking the input directory {args.input_dir}')

input_mask = args.input_dir + '/*run-wise/*.h5'
data_paths = glob.glob(input_mask)
data_paths.sort()

if data_paths == []:
    print('Erorr: Failed to find the input files. Please check the path to the directory. Exiting.')
    sys.exit()

print(f'--> {len(data_paths)} input files\n')

data_paths = np.reshape(data_paths, (20, int(len(data_paths)/20)))

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

output_path = args.input_dir + '/' + args.output_file
data.to_hdf(output_path, key='events/params')

for path in data_paths:
    os.remove(path)