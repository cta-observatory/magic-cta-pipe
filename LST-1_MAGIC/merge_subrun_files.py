import re
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
args = arg_parser.parse_args()

# ============
# === Main === 
# ============

input_mask = args.input_dir + '/subrun-wise/*.h5'
data_paths = glob.glob(input_mask)
data_paths.sort()

if data_paths == []:
    print('No input files are found. Please check the path to the directory. Exiting.')
    sys.exit()

run_ids = []

for data_path in data_paths:
    re_parser = re.findall('(\w+)_Run(\d+)\.(\d+).h5', data_path)[0]
    run_ids.append(re_parser[1])

file_name = re_parser[0]

for run_id in np.unique(run_ids):
    print(f'=== Run{run_id} ===')
    input_mask = args.input_dir + f'/subrun-wise/*_Run{run_id}.*.h5'
    data_paths = glob.glob(input_mask)
    data_paths.sort()

    data_stereo = pd.DataFrame()
    for data_path in data_paths:
        print(f'Combine {data_path}...')
        df = pd.read_hdf(data_path, key='events/params')
        data_stereo = pd.concat([data_stereo, df])
        del df

    output_file = args.input_dir + f'/{file_name}_Run{run_id}.h5'
    data_stereo.to_hdf(output_file, key='events/params')
    del data_stereo
