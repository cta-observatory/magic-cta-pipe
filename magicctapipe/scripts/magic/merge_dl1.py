import sys
import glob
import argparse
import pandas as pd

arg_parser = argparse.ArgumentParser(description="""This tools merges several DL1 files.""")

arg_parser.add_argument("--mc", action="store_true", help="Run on MC files")
arg_parser.add_argument("--stereo", action="store_true", help="Run on stereo files")
arg_parser.add_argument("--outfile", help="Name of output file.")
arg_parser.add_argument("--infiles", help="Input files.")

parsed_args = arg_parser.parse_args()

outfile = parsed_args.outfile
infiles = parsed_args.infiles
is_mc   = parsed_args.mc
stereo  = parsed_args.stereo

df_list_hillas = []
df_list_stereo = []
df_list_mch = []

for filename in sorted(glob.glob(infiles)):
    print(filename)
    store = pd.HDFStore(filename)
    df = store['/dl1/hillas_params']
    df_list_hillas.append(df)
    if stereo:
        df1 = store['/dl1/stereo_params']
        df_list_stereo.append(df1)
    if is_mc:
        df2 = store['/dl1/mc_header']
        df_list_mch.append(df2)
    store.close()

df_merge = pd.concat(df_list_hillas, ignore_index=True, sort=False)
df_merge.to_hdf(outfile, key="/dl1/hillas_params", mode='w')
if stereo:
    df_merge_1 = pd.concat(df_list_stereo, ignore_index=True, sort=False)
    df_merge_1.to_hdf(outfile, key="/dl1/stereo_params", mode='a')
if is_mc:
    df_merge_2 = pd.concat(df_list_mch, ignore_index=True, sort=False)
    df_merge_2.to_hdf(outfile, key="/dl1/mc_header", mode='a')
