"""
Bash scripts to run LSTnsb.py on all the LST runs by using parallel jobs

Usage: python nsb_level.py (-c config.yaml)
"""

import argparse
import glob
import logging
import os
import pandas as pd
from datetime import datetime

import numpy as np
import yaml

__all__ = ["bash_scripts"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def collect_nsb(df_LST):
    """
    Here we split the LST runs in NSB-wise .txt files

    Parameters
    ----------
    config : dict
        Configuration file
    """
    nsb_files=glob.glob('nsb_LST_*.txt')
    for file_nsb in nsb_files:
        run=file_nsb.split('_')[3]
        nsb=np.nan
        with open(file_nsb) as ff:
            line_str = ff.readline().rstrip("\n")
            nsb=line_str.split(',')[2]
        df_LST=df_LST.set_index("LST1_run")
        df_LST.loc[df_LST.index[run], 'nsb']=nsb
        df_LST=df_LST.reset_index()
        






def main():

    """
    Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_general.yaml",
        help="Path to a configuration file",
    )
    parser.add_argument(
        "--begin-date",
        "-b",
        dest="begin_date",
        type=str,
        help="Begin date to start NSB evaluation from the database.",
    )
    parser.add_argument(
        "--end-date",
        "-e",
        dest="end_date",
        type=str,
        help="End date to start NSB evaluation from the database.",
    )
    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    env_name = config["general"]["env_name"]

    df_LST = pd.read_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5", key="joint_obs"
    )
    min = datetime.strptime(args.begin_date, "%Y_%m_%d")
    max = datetime.strptime(args.end_date, "%Y_%m_%d")
    lst = pd.to_datetime(df_LST["date_LST"].str.replace("_", "-"))
    df_LST["date"] = lst
    df_LST = df_LST[df_LST["date"] >= min]
    df_LST = df_LST[df_LST["date"] <= max]
    

    df_new=collect_nsb(df_LST)

    df_old=pd.read_hdf('/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5', key="joint_obs")
    df_new=pd.concat([df_old, df_new]).drop_duplicates(keep='first')
    df_new= df_new.sort_values(by=["DATE","source"])
    

    df_new.to_hdf("/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5", key="joint_obs", mode="w")
if __name__ == "__main__":
    main()
