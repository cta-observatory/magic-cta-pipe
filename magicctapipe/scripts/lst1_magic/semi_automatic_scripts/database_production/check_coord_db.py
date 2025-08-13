"""
Check if source coordinates stored in the LSt database 
(from astropy or from a json dictionary) are consistehnt 
with the ones stored in one of the MAGIC files for each 
source. Note: only one MAGIC subrun per source is used!

Usage:
$ check_coord_db (-c config_auto_MCP.yaml)
"""

import os

import numpy as np
import pandas as pd
import yaml
import glob
import uproot
import math

from magicctapipe.io import resource_file
from magicctapipe.utils import auto_MCP_parser

degrees_per_hour = 15.0
seconds_per_hour = 3600.0

def main():

    """
    Main function
    """

    parser = auto_MCP_parser()
    args = parser.parse_args()
    
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config_general = yaml.safe_load(f)

    config = config_general["general"]["base_db_config_file"]
    if config == "":
        config = resource_file("database_config.yaml")

    with open(
        config, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)

    lst_h5 = config_dict["database_paths"]["LST"]
    lst_key = config_dict["database_keys"]["LST"]
    lst_df=pd.read_hdf(lst_h5, lst_key)
    df = pd.read_hdf(
        config_dict["database_paths"]["MAGIC+LST1"],
        key=config_dict["database_keys"]["MAGIC+LST1"],
    )  # TODO: put this file in a shared folder
    df2 = pd.read_hdf(
        config_dict["database_paths"]["MAGIC+LST1_bis"],
        key=config_dict["database_keys"]["MAGIC+LST1_bis"],
    )  # TODO: put this file in a shared folder
    df = pd.concat([df, df2]).drop_duplicates(subset="LST1_run", keep="first")
    
    df=df.drop_duplicates(subset=['source'])

    date_lst = pd.to_datetime(df["DATE"], format="%Y%m%d")

    delta = pd.Timedelta("1 day")
    date_magic = date_lst + delta

    date_magic = date_magic.dt.strftime("%Y/%m/%d").to_list()

    for i in range(len(df)):
        magic_first_run = (df["MAGIC_runs"].to_list())[i].rstrip("]").lstrip("[").split(", ")[0]
        lst_run = (df["LST1_run"].to_list())[i]
        source = (df["source"].to_list())[i]
        magic_file = glob.glob(
            f"/fefs/onsite/common/MAGIC/data/M[12]/event/Calibrated/{date_magic[i]}/*{magic_first_run}*{source}*.root"
        ) [0]
        ra_db= lst_df.loc[lst_df['LST1_run']==lst_run][0]['ra']
        dec_db= lst_df.loc[lst_df['LST1_run']==lst_run][0]['dec']
        file_tree=uproot.open(magic_file)
        branch_list = [
            "MRawRunHeader.fSourceRA",
            "MRawRunHeader.fSourceDEC",
        ]
        coord = magic_file["RunHeaders"].arrays(
            branch_list, library="np"
        )
        ra_file=coord["MRawRunHeader.fSourceRA"][0] / seconds_per_hour* degrees_per_hour
    

        dec_file=coord["MRawRunHeader.fSourceDEC"][0]/ seconds_per_hour 
        if (math.isclose(ra_db, ra_file, abs_tol=0.001)) and (math.isclose(dec_db, dec_file, abs_tol=0.001)):
            continue
        else:
            print(f'coordinates not equal in MAGIc Calibrated files and in LST database for {source}')
            print(f'database RA = {ra_db}, DEC ={dec_db}; MAGIC ROOT files RA = {ra_file}, DEC = {dec_file}')






    


if __name__ == "__main__":
    main()
