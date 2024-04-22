"""
This script creates the lists of MAGIC and LST runs (date and run number) from a dataframe in the .h5 format for a specific time range.
"""

import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import yaml
import argparse


def split_lst_date(df):

    """
    This function appends to the provided dataframe, which contains the LST date as YYYYMMDD in one of the columns, four new columns: the LST year, month and day and the date as YYYY_MM_DD

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe of the joint MAGIC+LST-1 observations based on the .h5 table.

    Returns
    -------
    :class:`pandas.DataFrame`
        The input dataframe with four added columns.
    """

    date = df["DATE"]
    df["YY_LST"] = date.str[:4]
    df["MM_LST"] = date.str[4:6]
    df["DD_LST"] = date.str[6:8]
    df["date_LST"] = df["YY_LST"] + "-" + df["MM_LST"] + "-" + df["DD_LST"]
    return df


def magic_date(df):

    """
    This function appends to the provided dataframe (which contains the LST date, year, month and day) a column with the MAGIC dates (in the YYYY_MM_DD format).

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe of the joint MAGIC+LST-1 observations based on the .h5 table.

    Returns
    -------
    :class:`pandas.DataFrame`
        The input dataframe with an added column.
    """

    date_lst = pd.to_datetime(df["Date (LST convention)"], format="%Y_%m_%d")
    delta = pd.Timedelta("1 day")
    date_magic = date_lst + delta
    date_magic = date_magic.dt.strftime("%Y-%m-%d")
    df["date_MAGIC"] = date_magic
    return df

def clear_files(source_in, source_out, df):

    """
    This function deletes any file named XXXX_LST_runs.txt and XXXX_MAGIC_runs.txt from the working directory.

    Parameters
    ----------
    source_in : string
        Target name in the database. If None, it stands for all the sources observed in a pre-set time interval.
    source_out: string
        Name tag for the target. Used only if source_in is not None.
    df: :class:`pandas.DataFrame`
        Dataframe of the joint MAGIC+LST-1 observations based on the .h5 table.

    Returns
    -------

    """

    source_list = []
    if source_in is None:
        source_list = np.unique(df["source"])
    else:
        source_list.append(source_out)
   
    print("Source list: ",source_list)
    joblib.dump(source_list, "list_sources.dat")
    for source_name in source_list:
        print("Target name: ",source_name)
        file_list = [
            f"{source_name}_LST_runs.txt",
            f"{source_name}_MAGIC_runs.txt",
        ]  # The order here must be LST before MAGIC!
        print(file_list)
        for j in file_list:
            if os.path.isfile(j):
                os.remove(j)
                print(f"{j} deleted.")
    

def list_run(source_in, source_out, df, skip_LST, skip_MAGIC, is_LST, M1_run_list=None):

    """
    This function creates the MAGIC_runs.txt and LST_runs.txt files, which contain the list of runs (with corresponding dates) to be processed.

    Parameters
    ----------
    source_in : str or None
        Name of the source in the database of joint observations. If None, it will process all sources for the given time range.
    source_out : str
        Name of the source to be used in the output file name. Useful only if source_in != None.
    df : :class:`pandas.DataFrame`
        Dataframe of the joint MAGIC+LST-1 observations.
    skip_LST : list
        List of the LST runs to be ignored.
    skip_MAGIC : list
        List of the MAGIC runs to be ignored.
    is_LST : bool
        If you are looking for LST runs, set to True. For MAGIC set False.

    """

    source_list = []
    if source_in is None:
        source_list = np.unique(df["source"])
        
    else:
        source_list.append(source_out)
    
    print("List of sources: ",source_list)
    for source_name in source_list:
        file_list = [
            f"{source_name}_LST_runs.txt",
            f"{source_name}_MAGIC_runs.txt",
        ]  # The order here must be LST before MAGIC!
        
        run_listed = []
        if source_in is None:
            df_source = df[df["source"] == source_name]
        else:
            df_source = df[df["source"] == source_in]

        if is_LST:
            print('LST')
            LST_run = df_source["LST1_run"].tolist()  # List with runs as strings
            LST_date = df_source["date_LST"].tolist()
            for k in range(len(df_source)):
                skip = False
                if np.isnan(LST_run[k]):
                    skip = True

                if (int(LST_run[k]) in skip_LST) or (int(LST_run[k]) in run_listed):
                    skip = True
                

                if skip is False:
                    with open(file_list[0], "a+") as f:
                        f.write(
                            f"{LST_date[k].replace('-','_')},{str(LST_run[k]).lstrip('0')}\n"
                        )
                    run_listed.append(int(LST_run[k]))

        if not is_LST:
            print('MAGIC')
            MAGIC_date = df_source["date_MAGIC"].tolist()
            M2_run=df_source['Run ID'].tolist()
            for k in range(len(df_source)):
                skip = False
                if np.isnan(M2_run[k]):
                    skip = True

                if (int(M2_run[k]) in skip_MAGIC) or (int(M2_run[k]) in run_listed):
                    skip = True
                if int(M2_run[k]) not in M1_run_list:
                    skip = True

                if skip is False:
                    with open(file_list[1], "a+") as f:
                        f.write(
                            f"{MAGIC_date[k].replace('-','_')},{int(M2_run[k])}\n"
                        )
                    run_listed.append(int(M2_run[k]))


def main():

    """
    This function is automatically called whe script is launched. 
    It calls the functions above to create the files XXXXXX_LST_runs.txt and XXXXX_MAGIC_runs.txt for the desired targets.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_general.yaml",
        help="Path to a configuration file config_general.yaml",
    )

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    source_in = config["data_selection"]["source_name_database"]
    source_out = config["data_selection"]["source_name_output"]
    range = config["data_selection"]["time_range"]
    skip_LST = config["data_selection"]["skip_LST_runs"]
    skip_MAGIC = config["data_selection"]["skip_MAGIC_runs"]

    df_LST = pd.read_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5", key="joint_obs"
    )  # TODO: put this file in a shared folder

    df_LST = split_lst_date(df_LST)
    df_LST = df_LST.astype({"YY_LST": int, "MM_LST": int, "DD_LST": int})

    stereo = True
    if source_in is None:
        df_LST.query(
            f'MAGIC_trigger=="L3T" & MAGIC_HV=="Nominal" & MAGIC_stereo == {stereo}',
            inplace=True,
        )
    else:
        df_LST.query(
            f'source=="{source_in}"& MAGIC_trigger=="L3T" & MAGIC_HV=="Nominal" & MAGIC_stereo == {stereo}',
            inplace=True,
        )
    
    if range:
        min = str(config["data_selection"]["min"])
        max = str(config["data_selection"]["max"])
        min = datetime.strptime(min, "%Y_%m_%d")
        max = datetime.strptime(max, "%Y_%m_%d")
        lst = pd.to_datetime(df_LST["date_LST"].str.replace("_", "-"))
        df_LST["date"] = lst
        df_LST = df_LST[df_LST["date"] >= min]
        df_LST = df_LST[df_LST["date"] <= max]

    else:
        dates = config["data_selection"]["date_list"]
        df_LST = df_LST[df_LST["date_LST"].isin(dates)]

    df_LST = df_LST.reset_index()
    df_LST = df_LST.drop("index", axis=1)

    clear_files(source_in, source_out, df_LST)
    list_run(source_in, source_out, df_LST, skip_LST, skip_MAGIC, True)
    list_date_LST=np.unique(df_LST['date_LST'])
    list_date_LST_low=[sub.replace('-', '_') for sub in list_date_LST]

    print(list_date_LST_low)
    df_MAGIC1=pd.read_hdf('/fefs/aswg/workspace/joanna.wojtowicz/data/Common_MAGIC_LST1_data_MAGIC_RUNS.h5', key='MAGIC1/runs_M1')
    df_MAGIC2=pd.read_hdf('/fefs/aswg/workspace/joanna.wojtowicz/data/Common_MAGIC_LST1_data_MAGIC_RUNS.h5', key='MAGIC2/runs_M2')
    
    print(list_date_LST)
    df_MAGIC1=df_MAGIC1[df_MAGIC1['Date (LST convention)'].isin(list_date_LST_low)]
    df_MAGIC2=df_MAGIC2[df_MAGIC2['Date (LST convention)'].isin(list_date_LST_low)]
    print(df_MAGIC2)
    
    df_MAGIC2=magic_date(df_MAGIC2)
    df_MAGIC1=magic_date(df_MAGIC1)
    df_MAGIC2 = df_MAGIC2.rename(columns={'Source': 'source'})
    print(df_MAGIC2)

    M1_runs=df_MAGIC1['Run ID'].tolist()
    list_run(source_in, source_out, df_MAGIC2, skip_LST, skip_MAGIC, False, M1_runs)
    

if __name__ == "__main__":
    main()
