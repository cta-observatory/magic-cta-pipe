"""
By using this scrip, the list of MAGIC and LST runs (date and run number) can be automatically created from a dataframe in the .h5 format
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import yaml


def split_lst_date(df):

    """
    This function appends to the provided dataframe, which contains the LST date as YYYYMMDD in one of the columns, four new columns: the LST year, month and day and the date as YYYY_MM_DD

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe of the joint MAGIC+LST-1 observations

    Returns
    -------
    :class:`pandas.DataFrame`
        The input dataframe with some added columns
    """

    date = df["DATE"]
    df["YY_LST"] = date.str[:4]
    df["MM_LST"] = date.str[4:6]
    df["DD_LST"] = date.str[6:8]    
    df["date_LST"] = df["YY_LST"]+ '-' + df["MM_LST"] + '-' + df["DD_LST"]  
    return df


def magic_date(df):

    """
    This function appends to the provided dataframe, which contains the LST date, year, month and day, a column with the MAGIC date (in the YYYY_MM_DD format)

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe of the joint MAGIC+LST-1 observations

    Returns
    -------
    :class:`pandas.DataFrame`
        The input dataframe with an added column
    """

    date_lst = pd.to_datetime(df['date_LST'])
    delta = pd.Timedelta("1 day")
    date_magic = date_lst + delta
    date_magic = date_magic.dt.strftime("%Y-%m-%d")
    df["date_MAGIC"] = date_magic
    return df


def list_run(source_out, df, skip_LST, skip_MAGIC):

    """
    This function create the MAGIC_runs.txt and LST_runs.txt files, which contain the list of runs (with date) to be processed

    Parameters
    ----------
    source_out : str
        Name of the source to be used in the output file name
    df : :class:`pandas.DataFrame`
        Dataframe of the joint MAGIC+LST-1 observations
    skip_LST : list
        List of the LST runs not to be added to the files
    skip_MAGIC : list
        List of the MAGIC runs not to be added to the files
    """

    file_list = [
        f"{source_out}_LST_runs.txt",
        f"{source_out}_MAGIC_runs.txt",
    ]  # LST, MAGIC!!!!
    for j in file_list:
        if os.path.isfile(j):
            os.remove(j)
            print(f"{j} deleted.")
    MAGIC_listed = []
    LST_listed = []
    for k in range(len(df)):
        skip = False
        LST = df["LST1_run"]

        if (int(LST[k]) in skip_LST) or (int(LST[k]) in LST_listed):
            skip = True

        if not skip:
            with open(file_list[0], "a+") as f:
                f.write(f"{df['date_LST'][k].replace('-','_')},{str(LST[k]).lstrip('0')}\n")
            LST_listed.append(int(LST[k]))
        MAGIC_min = int(df["MAGIC_first_run"][k])
        MAGIC_max = int(df["MAGIC_last_run"][k])
        for z in range(MAGIC_min, MAGIC_max + 1):
            skip = False

            if (int(z) in skip_MAGIC) or (int(z) in MAGIC_listed):
                skip = True
            if not skip:
                with open(file_list[1], "a+") as f:
                    f.write(f"{df['date_MAGIC'][k].replace('-','_')},{z}\n")
                MAGIC_listed.append(int(z))


def main():

    """
    Main function
    """

    with open("config_h5.yaml", "rb") as f:
        config = yaml.safe_load(f)
    df = pd.read_hdf(
        "/fefs/aswg/workspace/federico.dipierro/simultaneous_obs_summary.h5", key="/str"
    )  # TODO: put this file in a shared folder

    df = split_lst_date(df)

    df = magic_date(df)

    df.to_hdf("observations.h5", key="joint_obs", mode="w")

    source_in = config["data_selection_and_lists"]["source_name_database"]

    source_out = config["data_selection_and_lists"]["source_name_output"]
    range = config["data_selection_and_lists"]["time_range"]
    skip_LST = config["data_selection_and_lists"]["skipped_LST_runs"]
    skip_MAGIC = config["data_selection_and_lists"]["skipped_MAGIC_runs"]

    df = pd.read_hdf("observations.h5", key="joint_obs")
    df = df.astype({"YY_LST": int, "MM_LST": int, "DD_LST": int})

    stereo = True

    df.query(
        f'source=="{source_in}"& MAGIC_trigger=="L3T" & MAGIC_HV=="Nominal" & MAGIC_stereo == {stereo}',
        inplace=True,
    )  #

    if range:
        min = str(config["data_selection_and_lists"]["min"])
        max = str(config["data_selection_and_lists"]["max"])
        min = datetime.strptime(min, "%Y_%m_%d")
        max = datetime.strptime(max, "%Y_%m_%d")
        lst = pd.to_datetime(df['date_LST'].str.replace('_', '-'))
        df["date"] = lst
        df = df[df["date"] > min]
        df = df[df["date"] < max]

    else:
        dates = config["data_selection_and_lists"]["date_list"]

        df = df[df["date_LST"].isin(dates)]

    df = df.reset_index()
    df = df.drop("index", axis=1)

    df.to_hdf("observations_query.h5", key="joint_obs", mode="w")
    list_run(source_out, df, skip_LST, skip_MAGIC)


if __name__ == "__main__":
    main()
