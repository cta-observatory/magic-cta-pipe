"""
This script creates the lists of MAGIC and LST runs (date and run number) from a dataframe in the .h5 format for a specific time range (or specific dates).
"""

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from magicctapipe.io import resource_file
from magicctapipe.utils import auto_MCP_parse_config

__all__ = ["split_lst_date", "magic_date", "clear_files", "list_run"]


def split_lst_date(df):

    """
    This function appends to the provided dataframe, which contains the LST date as YYYYMMDD in one of the columns, four new columns: the LST year, month and day and the date as YYYY-MM-DD

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
    This function appends to the provided dataframe (which contains the LST date, year, month and day) a column with the MAGIC dates (in the YYYYMMDD format).

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        Dataframe of the joint MAGIC+LST-1 observations based on the .h5 table.

    Returns
    -------
    :class:`pandas.DataFrame`
        The input dataframe with an added column.
    """

    date_lst = pd.to_datetime(df["DATE"], format="%Y%m%d")
    delta = pd.Timedelta("1 day")
    date_magic = date_lst + delta
    date_magic = date_magic.dt.strftime("%Y%m%d")
    df["date_MAGIC"] = date_magic
    return df


def clear_files(source_in, source_out, df_LST, df_MAGIC1, df_MAGIC2, allowed_M_tels):

    """
    This function deletes any file named XXXX_LST_runs.txt and XXXX_MAGIC_runs.txt from the working directory.

    Parameters
    ----------
    source_in : str
        Target name in the database. If None, it stands for all the sources observed in a pre-set time interval.
    source_out : str
        Name tag for the target. Used only if source_in is not None.
    df_LST : :class:`pandas.DataFrame`
        LST-1 dataframe of the joint MAGIC+LST-1 observations.
    df_MAGIC1 : :class:`pandas.DataFrame`
        MAGIC-1 dataframe of the joint MAGIC+LST-1 observations.
    df_MAGIC2 : :class:`pandas.DataFrame`
        MAGIC-2 dataframe of the joint MAGIC+LST-1 observations.
    allowed_M_tels : list
        MAGIC telescopes allowed in the analysis.
    """

    source_list = []
    if source_in is None and allowed_M_tels == [1, 2]:
        source_list = np.intersect1d(
            np.intersect1d(np.unique(df_LST["source"]), np.unique(df_MAGIC1["source"])),
            np.unique(df_MAGIC2["source"]),
        )
    elif source_in is None and allowed_M_tels == [1]:
        source_list = np.intersect1d(
            np.unique(df_LST["source"]), np.unique(df_MAGIC1["source"])
        )
    elif source_in is None and allowed_M_tels == [2]:
        source_list = np.intersect1d(
            np.unique(df_LST["source"]), np.unique(df_MAGIC2["source"])
        )
    else:
        source_list.append(source_out)

    joblib.dump(source_list, "list_sources.dat")
    print("Cleaning pre-existing *_LST_runs.txt and *_MAGIC_runs.txt files")
    for source_name in source_list:
        file_list = [
            f"{source_name}_LST_runs.txt",
            f"{source_name}_MAGIC_runs.txt",
        ]  # The order here must be LST before MAGIC!
        for j in file_list:
            if os.path.isfile(j):
                os.remove(j)
                print(f"{j} deleted.")


def list_run(
    source_in,
    source_out,
    df,
    skip_LST,
    skip_MAGIC,
    is_LST,
    allowed_M_tels,
    M1_run_list=None,
):

    """
    This function creates the *_MAGIC_runs.txt and *_LST_runs.txt files, which contain the list of runs (with corresponding dates) to be processed for a given source.

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
        If you are looking for LST runs, set it to True. For MAGIC set False.
    allowed_M_tels : list
        MAGIC telescopes allowed in the analysis.
    M1_run_list : list
        If you are looking for MAGIC runs, pass the list of MAGIC-1 runs here, and the MAGIC-2 database as df.
        If the analysis concerns both MAGIC, only the runs both in the list and in the data frame
        (i.e., stereo MAGIC observations) will be saved in the output txt files.
        If mono MAGIC data are to be processed, they should be provided as a df, and M1_run_list will then be ignored.
    """

    source_list = []
    if source_in is None:
        source_list = np.unique(df["source"])

    else:
        source_list.append(source_out)

    for source_name in source_list:

        file_list = [
            f"{source_name}_LST_runs.txt",
            f"{source_name}_MAGIC_runs.txt",
        ]  # The order here must be LST before MAGIC!

        run_listed = []
        if source_in is None:
            df_source = df[df["source"] == source_name]
            print("Source: ", source_name)
        else:
            df_source = df[df["source"] == source_in]
            print("Source: ", source_in)

        if is_LST:
            print("Finding LST runs...")
            if len(df_source) == 0:
                print("NO LST run found. Exiting...")
                continue
            LST_run = df_source["LST1_run"].tolist()  # List with runs as strings
            LST_date = df_source["date_LST"].tolist()
            for k in range(len(df_source)):
                if np.isnan(LST_run[k]):
                    continue

                if (int(LST_run[k]) in skip_LST) or (int(LST_run[k]) in run_listed):
                    continue

                with open(file_list[0], "a+") as f:
                    f.write(
                        f"{LST_date[k].replace('-','_')},{str(LST_run[k]).lstrip('0')}\n"
                    )
                run_listed.append(int(LST_run[k]))

        if not is_LST:
            print("Finding MAGIC runs...")
            if len(df_source) == 0:
                print("NO MAGIC run found. Exiting...")
                continue
            MAGIC_date = df_source["date_MAGIC"].tolist()
            M2_run = df_source["Run ID"].tolist()
            for k in range(len(df_source)):
                if np.isnan(M2_run[k]):
                    continue

                if (int(M2_run[k]) in skip_MAGIC) or (int(M2_run[k]) in run_listed):
                    continue
                if len(allowed_M_tels) == 2 and int(M2_run[k]) not in M1_run_list:
                    continue

                with open(file_list[1], "a+") as f:
                    f.write(
                        f"{MAGIC_date[k][0:4]}_{MAGIC_date[k][4:6]}_{MAGIC_date[k][6:8]},{int(M2_run[k])}\n"
                    )
                run_listed.append(int(M2_run[k]))


def main():

    """
    Main function
    """

    config = auto_MCP_parse_config()
    config_db = config["general"]["base_db_config_file"]
    if config_db == "":
        config_db = resource_file("database_config.yaml")

    with open(
        config_db, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)

    LST_h5 = config_dict["database_paths"]["LST"]
    LST_key = config_dict["database_keys"]["LST"]
    MAGIC_h5 = config_dict["database_paths"]["MAGIC"]
    MAGIC1_key = config_dict["database_keys"]["MAGIC-I"]
    MAGIC2_key = config_dict["database_keys"]["MAGIC-II"]
    source_in = config["data_selection"]["source_name_database"]
    source_out = config["data_selection"]["source_name_output"]
    allowed_M_tels = sorted(config["general"]["allowed_M_tels"])

    if (source_out is None) and (source_in is not None):
        source_out = source_in
    range = config["data_selection"]["time_range"]
    skip_LST = config["data_selection"]["skip_LST_runs"]
    skip_MAGIC = config["data_selection"]["skip_MAGIC_runs"]

    df_LST = pd.read_hdf(
        LST_h5,
        key=LST_key,
    )  # TODO: put this file in a shared folder
    df_LST.dropna(subset=["LST1_run"], inplace=True)
    df_LST = split_lst_date(df_LST)
    df_LST = df_LST.astype(
        {"YY_LST": int, "MM_LST": int, "DD_LST": int, "nsb": float, "LST1_run": int}
    )

    lstchain_version = config["general"]["LST_version"]

    processed_v = df_LST["processed_lstchain_file"].str.split("/").str[-3]

    mask = processed_v == lstchain_version
    df_LST = df_LST[mask]

    if source_in is None:
        if len(allowed_M_tels) == 2:
            df_LST.query(
                'MAGIC_trigger=="L3T" & MAGIC_HV=="Nominal" & (MAGIC_stereo == True | MAGIC_stereo == "True") & perfect_match_time_min > 0.1 & error_code_nsb=="0"',
                inplace=True,
            )
        elif len(allowed_M_tels) == 1:
            df_LST.query(
                f'MAGIC_trigger=="L1_M{allowed_M_tels[0]}" & MAGIC_HV=="Nominal" & (MAGIC_stereo == False | MAGIC_stereo == "False") & perfect_match_time_min > 0.1 & error_code_nsb=="0"',
                inplace=True,
            )
    else:
        if len(allowed_M_tels) == 2:
            df_LST.query(
                f'source=="{source_in}" & MAGIC_trigger=="L3T" & MAGIC_HV=="Nominal" & (MAGIC_stereo == True | MAGIC_stereo == "True") & perfect_match_time_min > 0.1 & error_code_nsb=="0"',
                inplace=True,
            )
        elif len(allowed_M_tels) == 1:
            df_LST.query(
                f'source=="{source_in}" & MAGIC_trigger=="L1_M{allowed_M_tels[0]}" & MAGIC_HV=="Nominal" & (MAGIC_stereo == False | MAGIC_stereo == "False") & perfect_match_time_min > 0.1 & error_code_nsb=="0"',
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
    df_MAGIC1 = pd.read_hdf(
        MAGIC_h5,
        key=MAGIC1_key,
    )
    df_MAGIC2 = pd.read_hdf(
        MAGIC_h5,
        key=MAGIC2_key,
    )

    list_date_LST = np.unique(df_LST["date_LST"])
    list_date_LST_low = [int(sub.replace("-", "")) for sub in list_date_LST]

    df_MAGIC1 = df_MAGIC1[df_MAGIC1["DATE"].isin(list_date_LST_low)]
    df_MAGIC2 = df_MAGIC2[df_MAGIC2["DATE"].isin(list_date_LST_low)]

    clear_files(source_in, source_out, df_LST, df_MAGIC1, df_MAGIC2, allowed_M_tels)

    list_run(source_in, source_out, df_LST, skip_LST, skip_MAGIC, True, allowed_M_tels)

    df_MAGIC2 = magic_date(df_MAGIC2)
    df_MAGIC1 = magic_date(df_MAGIC1)

    M1_runs = df_MAGIC1["Run ID"].tolist()
    if len(allowed_M_tels) == 2 and (len(M1_runs) == 0) or (len(df_MAGIC2) == 0):
        print("NO MAGIC stereo run found. Exiting...")
        return

    df_MAGIC = df_MAGIC2 if 2 in allowed_M_tels else df_MAGIC1

    list_run(
        source_in,
        source_out,
        df_MAGIC,
        skip_LST,
        skip_MAGIC,
        False,
        allowed_M_tels,
        M1_runs,
    )


if __name__ == "__main__":
    main()
