"""
Add wobble offset info to the LST database (by checking MAGIC runs).

Usage:
$ wobble_db (-b YYYYMMDD -e YYYYMMDD)
"""

import argparse
import glob

import numpy as np
import pandas as pd
import yaml

from magicctapipe.io import resource_file


def main():

    """
    Main function
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--begin-date",
        "-b",
        dest="begin",
        type=int,
        default=0,
        help="First date to update database (YYYYMMDD)",
    )
    parser.add_argument(
        "--end-date",
        "-e",
        dest="end",
        type=int,
        default=0,
        help="End date to update database (YYYYMMDD)",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_auto_MCP.yaml",
        help="Path to a configuration file",
    )

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    config_db = config["general"]["base_db_config_file"]
    if config_db == "":
        config_db = resource_file("database_config.yaml")

    with open(
        config_db, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)

    out_h5 = config_dict["database_paths"]["LST"]
    out_key = config_dict["database_keys"]["LST"]

    df_LST = pd.read_hdf(
        out_h5,
        key=out_key,
    )
    df = pd.read_hdf(
        config_dict["database_paths"]["MAGIC+LST1"],
        key=config_dict["database_keys"]["MAGIC+LST1"],
    )  # TODO: put this file in a shared folder
    """
    df2 = pd.read_hdf(
        config_dict["database_paths"]["MAGIC+LST1_bis"],
        key=config_dict["database_keys"]["MAGIC+LST1_bis"],
    )  # TODO: put this file in a shared folder
    df = pd.concat([df, df2]).drop_duplicates(subset="LST1_run", keep="first")
    df = df.sort_values(by=["DATE", "source"])

    df = df.reset_index(drop=True)
    """
    if args.begin != 0:
        df = df[df["DATE"].astype(int) >= args.begin]
    if args.end != 0:
        df = df[df["DATE"].astype(int) <= args.end]
    if "wobble_offset" not in df_LST:
        df_LST["wobble_offset"] = np.nan

    date_lst = pd.to_datetime(df["DATE"], format="%Y%m%d")

    delta = pd.Timedelta("1 day")
    date_magic = date_lst + delta

    date_magic = date_magic.dt.strftime("%Y/%m/%d").to_list()
    for i in range(len(df)):
        magic_runs = (
            (df["MAGIC_runs"].to_list())[i]
            .rstrip("]")
            .lstrip("[")
            .replace(" ", "")
            .split(",")
        )
        lst_run = (df["LST1_run"].to_list())[i]
        wobble = []
        source = (df["source"].to_list())[i]
        for j in range(len(magic_runs)):
            print("MAGIC run:", magic_runs[j])
            runs = glob.glob(
                f"/fefs/onsite/common/MAGIC/data/M[12]/event/Calibrated/{date_magic[i]}/*{magic_runs[j]}*{source}*.root"
            )

            if len(runs) < 1:
                print(
                    f"Neither M1 nor M2 files could be found for {date_magic[i]}, run {magic_runs[j]}, {source}. Check database and stored data!"
                )
                continue
            wobble_run_info = runs[0].split("/")[-1].split(source)[1]
            if "-W" in wobble_run_info:
                wobble_run = (wobble_run_info.split("W")[1])[0:4]
            else:
                print(
                    f"No string matching for wobble offset found in the name of MAGIC files for {date_magic[i]}, run {magic_runs[j]}, {source}. Check it manually!"
                )
                continue
            print("wobble offset:", wobble_run)
            wobble.append(wobble_run)
        wobble = np.unique(wobble)
        if len(wobble) > 1:
            print(
                f"More than one wobble offset value for LST run {lst_run}: check data!"
            )
        wobble_str = "[" + ", ".join(str(x) for x in wobble) + "]"
        print(f"Wobble offset for LST run {lst_run}:", wobble_str)
        df_LST["wobble_offset"] = np.where(
            df_LST["LST1_run"] == lst_run, wobble_str, df_LST["wobble_offset"]
        )
    df_LST.to_hdf(
        out_h5,
        key=out_key,
        mode="w",
        min_itemsize={
            "lstchain_versions": 20,
            "last_lstchain_file": 90,
            "processed_lstchain_file": 90,
        },
    )


if __name__ == "__main__":
    main()
