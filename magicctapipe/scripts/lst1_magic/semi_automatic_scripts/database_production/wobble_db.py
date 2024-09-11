"""
Add wobble info to the LST database (by checking MAGIC runs).

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

    args = parser.parse_args()
    config_file = resource_file("database_config.yaml")

    with open(
        config_file, "rb"
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
    df2 = pd.read_hdf(
        config_dict["database_paths"]["MAGIC+LST1_bis"],
        key=config_dict["database_keys"]["MAGIC+LST1_bis"],
    )  # TODO: put this file in a shared folder
    df = pd.concat([df, df2]).drop_duplicates(subset="LST1_run", keep="first")
    if args.begin != 0:
        df = df[df["DATE"].astype(int) >= args.begin]
    if args.end != 0:
        df = df[df["DATE"].astype(int) <= args.end]
    if "wobble" not in df_LST:
        df_LST["wobble"] = np.nan

    date_lst = pd.to_datetime(df["DATE"], format="%Y%m%d")

    delta = pd.Timedelta("1 day")
    date_magic = date_lst + delta

    date_magic = date_magic.dt.strftime("%Y/%m/%d").to_list()
    for i in range(len(df)):
        magic_runs = (df["MAGIC_runs"].to_list())[i].rstrip("]").lstrip("[").split(",")
        lst_run = (df["LST1_run"].to_list())[i]
        wobble = []
        source = (df["source"].to_list())[i]
        for j in range(len(magic_runs)):
            print("MAGIC run:", magic_runs[j])
            runs = glob.glob(
                f"/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/{date_magic[i]}/*{magic_runs[j]}*{source}*.root"
            )

            if len(runs) < 1:
                runs = glob.glob(
                    f"/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/{date_magic[i]}/*{magic_runs[j]}*{source}*.root"
                )
            if len(runs) < 1:
                print(
                    f"Neither M1 nor M2 files could be found for {date_magic[i]}, run {magic_runs[j]}, {source}. Check database and stored data!"
                )
                continue
            wobble_run = runs[0].split("/")[-1].split(source)[1][2:6]
            print("Wobble:", wobble_run)
            wobble.append(wobble_run)

        wobble_str = "[" + "".join(str(x) for x in wobble) + "]"
        print(f"Wobble for LST run {lst_run}:", wobble_str)
        df_LST["wobble"] = np.where(
            df_LST["LST1_run"] == lst_run, wobble_str, df_LST["wobble"]
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
