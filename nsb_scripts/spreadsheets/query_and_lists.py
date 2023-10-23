"""
Query on the desired parameters and runs
"""
import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime


def list_run(source_out, df, skip_LST, skip_MAGIC):
    file_list = [
        f"{source_out}_LST_runs.txt",
        f"{source_out}_MAGIC_runs.txt",
    ]  #### LST, MAGIC!!!!
    for j in file_list:
        if os.path.isfile(j):
            os.remove(j)
            print(f"{j} deleted.")

    for k in range(len(df)):
        LST = np.fromstring(df["LST_runs"][k][1:-1], dtype="int", sep=", ")

        for j in range(len(LST)):
            skip = False
            for jkz in range(len(skip_LST)):
                if int(LST[j]) == skip_LST[jkz]:
                    skip = True

            if skip == False:
                with open(file_list[0], "a+") as f:
                    f.write(f'{df["date_LST"][k]},{LST[j]}\n')
        MAGIC_min = int(df["first_MAGIC"][k])
        MAGIC_max = int(df["last_MAGIC"][k])
        for z in range(MAGIC_min, MAGIC_max + 1):
            skip = False
            for msp in range(len(skip_MAGIC)):
                if int(z) == skip_MAGIC[msp]:
                    skip = True
            if skip == False:
                with open(file_list[1], "a+") as f:
                    f.write(f'{df["date_MAGIC"][k]},{str(z)}\n')


def main():
    with open(
        "config_google.yaml", "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    source_in = config["data_selection_and_lists"]["source_name_database"]

    source_out = config["data_selection_and_lists"]["source_name_output"]
    range = config["data_selection_and_lists"]["time_range"]
    skip_LST = config["data_selection_and_lists"]["skipped_LST_runs"]
    skip_MAGIC = config["data_selection_and_lists"]["skipped_MAGIC_runs"]

    df = pd.read_hdf("observations.h5", key="joint_obs")
    df = df.astype({"YY_LST": int, "MM_LST": int, "DD_LST": int})

    df["trigger"] = df["trigger"].str.rstrip("']")
    df["trigger"] = df["trigger"].str.lstrip("['")
    df["HV"] = df["HV"].str.rstrip("']")
    df["HV"] = df["HV"].str.lstrip("['")
    df["stereo"] = df["stereo"].str.rstrip("]")
    df["stereo"] = df["stereo"].str.lstrip("[")

    df.query(
        f'source=="{source_in}" & trigger=="L3T" & HV=="Nominal" & stereo == "True"',
        inplace=True,
    )

    if range == True:
        min = str(config["data_selection_and_lists"]["min"])
        max = str(config["data_selection_and_lists"]["max"])
        min = datetime.strptime(min, "%Y_%m_%d")
        max = datetime.strptime(max, "%Y_%m_%d")
        lst = pd.to_datetime(f'{df["YY_LST"].astype(str)}/{df["MM_LST"].astype(str)}/{df["DD_LST"].astype(str)}')
        df["date"] = lst
        df = df[df["date"] > min]
        df = df[df["date"] < max]

    if range == False:
        dates = config["data_selection_and_lists"]["date_list"]

        df = df[df["date_LST"].isin(dates)]

    df = df.reset_index()
    df = df.drop("index", axis=1)

    df.to_hdf("observations_query.h5", key="joint_obs", mode="w")
    list_run(source_out, df, skip_LST, skip_MAGIC)


if __name__ == "__main__":
    main()
