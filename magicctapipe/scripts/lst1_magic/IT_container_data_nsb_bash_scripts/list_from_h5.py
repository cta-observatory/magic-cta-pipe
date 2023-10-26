import os
import pandas as pd
import yaml
from datetime import datetime


def split_lst_date(df):
    date = df["DATE"]

    df["YY_LST"] = date.str[:4]
    df["MM_LST"] = date.str[4:6]
    df["DD_LST"] = date.str[6:8]
    a = f'{df["YY_LST"]}_{df["MM_LST"]}_{df["DD_LST"]}'
    df["date_LST"] = a
    return df


def magic_date(df):
    date_lst = pd.to_datetime(f'{df["YY_LST"]}/{df["MM_LST"]}/{df["DD_LST"]}')

    delta = pd.Timedelta("1 day")

    date_magic = date_lst + delta

    date_magic = date_magic.dt.strftime("%Y_%m_%d")

    df["date_MAGIC"] = date_magic
    return df


def list_run(source_out, df, skip_LST, skip_MAGIC):
    file_list = [
        f"{source_out}_LST_runs.txt",
        f"{source_out}_MAGIC_runs.txt",
    ]  #### LST, MAGIC!!!!
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

        if skip == False:
            with open(file_list[0], "a+") as f:
                f.write(f'{df["date_LST"][k]},{str(LST[k]).lstrip("0")}\n')
            LST_listed.append(int(LST[k]))
        MAGIC_min = int(df["MAGIC_first_run"][k])
        MAGIC_max = int(df["MAGIC_last_run"][k])
        for z in range(MAGIC_min, MAGIC_max + 1):
            skip = False

            if (int(z) in skip_MAGIC) or (int(z) in MAGIC_listed):
                skip = True
            if skip == False:
                with open(file_list[1], "a+") as f:
                    f.write(f'{df["date_MAGIC"][k]},{z}\n')
                MAGIC_listed.append(int(z))


def main():
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

    if range == True:
        min = str(config["data_selection_and_lists"]["min"])
        max = str(config["data_selection_and_lists"]["max"])
        min = datetime.strptime(min, "%Y_%m_%d")
        max = datetime.strptime(max, "%Y_%m_%d")
        lst = pd.to_datetime(f'{df["YY_LST"]}/{df["MM_LST"]}/{df["DD_LST"]}')
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
