"""
Google Sheets to pandas dataframe
"""

import json

import gspread
import pandas as pd
import yaml
from google.oauth2 import service_account


def load_to_pandas(config):
    json_key = config["google"]["json_key"]
    url = config["google"]["worksheet_url"]
    sheet = config["google"]["sheet"]
    col1 = config["google"]["first_column"]
    col2 = config["google"]["last_column"]
    row1 = config["google"]["min_read_line"]
    row2 = config["google"]["max_read_line"]

    key = open(json_key)
    json_load = json.load(key)
    credentials = service_account.Credentials.from_service_account_info(json_load)
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    auth = credentials.with_scopes(scopes)
    gspread_auth = gspread.authorize(auth)
    google_sheet = gspread_auth.open_by_url(url)
    table = google_sheet.get_worksheet(sheet)
    data = table.get(f"{col1}{row1}:{col2}{row2}")
    df = pd.DataFrame.from_dict(data)
    return df


def split_lst_date(df):
    date = df["date_LST"]
    y = date.str.split("_", expand=True)[0]
    m = date.str.split("_", expand=True)[1]
    d = date.str.split("_", expand=True)[2]

    df["YY_LST"] = y
    df["MM_LST"] = m
    df["DD_LST"] = d
    return df


def magic_date(df):
    date_lst = pd.to_datetime(f'{df["YY_LST"]}/{df["MM_LST"]}/{df["DD_LST"]}')

    delta = pd.Timedelta("1 day")

    date_magic = date_lst + delta

    date_magic = date_magic.dt.strftime("%Y_%m_%d")

    df["date_MAGIC"] = date_magic
    return df


def main():
    with open(
        "config_google.yaml", "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    df = load_to_pandas(config)
    df.columns = [
        "date_LST",
        "source",
        "moon",
        "trigger",
        "stereo",
        "HV",
        "min_Zd",
        "max_Zd",
        "LST_runs",
        "first_MAGIC",
        "last_MAGIC",
    ]

    df = split_lst_date(df)

    df = magic_date(df)

    df.to_hdf("observations.h5", key="joint_obs", mode="w")


if __name__ == "__main__":
    main()
