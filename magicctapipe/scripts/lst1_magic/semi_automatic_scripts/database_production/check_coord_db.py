"""
Check if source coordinates stored in the LST database
(from astropy or from a json dictionary) are consistent
with the ones stored in one of the MAGIC files for each
source. Note: only one MAGIC subrun per source is used!

Usage:
$ check_coord_db (-c config_auto_MCP.yaml -m dec_mc.json)
"""

import glob
import json
import math

import numpy as np
import pandas as pd
import uproot
import yaml
from astropy.coordinates import angular_separation

from magicctapipe.io import resource_file
from magicctapipe.utils import auto_MCP_parser, load_merge_databases

degrees_per_hour = 15.0
seconds_per_hour = 3600.0


def main():

    """
    Main function
    """

    parser = auto_MCP_parser()
    parser.add_argument(
        "--mc-dec",
        "-m",
        dest="dec_mc",
        type=str,
        default="./dec_mc.json",
        help="File with list of MC declinations",
    )
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
    lst_df = pd.read_hdf(lst_h5, lst_key)

    df = load_merge_databases(
        config_dict["database_paths"]["MAGIC+LST1"],
        config_dict["database_paths"]["MAGIC+LST1_bis"],
        config_dict["database_keys"]["MAGIC+LST1"],
        config_dict["database_keys"]["MAGIC+LST1_bis"],
    )

    df = df.drop_duplicates(subset=["source"])
    df = df.reset_index()

    date_lst = pd.to_datetime(df["DATE"], format="%Y%m%d")

    delta = pd.Timedelta("1 day")
    date_magic = date_lst + delta

    date_magic = date_magic.dt.strftime("%Y/%m/%d").to_list()

    for i in range(len(df)):
        magic_first_run = (
            (df["MAGIC_runs"].to_list())[i].rstrip("]").lstrip("[").split(", ")[0]
        )
        lst_run = (df["LST1_run"].to_list())[i]
        source = (df["source"].to_list())[i]
        if (
            len(
                glob.glob(
                    f"/fefs/onsite/common/MAGIC/data/M[12]/event/Calibrated/{date_magic[i]}/*{magic_first_run}*{source}*.root"
                )
            )
            == 0
        ):
            print(
                f"No MAGIC file for {source}, {date_magic[i]} (MAGIC convention), run {magic_first_run}\n\n"
            )
            continue
        magic_file = glob.glob(
            f"/fefs/onsite/common/MAGIC/data/M[12]/event/Calibrated/{date_magic[i]}/*{magic_first_run}*{source}*.root"
        )[0]

        if len(lst_df.loc[lst_df["LST1_run"] == lst_run]) == 0:
            print(f"{source}, run {lst_run}, not in LST database\n\n")
            continue
        ra_db = lst_df.loc[lst_df["LST1_run"] == lst_run]["ra"].values[0]
        dec_db = lst_df.loc[lst_df["LST1_run"] == lst_run]["dec"].values[0]

        if (np.isnan(ra_db)) or (np.isnan(dec_db)):
            print(f"RA or Dec is NaN in the database for {source}\n\n")
            continue

        file_tree = uproot.open(magic_file)
        branch_list = [
            "MRawRunHeader.fSourceRA",
            "MRawRunHeader.fSourceDEC",
        ]
        coord = file_tree["RunHeaders"].arrays(branch_list, library="np")
        ra_file = (
            coord["MRawRunHeader.fSourceRA"][0] / seconds_per_hour * degrees_per_hour
        )
        dec_file = coord["MRawRunHeader.fSourceDEC"][0] / seconds_per_hour

        # check RF declination line
        with open(args.dec_mc) as f:
            dec_mc = np.asarray(json.load(f)).astype(np.float64)

        mc_dec_file = float(dec_mc[np.argmin(np.abs(dec_file - dec_mc))])
        mc_dec_db = lst_df.loc[lst_df["LST1_run"] == lst_run]["MC_dec"].values[0]
        if angular_separation(
            lon1=math.radians(ra_file),
            lat1=math.radians(dec_file),
            lon2=math.radians(ra_db),
            lat2=math.radians(dec_db),
        ) < math.radians(0.02):
            continue
        else:
            print(
                f"Coordinates not equal in MAGIC Calibrated files and in LST database for {source}"
            )
            print(
                f"Database RA = {ra_db}, DEC ={dec_db}; MAGIC ROOT files RA = {ra_file}, DEC = {dec_file}"
            )
        if math.isclose(mc_dec_file, mc_dec_db, abs_tol=0.01):
            print("\n\n")
            continue
        else:
            print(
                f"Estimated RF declination line from MAGIC Calibrated files is not the same as the one in LST database for {source}"
            )
            print(
                f"From database MC_dec = {mc_dec_db}; from MAGIC ROOT files MC_dec {mc_dec_file}\n\n"
            )


if __name__ == "__main__":
    main()
