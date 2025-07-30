"""
Add, to LST database, infos about coordinates, wobble offset (by checking MAGIC runs) and extension of the sources and MC declination to be used to process the source

Usage:
$ set_coord_wobble (-b YYYYMMDD -e YYYYMMDD -s source_dict -m mc_dec -c config_file)
"""

import glob
import json

import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

from magicctapipe.io import resource_file
from magicctapipe.utils import auto_MCP_parser


def main():

    """
    Main function
    """

    parser = auto_MCP_parser(add_dates=True)
    parser.add_argument(
        "--dict-source",
        "-s",
        dest="source",
        type=str,
        default="./source_dict.json",
        help="File with dictionary of info (RA/Dec/point-source) for sources",
    )
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
        config = yaml.safe_load(f)
    config_db = config["general"]["base_db_config_file"]
    if config_db == "":
        config_db = resource_file("database_config.yaml")

    with open(
        config_db, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)

    LST_h5 = config_dict["database_paths"]["LST"]
    LST_key = config_dict["database_keys"]["LST"]

    df_LST = pd.read_hdf(
        LST_h5,
        key=LST_key,
    )
    for field in ["ra", "dec", "MC_dec", "point_source", "wobble_offset"]:
        if field not in df_LST:
            df_LST[field] = np.nan

    df_LST_full = df_LST.copy(deep=True)
    if args.begin != 0:
        df_LST = df_LST[df_LST["DATE"].astype(int) >= args.begin]
    if args.end != 0:
        df_LST = df_LST[df_LST["DATE"].astype(int) <= args.end]

    sources = np.unique(df_LST["source"])
    with open(args.source) as f:
        source_dict = json.load(f)
    with open(args.dec_mc) as f:
        dec_mc = np.asarray(json.load(f)).astype(np.float64)
    print("MC declinations: \t", dec_mc)
    print("\n\nChecking RA/Dec...\n\n")
    i = 0
    for src in sources:

        try:
            coord = SkyCoord.from_name(src)
            if src == "Crab":
                coord = SkyCoord.from_name("CrabNebula")
                # astropy retrieves two slightly different coordinates, in SkyCoord, for 'Crab' and 'CrabNebula,
                # but these two lables correspond to the same pointings for MAGIC and LST
            src_dec = coord.dec.degree
            src_ra = coord.ra.degree

        except NameResolveError:
            print(f"{i}: {src} not found in astropy. Looking to the dictionaries...")
            if (
                (src in source_dict.keys())
                and (source_dict.get(src)[0] != "NaN")
                and (source_dict.get(src)[1] != "NaN")
            ):
                src_ra = float(source_dict.get(src)[0])
                src_dec = float(source_dict.get(src)[1])

            else:
                print(
                    f"\t {i}: {src} RA and/or Dec not in the dictionary. Please update the dictionary"
                )
                src_ra = np.nan
                src_dec = np.nan

            i += 1
        df_LST["ra"] = np.where(df_LST["source"] == src, src_ra, df_LST["ra"])
        df_LST["dec"] = np.where(df_LST["source"] == src, src_dec, df_LST["dec"])
        if not (np.isnan(src_dec)):
            df_LST["MC_dec"] = np.where(
                df_LST["source"] == src,
                float(dec_mc[np.argmin(np.abs(src_dec - dec_mc))]),
                df_LST["MC_dec"],
            )
    print("\n\nChecking if point-like source...\n\n")
    i = 0
    for src in sources:
        if (src in source_dict.keys()) and (source_dict.get(src)[2] != "NaN"):
            src_point = str(source_dict.get(src)[2])
            df_LST["point_source"] = np.where(
                df_LST["source"] == src, src_point, df_LST["point_source"]
            )
        else:
            print(
                f"\t {i}: {src} extension information not in the dictionaries. Please add it to the dictionaries"
            )
            i += 1
    print("\n\nRetrieving wobble offset...\n\n")
    df = pd.read_hdf(
        config_dict["database_paths"]["MAGIC+LST1"],
        key=config_dict["database_keys"]["MAGIC+LST1"],
    )  # TODO: put this file in a shared folder
    if args.begin != 0:
        df = df[df["DATE"].astype(int) >= args.begin]
    if args.end != 0:
        df = df[df["DATE"].astype(int) <= args.end]
    date_lst = pd.to_datetime(df["DATE"], format="%Y%m%d")

    delta = pd.Timedelta("1 day")
    date_magic = date_lst + delta

    date_magic = date_magic.dt.strftime("%Y/%m/%d").to_list()

    for i in range(len(df)):
        magic_runs = (df["MAGIC_runs"].to_list())[i].rstrip("]").lstrip("[").split(", ")
        lst_run = (df["LST1_run"].to_list())[i]
        wobble = []
        source = (df["source"].to_list())[i]
        for j in range(len(magic_runs)):
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
                wobble_run = (wobble_run_info.split("W")[-1])[0:4]
            else:
                print(
                    f"No string matching for wobble offset found in the name of MAGIC files for {date_magic[i]}, run {magic_runs[j]}, {source}. Check it manually!"
                )
                continue
            wobble.append(wobble_run)
        wobble = np.unique(wobble)
        if len(wobble) > 1:
            print(
                f"More than one wobble offset value for LST run {lst_run}: check data!"
            )
        wobble_str = str(wobble).replace(" ", ", ")
        df_LST["wobble_offset"] = np.where(
            df_LST["LST1_run"] == lst_run, wobble_str, df_LST["wobble_offset"]
        )

    df_LST = pd.concat([df_LST, df_LST_full]).drop_duplicates(
        subset="LST1_run", keep="first"
    )
    df_LST = df_LST.sort_values(by=["DATE", "source", "LST1_run"])
    df_LST = df_LST[df_LST["source"].notna()]
    df_LST.to_hdf(
        LST_h5,
        key=LST_key,
        mode="w",
        min_itemsize={
            "lstchain_versions": 20,
            "last_lstchain_file": 90,
            "processed_lstchain_file": 90,
        },
    )


if __name__ == "__main__":
    main()
