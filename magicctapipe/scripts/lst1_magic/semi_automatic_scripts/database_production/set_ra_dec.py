"""
Add, to LST database, infos about coordinates and extension of the sources and MC declination to be used to process the source

Usage:
$ set_ra_dec (-b YYYYMMDD -e YYYYMMDD -s source_dict -m mc_dec)
"""

import argparse
import json

import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord
from astropy.coordinates.name_resolve import NameResolveError

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

    LST_h5 = config_dict["database_paths"]["LST"]
    LST_key = config_dict["database_keys"]["LST"]

    df_LST = pd.read_hdf(
        LST_h5,
        key=LST_key,
    )
    if "ra" not in df_LST:
        df_LST["ra"] = np.nan
    if "dec" not in df_LST:
        df_LST["dec"] = np.nan
    if "MC_dec" not in df_LST:
        df_LST["MC_dec"] = np.nan
    if "point_source" not in df_LST:
        df_LST["point_source"] = np.nan
    df_LST_full = df_LST.copy(deep=True)
    if args.begin != 0:
        df_LST = df_LST[df_LST["DATE"].astype(int) >= args.begin]
    if args.end != 0:
        df_LST = df_LST[df_LST["DATE"].astype(int) <= args.end]

    sources = np.unique(df_LST["source"])
    with open(args.source) as f:
        dict_source = f.read()
    with open(args.dec_mc) as f:
        mc_dec = f.read()
    source_dict = json.loads(dict_source)
    dec_mc = np.asarray(json.loads(mc_dec)).astype(np.float64)
    print("MC declinations: \t", dec_mc)
    print("\n\nChecking RA/Dec...\n\n")
    i = 0
    for src in sources:

        try:
            coord = SkyCoord.from_name(src)
            if src == "Crab":
                coord = SkyCoord.from_name("CrabNebula")
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
    print("\n\nChecking if point source...\n\n")
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
    df_LST = pd.concat([df_LST, df_LST_full]).drop_duplicates(
        subset="LST1_run", keep="first"
    )
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
