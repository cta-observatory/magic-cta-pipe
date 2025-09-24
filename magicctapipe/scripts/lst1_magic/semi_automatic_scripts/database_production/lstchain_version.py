"""
Fills the lstchain_versions column of the LST database with a list of the versions of LST data which are stored on the IT cluster

Moreover, it fills the last_lstchain_files column of the LST database with the path to the LST DL1 file processed with the last available lstchain version (the name and order of the versions to be considered is stored in the lstchain_versions variable here defined)

Usage:
$ lstchain_version
"""

import glob
import os

import numpy as np
import pandas as pd
import yaml

from magicctapipe.io import resource_file
from magicctapipe.utils import auto_MCP_parser

__all__ = ["version_lstchain"]


def version_lstchain(df_LST, lstchain_versions):
    """
    Evaluates (and store in the database) all the versions used to process a given file and the last version of a file

    Parameters
    ----------
    df_LST : :class:`pandas.DataFrame`
        Dataframe of the LST-1 observations.
    lstchain_versions : list
        List of the available lstchain varsions that can be processed by MCP (from older to newer)
    """

    for i, row in df_LST.iterrows():

        version = []
        run = row["LST1_run"]
        run = format(int(run), "05d")
        date = row["DATE"]
        for base_path in [
            "/fefs/aswg/data/real/DL1",
            "/fefs/onsite/data/lst-pipe/LSTN-01/DL1",
        ]:
            if not os.path.isdir(f"{base_path}/{date}"):
                continue
            directories_version = [
                i.split("/")[-1] for i in glob.glob(f"{base_path}/{date}/v*")
            ]
            tailcut_list = []

            for vers in directories_version:

                tailcut_list = [
                    i.split("/")[-1]
                    for i in glob.glob(f"{base_path}/{date}/{vers}/tailcut*")
                ]
                for tail in tailcut_list:
                    if os.path.isfile(
                        f"{base_path}/{date}/{vers}/{tail}/dl1_LST-1.Run{run}.h5"
                    ):
                        if vers not in version:
                            version.append(vers)

        version = list(version)
        df_LST.loc[i, "lstchain_versions"] = str(version)
        max_version = None

        for j in range(len(lstchain_versions)):

            if lstchain_versions[j] in version:

                max_version = lstchain_versions[j]

        if max_version is None:
            print(
                f"issue with lstchain versions for run {run}\nAvailable versions: {version}, allowed versions: {lstchain_versions}\n\n\n"
            )
            continue
        tail_file = []
        for base_path in [
            "/fefs/aswg/data/real/DL1",
            "/fefs/onsite/data/lst-pipe/LSTN-01/DL1",
        ]:
            if not os.path.isdir(f"{base_path}/{date}/{max_version}"):
                continue
            tail_file = []
            tailcut_list = [
                i.split("/")[-1]
                for i in glob.glob(f"{base_path}/{date}/{max_version}/tailcut*")
            ]

            for tail in tailcut_list:
                if os.path.isfile(
                    f"{base_path}/{date}/{max_version}/{tail}/dl1_LST-1.Run{run}.h5"
                ):
                    tail_file.append(tail)
                    name = (
                        f"{base_path}/{date}/{max_version}/{tail}/dl1_LST-1.Run{run}.h5"
                    )
        if (
            len(np.unique(tail_file)) > 1
        ):  # WARNING: if same date and version exist in both base dirs, only newest one considered
            print(
                f"More than one tailcut for the latest ({max_version}) lstchain version for run {run}. Tailcut = {tail_file}. Skipping..."
            )
            continue

        df_LST.loc[i, "last_lstchain_file"] = name


def main():

    """
    Main function
    """
    parser = auto_MCP_parser(add_dates=True)
    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)

    lstchain_versions = config["expert_parameters"]["lstchain_versions"]
    config_db = config["general"]["base_db_config_file"]
    if config_db == "":
        config_db = resource_file("database_config.yaml")

    with open(
        config_db, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)

    LST_h5 = config_dict["database_paths"]["LST"]
    LST_key = config_dict["database_keys"]["LST"]
    df_LST = pd.read_hdf(LST_h5, key=LST_key)

    version_lstchain(df_LST, lstchain_versions)
    df_LST = df_LST.sort_values(by=["DATE", "source", "LST1_run"])
    df_LST = df_LST[df_LST["source"].notna()]
    df_LST.to_hdf(
        LST_h5,
        key=LST_key,
        mode="w",
        min_itemsize={
            "lstchain_versions": 20,
            "last_lstchain_file": 100,
            "processed_lstchain_file": 100,
        },
    )


if __name__ == "__main__":
    main()
