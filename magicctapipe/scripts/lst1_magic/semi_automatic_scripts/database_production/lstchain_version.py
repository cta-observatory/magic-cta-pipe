"""
Fill the lstchain_0.9 and lstchain_0.10 columns of the LST database (i.e., which version of data is on the IT cluster)
"""


import glob
import os

import pandas as pd
from magicctapipe.io import resource_file

lstchain_versions = ["v0.9", "v0.10"]
__all__ = ["version_lstchain"]


def version_lstchain(df_LST):
    """
    Evaluates all the versions used to process a given file and the last version of a file

    Parameters
    ----------
    df_LST : :class:`pandas.DataFrame`
        Dataframe of the LST-1 observations.
    """
    for i, row in df_LST.iterrows():

        version = []
        run = row["LST1_run"]
        run = format(int(run), "05d")
        date = row["DATE"]
        directories_version = [
            i.split("/")[-1] for i in glob.glob(f"/fefs/aswg/data/real/DL1/{date}/v*")
        ]

        for vers in directories_version:

            if os.path.isfile(
                f"/fefs/aswg/data/real/DL1/{date}/{vers}/tailcut84/dl1_LST-1.Run{run}.h5"
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
            raise ValueError("issue with lstchain versions")
        name = f"/fefs/aswg/data/real/DL1/{date}/{max_version}/tailcut84/dl1_LST-1.Run{run}.h5"

        df_LST.loc[i, "last_lstchain_file"] = name


def main():

    """
    Main function
    """
    config_file = resource_file("database_config.yaml")

    with open(
        config_file, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)

    LST_h5=config_dict['database_paths']['LST']
    LST_key=config_dict['database_keys']['LST']
    df_LST = pd.read_hdf(
        LST_h5,
        key=LST_key
    )

    version_lstchain(df_LST)

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
