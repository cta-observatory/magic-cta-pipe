"""
Create a new h5 table from the one of joint observations.

Only the columns needed to produce the lists of LST runs to be processed are presenved, and two columns are added to store NSB level and error codes
"""

import os

import numpy as np
import pandas as pd


def main():

    """
    Main function
    """

    df = pd.read_hdf(
        "/fefs/aswg/workspace/federico.dipierro/simultaneous_obs_summary.h5", key="/str"
    )  # TODO: put this file in a shared folder
    df2 = pd.read_hdf(
        "/home/alessio.berti/MAGIC-LST_common/runfile/simultaneous_obs_summary.h5",
        key="/str",
    )  # TODO: put this file in a shared folder
    df = pd.concat([df, df2]).drop_duplicates(subset="LST1_run", keep="first")
    needed_cols = [
        "source",
        "DATE",
        "LST1_run",
        "MAGIC_stereo",
        "MAGIC_trigger",
        "MAGIC_HV",
    ]
    df_cut = df[needed_cols]

    df_cut = df_cut.assign(nsb=np.nan)
    df_cut = df_cut.assign(lstchain_versions="[]")
    df_cut = df_cut.assign(last_lstchain_file="")
    df_cut = df_cut.assign(processed_lstchain_file="")
    df_cut = df_cut.assign(error_code_nsb=-1)

    df_cut = df_cut.assign(error_code_coincidence=-1)
    df_cut = df_cut.assign(error_code_stereo=-1)

    if os.path.isfile(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5"
    ):
        df_old = pd.read_hdf(
            "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5",
            key="joint_obs",
        )
        df_cut = pd.concat([df_old, df_cut]).drop_duplicates(
            subset="LST1_run", keep="first"
        )
        df_cut = df_cut.sort_values(by=["DATE", "source"])
        # TODO check if fine with update and nsb
    df_cut = df_cut.reset_index(drop=True)
    df_cut.to_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5",
        key="joint_obs",
        mode="w",
        min_itemsize={
            "lstchain_versions": 20,
            "last_lstchain_file": 90,
            "processed_lstchain_file": 90,
        },
    )


if __name__ == "__main__":
    main()
