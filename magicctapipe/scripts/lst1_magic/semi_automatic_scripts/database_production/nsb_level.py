"""
Creates bash scripts to run LSTnsb.py on all the LST runs, in the provided time range (-b, -e), by using parallel jobs. It sets error_code_nsb = NaN for these runs

Moreover, it can modify the lstchain standard configuration file (used to evaluate NSB) by adding "use_flatfield_heuristic" = True

Usage:
$ nsb_level (-c config.yaml -b YYYY_MM_DD -e YYYY_MM_DD)
"""

import glob
import json
import logging
import os

import numpy as np
import pandas as pd
import yaml

from magicctapipe.io import resource_file
from magicctapipe.scripts.lst1_magic.semi_automatic_scripts.clusters import slurm_lines
from magicctapipe.utils import auto_MCP_parser

__all__ = ["bash_scripts"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def bash_scripts(run, date, config, env_name, cluster, lst_config, inputdir):

    """Here we create the bash scripts (one per LST run)

    Parameters
    ----------
    run : str
        LST run number
    date : str
        LST date
    config : str
        Name of the configuration file
    env_name : str
        Name of the environment
    cluster : str
        Cluster system
    lst_config : str
        Configuration file lstchain
    inputdir : str
        Path where LST DL1 file is located
    """
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return
    slurm = slurm_lines(
        queue="long",
        job_name="nsb",
        out_name=f"slurm-nsb_{run}-%x.%j",
    )
    lines = slurm + [
        f"conda run -n  {env_name} LSTnsb -c {config} -i {run} -d {date} -l {lst_config} -p {inputdir} > nsblog_{date}_{run}_",
        "${SLURM_JOB_ID}.log 2>&1 \n\n",
    ]

    with open(f"nsb_{date}_run_{run}.sh", "w") as f:
        f.writelines(lines)


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
    config_db = config["general"]["base_db_config_file"]
    if config_db == "":
        config_db = resource_file("database_config.yaml")

    with open(
        config_db, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)

    LST_h5 = config_dict["database_paths"]["LST"]
    LST_key = config_dict["database_keys"]["LST"]
    env_name = config["general"]["env_name"]

    cluster = config["general"]["cluster"]
    lstchain_versions = config["expert_parameters"]["lstchain_versions"]
    df_LST = pd.read_hdf(
        LST_h5,
        key=LST_key,
    )
    lstchain_v = config["general"]["LST_version"]
    lstchain_modified = config["expert_parameters"]["lstchain_modified_config"]
    conda_path = os.environ["CONDA_PREFIX"]
    lst_config_orig = (
        str(conda_path)
        + "/lib/python3.11/site-packages/lstchain/data/lstchain_standard_config.json"
    )
    with open(lst_config_orig, "r") as f_lst:
        lst_dict = json.load(f_lst)
    if lstchain_modified:
        lst_dict["source_config"]["LSTEventSource"]["use_flatfield_heuristic"] = True
    with open("lstchain.json", "w+") as outfile:
        json.dump(lst_dict, outfile)
    lst_config = "lstchain.json"

    if args.begin != 0:
        df_LST = df_LST[df_LST["DATE"].astype(int) >= args.begin]
    if args.end != 0:
        df_LST = df_LST[df_LST["DATE"].astype(int) <= args.end]

    print("***** Generating bashscripts...")
    for i, row in df_LST.iterrows():

        list_v = [eval(i) for i in row["lstchain_versions"].strip("][").split(", ")]

        if str(lstchain_v) not in list_v:
            continue

        common_v = [value for value in lstchain_versions if value in list_v]

        max_common = common_v[-1]

        if lstchain_v != str(max_common):
            continue

        run_number = row["LST1_run"]
        date = row["DATE"]
        tailcut = []
        for base_path in [
            "/fefs/aswg/data/real/DL1",
            "/fefs/onsite/data/lst-pipe/LSTN-01/DL1",
        ]:
            if not os.path.isdir(f"{base_path}/{date}/{max_common}"):
                continue
            tailcut = []
            tailcut_list = [
                i.split("/")[-1]
                for i in glob.glob(f"{base_path}/{date}/{max_common}/tailcut*")
            ]

            for tail in tailcut_list:
                if os.path.isfile(
                    f"{base_path}/{date}/{max_common}/{tail}/dl1_LST-1.Run{run_number}.h5"
                ):
                    tailcut.append(tail)
                    name = f"{base_path}/{date}/{max_common}/{tail}/dl1_LST-1.Run{run_number}.h5"
        if len(np.unique(tailcut)) > 1:  # WARNING: if same date and version exist in both base dirs, only newest one considered
            print(
                f"more than one tailcut for the latest ({max_common}) lstchain version for run {run_number}. Tailcut = {tailcut}. Skipping..."
            )
            continue

        df_LST.loc[i, "processed_lstchain_file"] = name

        inputdir = os.path.dirname(name)
        if len(tailcut) >0:
            df_LST.loc[i, "tailcut"] = str(tailcut[0])
        else:
            df_LST.loc[i, "tailcut"] = ""
        df_LST.loc[i, "error_code_nsb"] = np.nan

        bash_scripts(
            run_number, date, args.config_file, env_name, cluster, lst_config, inputdir
        )

    print("Process name: nsb")
    print("To check the jobs submitted to the cluster, type: squeue -n nsb")
    list_of_bash_scripts = np.sort(glob.glob("nsb_*_run_*.sh"))

    if len(list_of_bash_scripts) == 0:
        logger.warning(
            "No bash script has been produced to evaluate the NSB level for the provided LST runs. Please check the input dates"
        )
        return
    print("Update database and launch jobs")
    df_old = pd.read_hdf(
        LST_h5,
        key=LST_key,
    )
    df_LST = pd.concat([df_LST, df_old]).drop_duplicates(
        subset="LST1_run", keep="first"
    )
    df_LST = df_LST.sort_values(by=["DATE", "source", "LST1_run"])

    launch_jobs = ""
    for n, run in enumerate(list_of_bash_scripts):
        launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"

    os.system(launch_jobs)
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
