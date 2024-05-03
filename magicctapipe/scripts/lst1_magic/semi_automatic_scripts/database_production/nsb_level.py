"""
Bash scripts to run LSTnsb.py on all the LST runs by using parallel jobs

Usage: python nsb_level.py (-c config.yaml)
"""

import argparse
import glob
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

__all__ = ["bash_scripts"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def bash_scripts(run, date, config, env_name):

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
    """

    lines = [
        "#!/bin/sh\n\n",
        "#SBATCH -p short,long\n",
        "#SBATCH -J nsb\n",
        "#SBATCH -n 1\n\n",
        "#SBATCH --output=slurm-nsb-%x.%j.out"
        "#SBATCH --error=slurm-nsb-%x.%j.err"
        "ulimit -l unlimited\n",
        "ulimit -s unlimited\n",
        "ulimit -a\n\n",
        f"conda run -n  {env_name} LSTnsb -c {config} -i {run} -d {date} > nsblog_{date}_{run}_"
        + "${SLURM_JOB_ID}.log 2>&1 \n\n",
    ]
    with open(f"nsb_{date}_run_{run}.sh", "w") as f:
        f.writelines(lines)


def main():

    """
    Main function
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_general.yaml",
        help="Path to a configuration file",
    )
    parser.add_argument(
        "--begin-date",
        "-b",
        dest="begin_date",
        type=str,
        help="Begin date to start NSB evaluation from the database.",
    )
    parser.add_argument(
        "--end-date",
        "-e",
        dest="end_date",
        type=str,
        help="End date to start NSB evaluation from the database.",
    )
    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)

    env_name = config["general"]["env_name"]

    df_LST = pd.read_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5",
        key="joint_obs",
    )

    min = datetime.strptime(args.begin_date, "%Y_%m_%d")
    max = datetime.strptime(args.end_date, "%Y_%m_%d")
    lst = pd.to_datetime(df_LST["DATE"].str.replace("_", "-"))
    df_LST["date"] = lst
    df_LST = df_LST[df_LST["date"] >= min]
    df_LST = df_LST[df_LST["date"] <= max]

    df_LST = df_LST.drop(columns="date")
    run_LST = df_LST["LST1_run"]
    date_LST = df_LST["DATE"]
    print("***** Generating bashscripts...")
    for run_number, date in zip(run_LST, date_LST):
        bash_scripts(run_number, date, args.config_file, env_name)
    print("Process name: nsb")
    print("To check the jobs submitted to the cluster, type: squeue -n nsb")
    list_of_bash_scripts = np.sort(glob.glob("nsb_*_run_*.sh"))

    if len(list_of_bash_scripts) < 1:
        print(
            "Warning: no bash script has been produced to evaluate the NSB level for the provided LST runs. Please check the input list"
        )
        return
    for n, run in enumerate(list_of_bash_scripts):
        if n == 0:
            launch_jobs = f"nsb{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = f"{launch_jobs} && nsb{n}=$(sbatch --parsable {run})"

    os.system(launch_jobs)


if __name__ == "__main__":
    main()
