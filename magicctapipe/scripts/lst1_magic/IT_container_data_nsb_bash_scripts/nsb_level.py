"""
Bash scripts to run LSTnsb.py on all the LST runs by using parallel jobs

Usage: python nsb_level.py (-c config.yaml)
"""

import argparse
import glob
import logging
import os

import numpy as np
import yaml

__all__ = ["bash_scripts"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def bash_scripts(run, date, config, source, env_name):

    """Here we create the bash scripts (one per LST run)

    Parameters
    ----------
    run : str
        LST run number
    date : str
        LST date
    config : str
        Name of the configuration file
    source : str
        Target name
    env_name : str
        Name of the environment
    """

    lines = [
        "#!/bin/sh\n\n",
        "#SBATCH -p long\n",
        "#SBATCH -J nsb\n",
        "#SBATCH -N 1\n\n",
        "ulimit -l unlimited\n",
        "ulimit -s unlimited\n",
        "ulimit -a\n\n",
        f"time conda run -n  {env_name} LSTnsb -c {config} -i {run} -d {date} > {source}_nsblog_{run}.log 2>&1 \n\n",
    ]
    with open(f"{source}_{date}_run_{run}.sh", "w") as f:
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

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    source = config["directories"]["target_name"]
    lst_runs_filename = config["general"]["LST_runs"]
    env_name = config["general"]["env_name"]

    with open(str(lst_runs_filename), "r") as LSTfile:
        run_list = LSTfile.readlines()
    print("***** Generating bashscripts...")
    for run in run_list:
        run = run.rstrip()
        run_number = run.split(",")[1]
        date = run.split(",")[0]
        bash_scripts(run_number, date, args.config_file, source, env_name)
    print("Process name: nsb")
    print("To check the jobs submitted to the cluster, type: squeue -n nsb")
    list_of_bash_scripts = np.sort(glob.glob(f"{source}_*_run_*.sh"))

    if len(list_of_bash_scripts) < 1:
        return
    for n, run in enumerate(list_of_bash_scripts):
        if n == 0:
            launch_jobs = f"nsb{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = f"{launch_jobs} && nsb{n}=$(sbatch --parsable {run})"

    # print(launch_jobs)
    os.system(launch_jobs)


if __name__ == "__main__":
    main()
