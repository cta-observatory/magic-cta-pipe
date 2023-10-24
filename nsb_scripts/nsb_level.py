"""
Bash scripts to run LSTnsb.py on all the LST runs by using parallel jobs
"""

import argparse
import glob
import logging
import os

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def bash_scripts(run, config, source, env_name):
    lines = [
        "#!/bin/sh\n\n",
        "#SBATCH -p long\n",
        "#SBATCH -J nsb\n",
        "#SBATCH -N 1\n\n",
        "ulimit -l unlimited\n",
        "ulimit -s unlimited\n",
        "ulimit -a\n\n",
        f"time conda run -n  {env_name} python LSTnsb -c {config} -i {run} > {source}_nsblog_{run}.log 2>&1 \n\n",
    ]
    with open(f"{source}_run_{run}.sh", "w") as f:
        f.writelines(lines)


def main():
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
    runs = config["general"]["LST_runs"]
    env_name = config["general"]["env_name"]

    with open(str(runs), "r") as LSTfile:
        run = LSTfile.readlines()
    print("***** Generating bashscripts...")
    for i in run:
        i = i.rstrip()
        bash_scripts(i, args.config_file, source, env_name)
    print("Process name: nsb")
    print("To check the jobs submitted to the cluster, type: squeue -n nsb")
    list_of_bash_scripts = np.sort(glob.glob(f"{source}_run_*.sh"))

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
