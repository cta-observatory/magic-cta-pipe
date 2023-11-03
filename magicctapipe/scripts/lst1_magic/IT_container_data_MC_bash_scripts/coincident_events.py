"""
This scripts facilitates the usage of the script
"lst1_magic_event_coincidence.py". This script is
more like a "maneger" that organizes the analysis
process by:
1) Creating the bash scripts for looking for
coincidence events between MAGIC and LST in each
night.
2) Creating the subdirectories for the coincident
event files.


Usage:
$ python coincident_events.py (-c config.yaml)

"""

import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import yaml

__all__ = ["configfile_coincidence", "linking_lst", "bash_coincident"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configfile_coincidence(ids, target_dir):
    """
    This function creates the configuration file needed for the event coincidence step

    Parameters
    ----------
    ids: list
        list of telescope IDs
    target_dir: str
        Path to the working directory
    """

    with open(f"{target_dir}/config_coincidence.yaml", "w") as f:
        lines = [
            f"mc_tel_ids:\n    LST-1: {ids[0]}\n    LST-2: {ids[1]}\n    LST-3: {ids[2]}\n    LST-4: {ids[3]}\n    MAGIC-I: {ids[4]}\n    MAGIC-II: {ids[5]}\n\n",
            'event_coincidence:\n    timestamp_type_lst: "dragon_time"  # select "dragon_time", "tib_time" or "ucts_time"\n    pre_offset_search: true\n    n_pre_offset_search_events: 100\n    window_half_width: "300 ns"\n',
            '    time_offset:\n        start: "-10 us"\n        stop: "0 us"\n',
        ]

        f.writelines(lines)


def linking_lst(target_dir, LST_runs, LST_version):
    """
    This function links the LST data paths to the working directory. This is a preparation step required for running lst1_magic_event_coincidence.py

    Parameters
    ----------
    target_dir: str
        Path to the working directory
    LST_runs: matrix of strings
        This matrix is imported from config_general.yaml and tells the function where to find the LST data and link them to our working directory
    LST_version: str
        Version of lstchain used to process data
    """

    coincidence_DL1_dir = f"{target_dir}/DL1/Observations"
    if not os.path.exists(f"{coincidence_DL1_dir}/Coincident"):
        os.mkdir(f"{coincidence_DL1_dir}/Coincident")

    for i in LST_runs:
        lstObsDir = i[0].split("_")[0] + i[0].split("_")[1] + i[0].split("_")[2]
        inputdir = f"/fefs/aswg/data/real/DL1/{lstObsDir}/{LST_version}/tailcut84"
        outputdir = f"{coincidence_DL1_dir}/Coincident/{lstObsDir}"
        list_of_subruns = np.sort(glob.glob(f"{inputdir}/dl1*Run*{i[1]}*.*.h5"))
        if os.path.exists(f"{outputdir}/list_LST.txt"):
            with open(f"{outputdir}/list_LST.txt", "a") as LSTdataPathFile:
                for subrun in list_of_subruns:
                    LSTdataPathFile.write(
                        f"{subrun}\n"
                    )  # If this files already exists, simply append the new information
        else:
            os.mkdir(outputdir)
            with open(
                f"{outputdir}/list_LST.txt", "w"
            ) as f:  # If the file list_LST.txt does not exist, it will be created here
                for subrun in list_of_subruns:
                    f.write(f"{subrun}\n")


def bash_coincident(target_dir, env_name):
    """
    This function generates the bashscript for running the coincidence analysis.

    Parameters
    ----------
    target_dir: str
        Path to the working directory
    env_name: str
        Name of the environment
    """

    process_name = target_dir.split("/")[-2:][1]

    listOfNightsLST = np.sort(glob.glob(f"{target_dir}/DL1/Observations/Coincident/*"))
    listOfNightsMAGIC = np.sort(
        glob.glob(f"{target_dir}/DL1/Observations/Merged/Merged*")
    )

    for nightMAGIC, nightLST in zip(listOfNightsMAGIC, listOfNightsLST):
        process_size = len(np.genfromtxt(f"{nightLST}/list_LST.txt", dtype="str")) - 1

        with open(f"LST_coincident_{nightLST.split('/')[-1]}.sh", "w") as f:
            lines = [
                "#!/bin/sh\n\n",
                "#SBATCH -p short\n",
                f"#SBATCH -J {process_name}_coincidence\n",
                f"#SBATCH --array=0-{process_size}%50\n",
                "#SBATCH -N 1\n\n",
                "ulimit -l unlimited\n",
                "ulimit -s unlimited\n",
                "ulimit -a\n\n",
                f"export INM={nightMAGIC}\n",
                f"export OUTPUTDIR={nightLST}\n",
                "SAMPLE_LIST=($(<$OUTPUTDIR/list_LST.txt))\n",
                "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
                "export LOG=$OUTPUTDIR/coincidence_${SLURM_ARRAY_TASK_ID}.log\n",
                f"conda run -n {env_name} lst1_magic_event_coincidence --input-file-lst $SAMPLE --input-dir-magic $INM --output-dir $OUTPUTDIR --config-file {target_dir}/config_coincidence.yaml >$LOG 2>&1",
            ]
            f.writelines(lines)


def main():
    """
    Here we read the config_general.yaml file and call the functions defined above.
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

    telescope_ids = list(config["mc_tel_ids"].values())
    target_dir = f'{Path(config["directories"]["workspace_dir"])}/{config["directories"]["target_name"]}'

    env_name = config["general"]["env_name"]

    LST_runs_and_dates = config["general"]["LST_runs"]
    LST_runs = np.genfromtxt(LST_runs_and_dates, dtype=str, delimiter=",")
    LST_version = config["general"]["LST_version"]

    print("***** Generating file config_coincidence.yaml...")
    print("***** This file can be found in ", target_dir)
    configfile_coincidence(telescope_ids, target_dir)

    print("***** Linking the paths to LST data files...")
    linking_lst(
        target_dir, LST_runs, LST_version
    )  # linking the data paths to current working directory

    print("***** Generating the bashscript...")
    bash_coincident(target_dir, env_name)

    print("***** Submitting processess to the cluster...")
    print(f"Process name: {target_dir.split('/')[-2:][1]}_coincidence")
    print(
        f"To check the jobs submitted to the cluster, type: squeue -n {target_dir.split('/')[-2:][1]}_coincidence"
    )

    # Below we run the bash scripts to find the coincident events
    list_of_coincidence_scripts = np.sort(glob.glob("LST_coincident*.sh"))

    for n, run in enumerate(list_of_coincidence_scripts):
        if n == 0:
            launch_jobs = f"coincidence{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = f"{launch_jobs} && coincidence{n}=$(sbatch --parsable --dependency=afterany:$coincidence{n-1} {run})"

    os.system(launch_jobs)


if __name__ == "__main__":
    main()
