"""
This script creates the bashscripts necessary to apply "lst1_magic_dl1_stereo_to_dl2.py"
to the DL1 stereo data (real and MC). It also creates new subdirectories associated with
the data level 2. The DL2 files are saved at:
WorkingDirectory/DL2/
and in the subdirectories therein.

Usage:
$ python DL1_to_DL2.py

"""
import argparse
import glob
import logging
import os
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml
from magicctapipe import __version__
from magicctapipe.io import resource_file
from magicctapipe.scripts.lst1_magic.semi_automatic_scripts.clusters import (
    rc_lines,
    slurm_lines,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def ST_NSB_List(target_dir, nsb_list, nsb_limit):
    """
    This function creates the sists of runs separeted by run period and NSB level.

    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """ 

    "NSB dataframe"
    df_LST = pd.read_hdf(
        "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/observations_LST.h5",
        key="joint_obs",
    )

    "Period Lists"
    ST_list = ["ST0320A", "ST0319A", "ST0318A", "ST0317A", "ST0316A"]
    ST_begin = ["2023_03_10", "2022_12_15", "2022_06_10", "2021_12_30", "2020_10_24"]
    ST_end = ["2025_01_30", "2023_03_09", "2022_08_31", "2022_06_09", "2021_09_29"]
    # ST0320 ongoing -> 'service' end date

    "Loops over all runs of all nights"
    Nights_list = np.sort(glob.glob(f"{target_dir}/v{__version__}/DL1Stereo/*"))
    for night in Nights_list:
        "Night period"
        date_lst = night[:4] + "_" + night[4:6] + "_" + night[6:8]
        delta = timedelta(days=1)
        date_magic = date_lst + delta
        for p in range(len(ST_begin)):
            if (
                time.strptime(date_magic, "%Y_%m_%d")
                >= time.strptime(ST_begin[p], "%Y_%m_%d")
                ) and (
                time.strptime(date_magic, "%Y_%m_%d")
                <= time.strptime(ST_end[p], "%Y_%m_%d")
                ):
                period = ST_list[p]

        Run_list = [
            os.path.basename(x)
            for x in glob.glob(f"{target_dir}/v{__version__}/DL1Stereo/{night}/*")
        ]
        for Run in Run_list:
            "getting the run NSB"
            run_str=Run.split('.')[1]
            run_LST_id = run_str[4:]
            nsb = df_LST[df_LST["LST_id"] == run_LST_id]["nsb"]
            "rounding the NSB to the nearest MC nsb value"
            for j in range(0, len(nsb_list)-1):
                if (nsb < nsb_limit[j + 1]) & (nsb > nsb_limit[j]):
                    nsb = nsb_list[j]
            "Writing on output .txt file"
            if (nsb <= 3):
                with open(f"{target_dir}/v{__version__}/DL1Stereo/{night}/{period}_{nsb}.txt","a+") as file:
                    file.write(f"{target_dir}/v{__version__}/DL1Stereo/{night}/{Run}\n")


def bash_DL1Stereo_to_DL2(target_dir, source, env_name):
    """
    This function generates the bashscript for running the DL1Stereo to DL2 analisys.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    """

    process_name = source

    Nights_list = np.sort(glob.glob(f"{target_dir}/v{__version__}/DL1Stereo/*"))
    for night in Nights_list:
        File_list = glob.glob(f"{target_dir}/v{__version__}/DL1Stereo/{night}/*.txt")
        for file in File_list:
            with open(f"{target_dir}/v{__version__}/DL1Stereo/{night}/{file}.txt", "r") as f:
                process_size = len(f.readlines()) - 1
            if process_size < 0:
                continue
            with open(f'{source}_DL1_to_DL2_{night.split("/")[-1]}.sh', "w") as f:
                f.write("#!/bin/sh\n\n")
                f.write("#SBATCH -p short\n")
                f.write("#SBATCH -J " + process_name + "\n")
                f.write(f"#SBATCH --array=0-{process_size}%100\n")
                f.write("#SBATCH --mem=30g\n")
                f.write("#SBATCH -N 1\n\n")
                f.write("ulimit -l unlimited\n")
                f.write("ulimit -s unlimited\n")
                f.write("ulimit -a\n\n")

                f.write(
                    f"SAMPLE_LIST=($(<{target_dir}/v{__version__}/DL1Stereo/{night}/{file}.txt))\n"
                )
                f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
                f.write(
                    f"export LOG={target_dir}/v{__version__}/DL2"
                    + "/logs/DL1_to_DL2_${SLURM_ARRAY_TASK_ID}.log\n"
                )
                f.write(
                    f"conda run -n {env_name} lst1_magic_dl1_stereo_to_dl2 --input-file-dl1 $SAMPLE --input-dir-rfs {RFs_dir} --output-dir {target_dir}/v{__version__}/DL2 >$LOG 2>&1\n\n"
                )



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

    target_dir = str(
        Path(config["directories"]["workspace_dir"])
        / config["directories"]["target_name"]
    )
    env_name = config["general"]["env_name"]
    scripts_dir = str(Path(config["directories"]["scripts_dir"]))
    source = config["directories"]["target_name"]
    listnsb = np.sort(glob.glob(f"{source}_LST_*_.txt"))
    nsb_list = config["general"]["nsb"]
    width = [a / 2 - b / 2 for a, b in zip(nsb_list[1:], nsb_list[:-1])]
    width.append(0.25)
    nsb_limit = [a + b for a, b in zip(nsb_list[:], width[:])]
    nsb_limit.insert(0, 0)

    ST_NSB_List(target_dir, nsb_list, nsb_limit)

    bash_DL1Stereo_to_DL2(target_dir,source, env_name)


if __name__ == "__main__":
    main()
