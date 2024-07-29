"""
This script creates the bashscripts necessary to apply "lst1_magic_dl1_stereo_to_dl2.py"
to the DL1 stereo data (real and MC). It also creates new subdirectories associated with
the data level 2. The DL2 files are saved at:
WorkingDirectory/DL2/
and in the subdirectories therein.

Usage:
$ python DL1_to_DL2.py -c configuration_file.yaml

"""
import argparse
import glob
import logging
import os
import time
from pathlib import Path
import datetime 
import joblib
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


def ST_NSB_List(target_dir, nsb_list, nsb_limit, source):
    """
    This function creates the lists of runs separeted by run period and NSB level.

    Parameters
    ----------
    target_dir: str
        Path to the working directory
    nsb_list:
    nsb_limit:
    	
    source:
    	source name
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
    Nights_list = np.sort(glob.glob(f"{target_dir}/v{__version__}/{source}/DL1Stereo/*"))
    for night in Nights_list:
        "Night period"
        night_date=night.split('/')[-1]
        print('night',night_date)
        date_lst = night_date[:4] + "_" + night_date[4:6] + "_" + night_date[6:8]
        print(date_lst)
        date_magic = datetime.datetime.strptime(date_lst, "%Y_%m_%d") + datetime.timedelta(days=1)
        print('dates', date_lst, str(date_magic))
        for p in range(len(ST_begin)):
            if (
                date_magic
                >= datetime.datetime.strptime(ST_begin[p], "%Y_%m_%d")
                ) and (
                date_magic
                <= datetime.datetime.strptime(ST_end[p], "%Y_%m_%d")
                ):
                period = ST_list[p]

        Run_list = glob.glob(f"{night}/Merged/*.h5")
        for Run in Run_list:
            "getting the run NSB"
            print(Run)
            run_str=Run.split('/')[-1].split('.')[1]
            print('run', run_str)
            run_LST_id = run_str.lstrip('Run')
            print('run_lst', run_LST_id)
            nsb = df_LST[df_LST["LST1_run"] == run_LST_id]["nsb"].tolist()[0]
            print('nsb', nsb)
            "rounding the NSB to the nearest MC nsb value"
            for j in range(0, len(nsb_list)-1):
                if (nsb < nsb_limit[j + 1]) & (nsb > nsb_limit[j]):
                    nsb = nsb_list[j]
            "Writing on output .txt file"
            if (nsb <= 3.1):
                with open(f"{night}/Merged/logs/{period}_{nsb}.txt","a+") as file:
                    file.write(f"{Run}\n")


def bash_DL1Stereo_to_DL2(target_dir, source, env_name):
    """
    This function generates the bashscript for running the DL1Stereo to DL2 analisys.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source:
    	source name
    env_name:
    	conda enviroment name
    """
    
    process_name = source
    Nights_list = np.sort(glob.glob(f"{target_dir}/v{__version__}/{source}/DL1Stereo/*"))
    for night in Nights_list:
        File_list = glob.glob(f"{night}/Merged/logs/ST*.txt")
        night_date=night.split('/')[-1]
        os.makedirs(f'{target_dir}/v{__version__}/{source}/DL2/{night_date}/logs',exist_ok=True)
        for file in File_list:
            with open(file, "r") as f:
                process_size = len(f.readlines()) - 1
            if process_size < 0:
                continue
            nsb=file.split("/")[-1].split("_")[-1][:3]
            p=file.split("/")[-1].split("_")[0]           
            RFdir=f'/fefs/aswg/LST1MAGIC/mc/models/{p}/NSB{nsb}/v01.2/dec_2276/'
            with open(f'{source}_DL1_to_DL2_{night_date}_{file.split("/")[-1].rstrip("txt")}sh', "w") as f:
                f.write("#!/bin/sh\n\n")
                f.write("#SBATCH -p long\n")
                f.write("#SBATCH -J " + process_name + "\n")
                f.write(f"#SBATCH --array=0-{process_size}%100\n")
                f.write("#SBATCH --mem=90g\n")
                f.write("#SBATCH -N 1\n\n")
                f.write("ulimit -l unlimited\n")
                f.write("ulimit -s unlimited\n")
                f.write("ulimit -a\n\n")

                f.write(
                    f"SAMPLE_LIST=($(<{file}))\n"
                )
                f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
                f.write(
                    f"export LOG={target_dir}/v{__version__}/{source}/DL2/{night.split('/')[-1]}/logs"
                    + "/DL1_to_DL2_${SLURM_ARRAY_TASK_ID}.log\n"
                )
                f.write(
                    f"conda run -n {env_name} lst1_magic_dl1_stereo_to_dl2 --input-file-dl1 $SAMPLE --input-dir-rfs {RFdir} --output-dir {target_dir}/v{__version__}/{source}/DL2/{night.split('/')[-1]} >$LOG 2>&1\n\n"
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

    target_dir = Path(config["directories"]["workspace_dir"])
    env_name = config["general"]["env_name"]
    nsb_list = config["general"]["nsb"]
    width = [a / 2 - b / 2 for a, b in zip(nsb_list[1:], nsb_list[:-1])]
    width.append(0.25)
    nsb_limit = [a + b for a, b in zip(nsb_list[:], width[:])]
    nsb_limit.insert(0, 0)
    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]

    cluster = config["general"]["cluster"]

    if source_in is None:
        source_list = joblib.load("list_sources.dat")
    else:
        source_list = [source]
    for source_name in source_list:
        ST_NSB_List(target_dir, nsb_list, nsb_limit, source_name)

        bash_DL1Stereo_to_DL2(target_dir,source_name, env_name)
        list_of_stereo_scripts = np.sort(glob.glob(f'{source_name}_DL1_to_DL2*.sh'))
        print(list_of_stereo_scripts)
        for n, run in enumerate(list_of_stereo_scripts):
            if n == 0:
                launch_jobs = f"stereo{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = (
                    f"{launch_jobs} && stereo{n}=$(sbatch --parsable {run})"
                )
        print(launch_jobs)
        os.system(launch_jobs)
