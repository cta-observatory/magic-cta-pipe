"""
This script creates the bashscripts necessary to apply "lst1_magic_dl1_stereo_to_dl2.py"
to the DL1 stereo data. It also creates new subdirectories associated with
the data level 2.

Usage:
$ DL1_to_DL2 -c configuration_file.yaml
"""
import argparse
import datetime
import glob
import logging
import os
from pathlib import Path

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

__all__ = ["ST_NSB_List", "bash_DL1Stereo_to_DL2"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def ST_NSB_List(
    target_dir, nsb_list, nsb_limit, source, df_LST, ST_list, ST_begin, ST_end
):
    """
    This function creates the lists of runs separeted by run period and NSB level.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    nsb_list : list
        List of the MC NSB values
    nsb_limit : list
        Edges of the NSB binning
    source : str
        Source name
    df_LST : :class:`pandas.DataFrame`
        Dataframe collecting the LST1 runs (produced by the create_LST_table script)
    ST_list : list
        List of the observastion periods
    ST_begin : list
        List of beginning dates for the observation periods
    ST_end : list
        List of ending dates for the observation periods
    """

    "Loops over all runs of all nights"
    Nights_list = np.sort(
        glob.glob(f"{target_dir}/v{__version__}/{source}/DL1Stereo/Merged/*")
    )
    for night in Nights_list:
        "Night period"
        night_date = night.split("/")[-1]
        print("night", night_date)
        date_lst = night_date[:4] + "_" + night_date[4:6] + "_" + night_date[6:8]
        print(date_lst)
        date_magic = datetime.datetime.strptime(
            date_lst, "%Y_%m_%d"
        ) + datetime.timedelta(days=1)
        print("dates", date_lst, str(date_magic))
        for p in range(len(ST_begin)):
            if (date_magic >= datetime.datetime.strptime(ST_begin[p], "%Y_%m_%d")) and (
                date_magic <= datetime.datetime.strptime(ST_end[p], "%Y_%m_%d")
            ):
                period = ST_list[p]

        Run_list = glob.glob(f"{night}/*.h5")
        for Run in Run_list:
            "getting the run NSB"
            print(Run)
            run_str = Run.split("/")[-1].split(".")[1]
            print("run", run_str)
            run_LST_id = run_str.lstrip("Run")
            print("run_lst", run_LST_id)
            nsb = df_LST[df_LST["LST1_run"] == run_LST_id]["nsb"].tolist()[0]
            print("nsb", nsb)
            "rounding the NSB to the nearest MC nsb value"
            for j in range(0, len(nsb_list) - 1):
                if (nsb <= nsb_limit[j + 1]) & (nsb > nsb_limit[j]):
                    nsb = nsb_list[j]
            "Writing on output .txt file"
            if nsb <= nsb_limit[-1]:
                with open(f"{night}/logs/{period}_{nsb}.txt", "a+") as file:
                    file.write(f"{Run}\n")


def bash_DL1Stereo_to_DL2(target_dir, source, env_name, cluster, RF_dir, df_LST):
    """
    This function generates the bashscript for running the DL1Stereo to DL2 analisys.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source : str
        Source name
    env_name : str
        Conda enviroment name
    cluster : str
        Cluster system
    RF_dir : str
        Path to the RFs
    df_LST : :class:`pandas.DataFrame`
        Dataframe collecting the LST1 runs (produced by the create_LST_table script)
    """
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return
    process_name = source
    Nights_list = np.sort(
        glob.glob(f"{target_dir}/v{__version__}/{source}/DL1Stereo/Merged/*")
    )
    for night in Nights_list:
        File_list = glob.glob(f"{night}/logs/ST*.txt")
        night_date = night.split("/")[-1]
        os.makedirs(
            f"{target_dir}/v{__version__}/{source}/DL2/{night_date}/logs", exist_ok=True
        )
        for file in File_list:
            with open(file, "r") as f:
                process_size = len(f.readlines()) - 1
            if process_size < 0:
                continue
            nsb = file.split("/")[-1].split("_")[-1][:3]
            p = file.split("/")[-1].split("_")[0]
            dec = df_LST[df_LST.source == source].iloc[0]["MC_dec"]
            RFdir = f"{RF_dir}/{p}/NSB{nsb}/dec_{dec}/"
            if (not os.path.isdir(RFdir)) or (len(os.listdir(RFdir)) == 0):
                continue
            slurm = slurm_lines(
                queue="short",
                job_name=f"{process_name}_DL1_to_DL2",
                array=process_size,
                mem="2g",
                out_name=f"{target_dir}/v{__version__}/{source}/DL2/{night.split('/')[-1]}/logs/slurm-%x.%A_%a",
            )
            rc = rc_lines(
                store="$SAMPLE ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID}",
                out="$OUTPUTDIR/logs/list",
            )

            lines = (
                slurm
                + [
                    f"SAMPLE_LIST=($(<{file}))\n",
                    "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
                    f"export LOG={target_dir}/v{__version__}/{source}/DL2/{night.split('/')[-1]}/logs",
                    "/DL1_to_DL2_${SLURM_ARRAY_TASK_ID}.log\n",
                    f"conda run -n {env_name} lst1_magic_dl1_stereo_to_dl2 --input-file-dl1 $SAMPLE --input-dir-rfs {RFdir} --output-dir {target_dir}/v{__version__}/{source}/DL2/{night.split('/')[-1]} >$LOG 2>&1\n\n",
                ]
                + rc
            )
            with open(
                f'{source}_DL1_to_DL2_{night_date}_{file.split("/")[-1].rstrip("txt")}sh',
                "w",
            ) as f:
                f.writelines(lines)


def main():
    """
    Here we read the config_general.yaml file and call the functions defined above.
    """

    parser = argparse.ArgumentParser()[
        "2026_01_01",
        "2024_05_18",
        "2023_03_09",
        "2022_08_31",
        "2022_06_09",
        "2021_09_29",
    ]
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
    RF_dir = config["directories"]["RF"]
    env_name = config["general"]["env_name"]
    ST_list = config["needed_parameters"]["ST_list"]
    ST_begin = config["needed_parameters"]["ST_begin"]
    ST_end = config["needed_parameters"]["ST_end"]
    nsb_list = config["needed_parameters"]["nsb"]
    width = [a / 2 - b / 2 for a, b in zip(nsb_list[1:], nsb_list[:-1])]
    width.append(0.25)
    nsb_limit = [a + b for a, b in zip(nsb_list[:], width[:])]
    nsb_limit.insert(0, 0)
    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]

    cluster = config["general"]["cluster"]

    "LST dataframe"
    config_db = resource_file("database_config.yaml")

    with open(
        config_db, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict_db = yaml.safe_load(fc)

    LST_h5 = config_dict_db["database_paths"]["LST"]
    LST_key = config_dict_db["database_keys"]["LST"]
    df_LST = pd.read_hdf(
        LST_h5,
        key=LST_key,
    )

    if source_in is None:
        source_list = joblib.load("list_sources.dat")
    else:
        source_list = [source]
    for source_name in source_list:
        ST_NSB_List(
            target_dir,
            nsb_list,
            nsb_limit,
            source_name,
            df_LST,
            ST_list,
            ST_begin,
            ST_end,
        )

        bash_DL1Stereo_to_DL2(target_dir, source_name, env_name, cluster, RF_dir)
        list_of_stereo_scripts = np.sort(glob.glob(f"{source_name}_DL1_to_DL2*.sh"))
        print(list_of_stereo_scripts)
        if len(list_of_stereo_scripts) < 1:
            logger.warning(f"No bash scripts for {source_name}")
            continue
        launch_jobs = ""
        for n, run in enumerate(list_of_stereo_scripts):
            launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"
        os.system(launch_jobs)


if __name__ == "__main__":
    main()
