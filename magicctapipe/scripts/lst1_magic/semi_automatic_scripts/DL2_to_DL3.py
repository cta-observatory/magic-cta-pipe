"""
This script creates the bashscripts necessary to apply "lst1_magic_dl2_to_dl3.py"
to the DL2. It also creates new subdirectories associated with
the data level 3.

Usage:
$ python new_DL2_to_DL3.py -c configuration_file.yaml
"""
import argparse
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

__all__ = ["configuration_DL3", "DL2_to_DL3"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configuration_DL3(target_dir, source_name, config_file, ra, dec):
    """
    This function creates the configuration file needed for the DL2 to DL3 conversion

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source_name : str
        Source name
    config_file : str
        Path to MCP configuration file (e.g., resources/config.yaml)
    ra : float
        Source RA
    dec : float
        Source Dec
    """

    if config_file == "":
        config_file = resource_file("config.yaml")

    with open(
        config_file, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)
    DL3_config = config_dict["dl2_to_dl3"]
    DL3_config["source_name"] = source_name
    DL3_config["source_ra"] = ra
    DL3_config["source_dec"] = dec
    conf = {
        "mc_tel_ids": config_dict["mc_tel_ids"],
        "dl2_to_dl3": DL3_config,
    }

    conf_dir = f"{target_dir}/v{__version__}/{source_name}"
    os.makedirs(conf_dir, exist_ok=True)

    file_name = f"{conf_dir}/config_DL3.yaml"

    with open(file_name, "w") as f:

        yaml.dump(conf, f, default_flow_style=False)


def DL2_to_DL3(target_dir, source, env_name, IRF_dir, df_LST, cluster):
    """
    This function creates the bash scripts to run lst1_magic_dl2_to_dl3.py on the real data.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source : str
        Source name
    env_name : str
        Conda enviroment name
    IRF_dir : str
        Path to the IRFs
    df_LST : :class:`pandas.DataFrame`
        Dataframe collecting the LST1 runs (produced by the create_LST_table script)
    cluster : str
        Cluster system
    """
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return

    os.makedirs(f"{target_dir}/v{__version__}/{source}/DL3/logs", exist_ok=True)

    # Loop over all nights
    Nights_list = np.sort(glob.glob(f"{target_dir}/v{__version__}/{source}/DL2/*"))
    for night in Nights_list:
        # Loop over every run:
        File_list = np.sort(glob.glob(f"{night}/*.txt"))
        for file in File_list:
            with open(file, "r") as f:
                runs = f.readlines()
                process_size = len(runs) - 1
                run_new = [
                    run.split("DL1Stereo")[0]
                    + "DL2"
                    + run.split("DL1Stereo")[1]
                    .replace("dl1_stereo", "dl2")
                    .replace("/Merged", "")
                    for run in runs
                ]
                with open(file, "w") as g:
                    g.writelines(run_new)

            nsb = file.split("/")[-1].split("_")[-1][:3]
            period = file.split("/")[-1].split("_")[0]
            dec = df_LST[df_LST.source == source].iloc[0]["MC_dec"]
            IRFdir = f"{IRF_dir}/{period}/NSB{nsb}/dec_{dec}/"
            if (not os.path.isdir(IRFdir)) or (len(os.listdir(IRFdir)) == 0):
                continue
            process_name = source
            output = f"{target_dir}/v{__version__}/{source}/DL3"

            slurm = slurm_lines(
                queue="short",
                job_name=f"{process_name}_DL2_to_DL3",
                array=process_size,
                mem="1g",
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
                    f"export LOG={output}/logs",
                    "/DL2_to_DL3_${SLURM_ARRAY_TASK_ID}.log\n",
                    f"conda run -n {env_name} lst1_magic_dl2_to_dl3 --input-file-dl2 $SAMPLE --input-dir-irf {IRFdir} --output-dir {output} --config-file {target_dir}/config_DL3.yaml >$LOG 2>&1\n\n",
                ]
                + rc
            )
            with open(
                f'{source}_DL2_to_DL3_{nsb}_{period}_{night.split("/")[-1]}.sh', "w"
            ) as f:
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

    target_dir = Path(config["directories"]["workspace_dir"])
    IRF_dir = config["directories"]["IRF"]

    print("***** Generating file config_DL3.yaml...")
    print("***** This file can be found in ", target_dir)

    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    env_name = config["general"]["env_name"]
    config_file = config["general"]["base_config_file"]
    cluster = config["general"]["cluster"]

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

    # cp the .txt files from DL1 stereo anaysis to be used again.
    DL1stereo_Nights = np.sort(
        glob.glob(f"{target_dir}/v{__version__}/{source}/DL1Stereo/Merged/*")
    )
    for night in DL1stereo_Nights:
        File_list = glob.glob(f"{night}/logs/ST*.txt")
        night_date = night.split("/")[-1]
        for file in File_list:
            cp_dir = f"{target_dir}/v{__version__}/{source}/DL2/{night_date}"
            os.system(f"cp {file} {cp_dir}")

    if source_in is None:
        source_list = joblib.load("list_sources.dat")
    else:
        source_list = [source]
    for source_name in source_list:
        wobble_offset = df_LST[df_LST.source == source].iloc[0]["wobble_offset"]
        if str(wobble_offset) != "0.4":
            continue
        ra = df_LST[df_LST.source == source].iloc[0]["ra"]
        dec = df_LST[df_LST.source == source].iloc[0]["dec"]
        configuration_DL3(target_dir, source_name, config_file, ra, dec)
        DL2_to_DL3(target_dir, source_name, env_name, IRF_dir, df_LST, cluster)
        list_of_dl3_scripts = np.sort(glob.glob(f"{source_name}_DL2_to_DL3*.sh"))
        if len(list_of_dl3_scripts) < 1:
            logger.warning(f"No bash scripts for {source_name}")
            continue
        launch_jobs = ""
        for n, run in enumerate(list_of_dl3_scripts):
            launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"
        os.system(launch_jobs)


if __name__ == "__main__":
    main()
