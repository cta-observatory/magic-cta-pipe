"""
This scripts generates and runs the bashscripts
to compute the stereo parameters of DL1
Coincident MAGIC+LST data files.

Usage:
$ stereo_events (-c config.yaml)
"""

import argparse
import glob
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import yaml

from magicctapipe import __version__
from magicctapipe.io import resource_file
from magicctapipe.scripts.lst1_magic.semi_automatic_scripts.clusters import (
    rc_lines,
    slurm_lines,
)

__all__ = ["configfile_stereo", "bash_stereo"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configfile_stereo(target_dir, source_name, config_file):

    """
    This function creates the configuration file needed for the stereo reconstruction step

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source_name : str
        Name of the target source
    config_file : str
        Path to MCP configuration file (e.g., resources/config.yaml)
    """

    if config_file == "":
        config_file = resource_file("config.yaml")

    with open(
        config_file, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)
    conf = {
        "mc_tel_ids": config_dict["mc_tel_ids"],
        "stereo_reco": config_dict["stereo_reco"],
    }

    conf_dir = f"{target_dir}/v{__version__}/{source_name}"
    os.makedirs(conf_dir, exist_ok=True)

    file_name = f"{conf_dir}/config_stereo.yaml"

    with open(file_name, "w") as f:

        yaml.dump(conf, f, default_flow_style=False)


def bash_stereo(target_dir, source, env_name, cluster, version):

    """
    This function generates the bashscripts for running the stereo analysis.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source : str
        Target name
    env_name : str
        Name of the environment
    cluster : str
        Cluster system
    version : str
        Version of the input (coincident) data
    """

    process_name = source

    coincidence_DL1_dir = f"{target_dir}/v{version}/{source}"

    listOfNightsLST = np.sort(glob.glob(f"{coincidence_DL1_dir}/DL1Coincident/*"))
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return
    for nightLST in listOfNightsLST:
        night = nightLST.split("/")[-1]
        stereoDir = f"{target_dir}/v{__version__}/{source}/DL1Stereo/{night}"
        os.makedirs(f"{stereoDir}/logs", exist_ok=True)
        if not os.listdir(f"{nightLST}"):
            continue
        if len(os.listdir(nightLST)) < 2:
            continue

        os.system(
            f"ls {nightLST}/*LST*.h5 >  {stereoDir}/logs/list_coin.txt"
        )  # generating a list with the DL1 coincident data files.
        with open(f"{stereoDir}/logs/list_coin.txt", "r") as f:
            process_size = len(f.readlines()) - 1

        if process_size < 0:
            continue

        slurm = slurm_lines(
            queue="short",
            job_name=f"{process_name}_stereo",
            array=process_size,
            mem="2g",
            out_name=f"{stereoDir}/logs/slurm-%x.%A_%a",
        )
        rc = rc_lines(
            store="$SAMPLE ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID}",
            out="$OUTPUTDIR/logs/list",
        )
        lines = (
            slurm
            + [
                f"export INPUTDIR={nightLST}\n",
                f"export OUTPUTDIR={stereoDir}\n",
                "SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_coin.txt))\n",
                "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
                "export LOG=$OUTPUTDIR/logs/stereo_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                f"conda run -n {env_name} lst1_magic_stereo_reco --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/v{__version__}/{source}/config_stereo.yaml >$LOG 2>&1\n",
            ]
            + rc
        )
        with open(f"{source}_StereoEvents_{night}.sh", "w") as f:
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
        default="./config_auto_MCP.yaml",
        help="Path to a configuration file",
    )

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)

    target_dir = Path(config["directories"]["workspace_dir"])

    env_name = config["general"]["env_name"]
    config_file = config["general"]["base_config_file"]

    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]

    cluster = config["general"]["cluster"]
    in_version = config["directories"]["real_input_version"]
    if in_version == "":
        in_version = __version__

    if source_in is None:
        source_list = joblib.load("list_sources.dat")
    else:
        if source is None:
            source = source_in
        source_list = [source]

    for source_name in source_list:

        print("***** Generating file config_stereo.yaml...")
        configfile_stereo(target_dir, source_name, config_file)

        # Below we run the analysis on the real data

        print("***** Generating the bashscript...")
        bash_stereo(target_dir, source_name, env_name, cluster, in_version)

        print("***** Submitting processess to the cluster...")
        print(f"Process name: {source_name}_stereo")
        print(
            f"To check the jobs submitted to the cluster, type: squeue -n {source_name}_stereo"
        )

        # Below we run the bash scripts to find the stereo events
        list_of_stereo_scripts = np.sort(glob.glob(f"{source_name}_StereoEvents*.sh"))
        if len(list_of_stereo_scripts) < 1:
            logger.warning("No bash scripts for real data")
            continue
        launch_jobs = ""
        for n, run in enumerate(list_of_stereo_scripts):
            launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"

        os.system(launch_jobs)


if __name__ == "__main__":
    main()
