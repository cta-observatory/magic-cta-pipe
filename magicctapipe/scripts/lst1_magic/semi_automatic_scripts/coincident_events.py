"""
This scripts facilitates the usage of the script
"lst1_magic_event_coincidence.py". This script is
more like a "manager" that organizes the analysis
process by:
1) Creating the bash scripts for looking for
coincidence events between MAGIC and LST in each
night.
2) Creating the subdirectories for the coincident
event files.

Usage:
$ coincident_events (-c config)
"""
import argparse
import glob
import logging
import os
from datetime import date as dtdt
from datetime import timedelta
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

__all__ = ["configfile_coincidence", "linking_bash_lst"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configfile_coincidence(target_dir, source_name, config_file):

    """
    This function creates the configuration file needed for the event coincidence step

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
        "event_coincidence": config_dict["event_coincidence"],
    }

    conf_dir = f"{target_dir}/v{__version__}/{source_name}"
    os.makedirs(conf_dir, exist_ok=True)

    file_name = f"{conf_dir}/config_coincidence.yaml"

    with open(file_name, "w") as f:

        yaml.dump(conf, f, default_flow_style=False)


def linking_bash_lst(
    target_dir, LST_runs, source_name, LST_version, env_name, cluster, version, nice
):
    """
    This function links the LST data paths to the working directory and creates bash scripts.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    LST_runs : matrix of strings
        This matrix ([['date','run'],['date','run']...]) is imported from *_LST_runs.txt files and tells the function where to find the LST data and link them to our working directory
    source_name : str
        Target name
    LST_version : str
        The lstchain version used to process the LST data
    env_name : str
        Name of the conda environment
    cluster : str
        Cluster system
    version : str
        Version of the input (DL1 MAGIC runs) data
    nice : int or None
        Job priority
    """

    coincidence_DL1_dir = f"{target_dir}/v{__version__}/{source_name}"

    MAGIC_DL1_dir = f"{target_dir}/v{version}/{source_name}/DL1"

    dates = [os.path.basename(x) for x in glob.glob(f"{MAGIC_DL1_dir}/Merged/[0-9]*")]
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return
    for d in dates:
        Y_M, M_M, D_M = [int(x) for x in d.split("_")[:]]

        day_MAGIC = dtdt(Y_M, M_M, D_M)

        delta = timedelta(days=1)
        for i in LST_runs:
            Y_L, M_L, D_L = [int(x) for x in i[0].split("_")]

            day_LST = dtdt(int(Y_L), int(M_L), int(D_L))
            if day_MAGIC == day_LST + delta:

                lstObsDir = i[0].replace("_", "")
                inputdir = (
                    f"/fefs/aswg/data/real/DL1/{lstObsDir}/{LST_version}/tailcut84"
                )

                outputdir = f"{coincidence_DL1_dir}/DL1Coincident/{lstObsDir}"
                os.makedirs(f"{outputdir}/logs", exist_ok=True)

                list_of_subruns = np.sort(glob.glob(f"{inputdir}/dl1*Run*{i[1]}*.*.h5"))

                with open(f"{outputdir}/logs/list_LST.txt", "a+") as LSTdataPathFile:
                    for subrun in list_of_subruns:
                        LSTdataPathFile.write(f"{subrun}\n")

                if not os.path.exists(f"{outputdir}/logs/list_LST.txt"):
                    continue
                with open(f"{outputdir}/logs/list_LST.txt", "r") as f:
                    process_size = len(f.readlines()) - 1

                if process_size < 0:
                    continue
                slurm = slurm_lines(
                    queue="short",
                    job_name=f"{source_name}_coincidence",
                    nice_parameter=nice,
                    array=process_size,
                    mem="6g",
                    out_name=f"{outputdir}/logs/slurm-%x.%A_%a",
                )
                rc = rc_lines(
                    store="$SAMPLE ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID}",
                    out="$OUTPUTDIR/logs/list",
                )

                lines = (
                    slurm
                    + [
                        f"export INM={MAGIC_DL1_dir}/Merged/{d}\n",
                        f"export OUTPUTDIR={outputdir}\n",
                        "SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_LST.txt))\n",
                        "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
                        "export LOG=$OUTPUTDIR/logs/coincidence_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                        f"conda run -n {env_name} lst1_magic_event_coincidence --input-file-lst $SAMPLE --input-dir-magic $INM --output-dir $OUTPUTDIR --config-file {target_dir}/v{__version__}/{source_name}/config_coincidence.yaml >$LOG 2>&1\n",
                    ]
                    + rc
                )
                with open(
                    f"{source_name}_LST_coincident_{outputdir.split('/')[-1]}.sh",
                    "w",
                ) as f:
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
    LST_version = config["general"]["LST_version"]
    config_file = config["general"]["base_config_file"]
    nice_parameter = config["general"]["nice"] if "nice" in config["general"] else None

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

        print("***** Generating file config_coincidence.yaml...")
        configfile_coincidence(target_dir, source_name, config_file)

        LST_runs_and_dates = f"{source_name}_LST_runs.txt"
        LST_runs = np.genfromtxt(LST_runs_and_dates, dtype=str, delimiter=",", ndmin=2)

        try:

            print("***** Linking the paths to LST data files...")

            print("***** Generating the bashscript...")
            linking_bash_lst(
                target_dir,
                LST_runs,
                source_name,
                LST_version,
                env_name,
                cluster,
                in_version,
                nice_parameter,
            )  # linking the data paths to current working directory

            print("***** Submitting processess to the cluster...")
            print(f"Process name: {source_name}_coincidence")
            print(
                f"To check the jobs submitted to the cluster, type: squeue -n {source_name}_coincidence"
            )

            # Below we run the bash scripts to find the coincident events
            list_of_coincidence_scripts = np.sort(
                glob.glob(f"{source_name}_LST_coincident*.sh")
            )
            if len(list_of_coincidence_scripts) < 1:
                logger.warning("No bash scripts")
                continue
            launch_jobs = ""
            for n, run in enumerate(list_of_coincidence_scripts):
                launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"

            os.system(launch_jobs)

        except OSError as exc:
            logger.error(exc)


if __name__ == "__main__":
    main()
