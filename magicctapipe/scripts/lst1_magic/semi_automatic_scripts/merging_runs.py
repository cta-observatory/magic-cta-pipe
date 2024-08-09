"""
This script generates the bash
scripts to merge real data files by calling the script "merge_hdf_files.py":

MAGIC:

Merge the subruns into runs for M1 and M2 individually.

Usage:
$ merging_runs (-c config.yaml)
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
from magicctapipe.scripts.lst1_magic.semi_automatic_scripts.clusters import (
    rc_lines,
    slurm_lines,
)

__all__ = ["merge"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def merge(target_dir, MAGIC_runs, env_name, source, cluster):

    """
    This function creates the bash scripts to run merge_hdf_files.py for real data

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    MAGIC_runs : matrix of strings
        Matrix of [(date,run)] n-tuples
    env_name : str
        Name of the environment
    source : str
        Target name
    cluster : str
        Cluster system
    """

    process_name = f"merging_{source}"

    MAGIC_DL1_dir = f"{target_dir}/v{__version__}/{source}/DL1/"

    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return
    lines = slurm_lines(
        queue="short",
        job_name=process_name,
        mem="2g",
        out_name=f"{MAGIC_DL1_dir}/Merged/logs/slurm-%x.%j",
    )
    os.makedirs(f"{MAGIC_DL1_dir}/Merged/logs", exist_ok=True)

    with open(f"{source}_Merge_MAGIC.sh", "w") as f:
        f.writelines(lines)
        for magic in [1, 2]:
            for i in MAGIC_runs:
                # Here is a difference w.r.t. original code. If only one telescope data are available they will be merged now for this telescope
                indir = f"{MAGIC_DL1_dir}/M{magic}/{i[0]}/{i[1]}"
                if os.path.exists(f"{indir}"):
                    outdir = f"{MAGIC_DL1_dir}/Merged/{i[0]}"
                    os.makedirs(f"{outdir}/logs", exist_ok=True)

                    f.write(
                        f"conda run -n {env_name} merge_hdf_files --input-dir {indir} --output-dir {outdir} >{outdir}/logs/merge_M{magic}_{i[0]}_{i[1]}_${{SLURM_JOB_ID}}.log 2>&1\n"
                    )
                    rc = rc_lines(
                        store=f"{indir} ${{SLURM_JOB_ID}}",
                        out=f"{outdir}/logs/list",
                    )
                    f.writelines(rc)
                    os.system(f"echo {indir} >> {outdir}/logs/list_dl0.txt")
                else:
                    logger.error(f"{indir} does not exist")


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
    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    cluster = config["general"]["cluster"]

    source_list = []
    if source_in is None:
        source_list = joblib.load("list_sources.dat")

    else:
        source_list.append(source)

    for source_name in source_list:
        MAGIC_runs_and_dates = f"{source_name}_MAGIC_runs.txt"
        MAGIC_runs = np.genfromtxt(
            MAGIC_runs_and_dates, dtype=str, delimiter=",", ndmin=2
        )

        # Below we run the analysis on the MAGIC data

        print("***** Generating merge_MAGIC bashscripts...")
        merge(
            target_dir,
            MAGIC_runs,
            env_name,
            source_name,
            cluster,
        )  # generating the bash script to merge the subruns

        print("***** Running merge_hdf_files.py on the MAGIC data files...")

        # Below we run the bash scripts to merge the MAGIC files
        list_of_merging_scripts = np.sort(glob.glob(f"{source_name}_Merge_MAGIC*.sh"))
        if len(list_of_merging_scripts) < 1:
            logger.warning("No bash scripts for real data")
            continue
        launch_jobs = ""
        for n, run in enumerate(list_of_merging_scripts):
            launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"

        os.system(launch_jobs)

        print(f"Process name: merging_{source_name}")
        print(
            f"To check the jobs submitted to the cluster, type: squeue -n merging_{source_name}"
        )


if __name__ == "__main__":
    main()
