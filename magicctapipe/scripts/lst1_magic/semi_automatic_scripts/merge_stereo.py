"""
This scripts merges LST DL1Stereo subruns into runs

Usage:
$ merge_stereo (-c config_file.yaml)
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

__all__ = ["MergeStereo"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def MergeStereo(target_dir, env_name, source, cluster, version):
    """
    This function creates the bash scripts to run merge_hdf_files.py in all DL1Stereo subruns.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    env_name : str
        Name of the environment
    source : str
        Name of the target
    cluster : str
        Cluster system
    version : str
        Version of the input (stereo subruns) data
    """

    process_name = source
    stereo_DL1_dir = f"{target_dir}/v{version}/{source}"
    listOfNightsLST = np.sort(glob.glob(f"{stereo_DL1_dir}/DL1Stereo/*"))
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return
    for nightLST in listOfNightsLST:
        night = nightLST.split("/")[-1]
        stereoMergeDir = (
            f"{target_dir}/v{__version__}/{source}/DL1Stereo/Merged/{night}"
        )
        os.makedirs(f"{stereoMergeDir}/logs", exist_ok=True)

        if len(glob.glob(f"{nightLST}/dl1_stereo*.h5")) < 1:
            continue

        slurm = slurm_lines(
            queue="short",
            job_name=f"{process_name}_stereo_merge",
            mem="2g",
            out_name=f"{stereoMergeDir}/logs/slurm-%x.%A_%a",
        )
        rc = rc_lines(
            store=f"{nightLST} ${{SLURM_JOB_ID}}", out=f"{stereoMergeDir}/logs/list"
        )
        os.system(f"echo {nightLST} >> {stereoMergeDir}/logs/list_dl0.txt")
        lines = (
            slurm
            + [
                f"conda run -n {env_name} merge_hdf_files --input-dir {nightLST} --output-dir {stereoMergeDir} --run-wise >{stereoMergeDir}/logs/merge_{night}_${{SLURM_JOB_ID}}.log 2>&1\n"
            ]
            + rc
        )

        with open(f"{source}_StereoMerge_{night}.sh", "w") as f:
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

    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    cluster = config["general"]["cluster"]
    in_version = config["directories"]["real_input_version"]
    if in_version == "":
        in_version == __version__

    if source_in is None:
        source_list = joblib.load("list_sources.dat")

    else:
        if source is None:
            source = source_in
        source_list = [source]

    for source_name in source_list:

        print("***** Merging DL1Stereo files run-wise...")
        MergeStereo(target_dir, env_name, source_name, cluster, in_version)

        list_of_merge = glob.glob(f"{source_name}_StereoMerge_*.sh")
        if len(list_of_merge) < 1:
            print("Warning: no bash script has been produced")
            continue

        launch_jobs = ""

        for n, run in enumerate(list_of_merge):
            launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"

        os.system(launch_jobs)


if __name__ == "__main__":
    main()
