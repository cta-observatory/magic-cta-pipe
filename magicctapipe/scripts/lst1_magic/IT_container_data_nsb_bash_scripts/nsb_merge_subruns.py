"""
This script generates the bash scripts to merge the data
files calling the script "merge_hdf_files.py" to
merge the subruns into runs for M1 and M2 individually.

Usage:
$ python merge_subruns.py
(-c config_file.yaml)
"""
import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import yaml

from magicctapipe import __version__

__all__ = ["merge1"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def merge1(target_dir, source, env_name):

    """
    This function creates the bash scripts to run merge_hdf_files.py

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    source : str
        Target name
    env_name : str
        Name of the conda environment
    """

    ST_list = [
        os.path.basename(x) for x in glob.glob(f"{target_dir}/v{__version__}/DL1/*")
    ]

    for p in ST_list:
        process_name = f'merging_{target_dir.split("/")[-2:][1]}'

        MAGIC_DL1_dir = f"{target_dir}/v{__version__}/DL1/{p}"

        if os.path.exists(f"{MAGIC_DL1_dir}/M1") & os.path.exists(
            f"{MAGIC_DL1_dir}/M2"
        ):
            if not os.path.exists(f"{MAGIC_DL1_dir}/Merged"):
                os.mkdir(f"{MAGIC_DL1_dir}/Merged")
        lines = [
            "#!/bin/sh\n\n",
            "#SBATCH -p short\n",
            f"#SBATCH -J {process_name}\n",
            "#SBATCH -N 1\n\n",
            "ulimit -l unlimited\n",
            "ulimit -s unlimited\n",
            "ulimit -a\n\n",
        ]
        with open(f"{source}_Merge_0_{p}.sh", "w") as f:
            f.writelines(lines)
            if os.path.exists(f"{MAGIC_DL1_dir}/M1"):
                dates = [
                    os.path.basename(x) for x in glob.glob(f"{MAGIC_DL1_dir}/M1/*")
                ]
                for i in dates:
                    runs = [
                        os.path.basename(x)
                        for x in glob.glob(f"{MAGIC_DL1_dir}/M1/{i}/*")
                    ]
                    if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i}"):
                        os.mkdir(
                            f"{MAGIC_DL1_dir}/Merged/{i}"
                        )  # Creating a merged directory for the respective night
                    for r in runs:
                        if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i}/{r}"):
                            os.mkdir(
                                f"{MAGIC_DL1_dir}/Merged/{i}/{r}"
                            )  # Creating a merged directory for the respective run
                        if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i}/{r}/logs"):
                            os.mkdir(
                                f"{MAGIC_DL1_dir}/Merged/{i}/{r}/logs"
                            )  # Creating a merged directory for the respective run

                        f.write(
                            f"time conda run -n {env_name} merge_hdf_files --input-dir {MAGIC_DL1_dir}/M1/{i}/{r} --output-dir {MAGIC_DL1_dir}/Merged/{i}/{r} >{MAGIC_DL1_dir}/Merged/{i}/{r}/logs/merge_M1_{i}_{r}.log \n"
                        )

            if os.path.exists(f"{MAGIC_DL1_dir}/M2"):
                dates = [
                    os.path.basename(x) for x in glob.glob(f"{MAGIC_DL1_dir}/M2/*")
                ]

                for i in dates:
                    runs = [
                        os.path.basename(x)
                        for x in glob.glob(f"{MAGIC_DL1_dir}/M2/{i}/*")
                    ]
                    if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i}"):
                        os.mkdir(
                            f"{MAGIC_DL1_dir}/Merged/{i}"
                        )  # Creating a merged directory for the respective night
                    for r in runs:
                        if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i}/{r}"):
                            os.mkdir(
                                f"{MAGIC_DL1_dir}/Merged/{i}/{r}"
                            )  # Creating a merged directory for the respective run
                        if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i}/{r}/logs"):
                            os.mkdir(
                                f"{MAGIC_DL1_dir}/Merged/{i}/{r}/logs"
                            )  # Creating a merged directory for the respective run

                        f.write(
                            f"time conda run -n {env_name} merge_hdf_files --input-dir {MAGIC_DL1_dir}/M2/{i}/{r} --output-dir {MAGIC_DL1_dir}/Merged/{i}/{r} >{MAGIC_DL1_dir}/Merged/{i}/{r}/logs/merge_M2_{i}_{r}.log \n"
                        )


def main():

    """
    Here we read the config_general.yaml file, split the pronton sample into "test" and "train", and merge the MAGIC files.
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
    source = config["directories"]["target_name"]

    print("***** Generating merge bashscripts...")
    merge1(
        target_dir, source, env_name
    )  # generating the bash script to merge the subruns

    print("***** Running merge_hdf_files.py in the MAGIC data files...")
    print(f'Process name: merging_{target_dir.split("/")[-2:][1]}')
    print(
        f'To check the jobs submitted to the cluster, type: squeue -n merging_{target_dir.split("/")[-2:][1]}'
    )

    # Below we run the bash scripts to merge the MAGIC files
    list_of_merging_scripts = np.sort(glob.glob(f"{source}_Merge_0_*.sh"))
    if len(list_of_merging_scripts) < 1:
        return
    for n, run in enumerate(list_of_merging_scripts):
        if n == 0:
            launch_jobs = f"merging{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = f"{launch_jobs} && merging{n}=$(sbatch --parsable {run})"

    # print(launch_jobs)
    os.system(launch_jobs)


if __name__ == "__main__":
    main()
