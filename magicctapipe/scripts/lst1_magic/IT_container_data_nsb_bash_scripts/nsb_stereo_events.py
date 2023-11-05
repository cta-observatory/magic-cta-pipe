"""
This scripts generates and runs the bashscripts
to compute the stereo parameters of
Coincident MAGIC+LST data files.

Usage:
$ python stereo_events.py
"""
import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import yaml

from magicctapipe import __version__

__all__ = ["configfile_stereo", "bash_stereo"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configfile_stereo(ids, target_dir):

    """
    This function creates the configuration file needed for the event stereo step

    Parameters
    ----------
    ids : list
        List of telescope IDs
    target_dir : str
        Path to the working directory
    """

    lines = [
        f"mc_tel_ids:\n    LST-1: {ids[0]}\n    LST-2: {ids[1]}\n    LST-3: {ids[2]}\n    LST-4: {ids[3]}\n    MAGIC-I: {ids[4]}\n    MAGIC-II: {ids[5]}\n\n",
        'stereo_reco:\n    quality_cuts: "(intensity > 50) & (width > 0)"\n    theta_uplim: "6 arcmin"\n',
    ]

    with open(f"{target_dir}/config_stereo.yaml", "w") as f:
        f.writelines(lines)


def bash_stereo(target_dir, nsb, source, env_name):

    """
    This function generates the bashscript for running the stereo analysis.

    Parameters
    ----------
    target_dir : str
        Path to the working directory
    nsb : double
        NSB level in which the run has been classified
    source : str
        Target level
    env_name : str
        Name of the conda environment
    """

    process_name = target_dir.split("/")[-2:][1]

    if not os.path.exists(f"{target_dir}/v{__version__}/DL1CoincidentStereo"):
        os.mkdir(f"{target_dir}/v{__version__}/DL1CoincidentStereo")

    ST_list = [
        os.path.basename(x)
        for x in glob.glob(f"{target_dir}/v{__version__}/DL1Coincident/*")
    ]

    for p in ST_list:
        if not os.path.exists(f"{target_dir}/v{__version__}/DL1CoincidentStereo/{p}"):
            os.mkdir(f"{target_dir}/v{__version__}/DL1CoincidentStereo/{p}")

        if (
            not os.path.exists(
                f"{target_dir}/v{__version__}/DL1CoincidentStereo/{p}/NSB{nsb}"
            )
        ) and (
            os.path.exists(f"{target_dir}/v{__version__}/DL1Coincident/{p}/NSB{nsb}")
        ):
            os.mkdir(f"{target_dir}/v{__version__}/DL1CoincidentStereo/{p}/NSB{nsb}")
        listOfNightsLST = np.sort(
            glob.glob(f"{target_dir}/v{__version__}/DL1Coincident/{p}/NSB{nsb}/*")
        )
        for nightLST in listOfNightsLST:
            stereoDir = f'{target_dir}/v{__version__}/DL1CoincidentStereo/{p}/NSB{nsb}/{nightLST.split("/")[-1]}'
            if not os.path.exists(stereoDir):
                os.mkdir(stereoDir)
            if not os.path.exists(f"{stereoDir}/logs"):
                os.mkdir(f"{stereoDir}/logs")
            if not os.listdir(f"{nightLST}"):
                continue
            if len(os.listdir(nightLST)) < 2:
                continue
            os.system(
                f"ls {nightLST}/*LST*.h5 >  {stereoDir}/logs/list_coin_{nsb}.txt"
            )  # generating a list with the DL1 coincident data files.
            process_size = (
                len(np.genfromtxt(f"{stereoDir}/logs/list_coin_{nsb}.txt", dtype="str"))
                - 1
            )
            if process_size < 0:
                continue
            lines = [
                "#!/bin/sh\n\n",
                "#SBATCH -p short\n",
                f"#SBATCH -J {process_name}_stereo_{nsb}\n",
                f"#SBATCH --array=0-{process_size}\n",
                "#SBATCH -N 1\n\n",
                "ulimit -l unlimited\n",
                "ulimit -s unlimited\n",
                "ulimit -a\n\n",
                f"export INPUTDIR={nightLST}\n",
                f"export OUTPUTDIR={stereoDir}\n",
                f"SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_coin_{nsb}.txt))\n",
                "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
                "export LOG=$OUTPUTDIR/logs/stereo_${SLURM_ARRAY_TASK_ID}.log\n",
                f"time conda run -n {env_name} lst1_magic_stereo_reco --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_stereo.yaml >$LOG 2>&1",
            ]
            with open(
                f"{source}_StereoEvents_{nsb}_{nightLST.split('/')[-1]}.sh", "w"
            ) as f:
                f.writelines(lines)


def main():

    """
    Here we read the config file and call the functions defined above.
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
    telescope_ids = list(config["mc_tel_ids"].values())
    source = config["directories"]["target_name"]
    print("***** Generating file config_stereo.yaml...")
    print("***** This file can be found in ", target_dir)
    configfile_stereo(telescope_ids, target_dir)
    listnsb = np.sort(glob.glob(f"{source}_LST_*_.txt"))
    nsb = []
    for f in listnsb:
        nsb.append(f.split("_")[2])

    for nsblvl in nsb:
        print("***** Generating the bashscript...")
        bash_stereo(target_dir, nsblvl, source, env_name)

        print("***** Submitting processess to the cluster...")
        print(f'Process name: {target_dir.split("/")[-2:][1]}_stereo_{nsblvl}')
        print(
            f'To check the jobs submitted to the cluster, type: squeue -n {target_dir.split("/")[-2:][1]}_stereo_{nsblvl}'
        )

        # Below we run the bash scripts to find the stereo events
        list_of_stereo_scripts = np.sort(
            glob.glob(f"{source}_StereoEvents_{nsblvl}*.sh")
        )
        if len(list_of_stereo_scripts) < 1:
            continue
        for n, run in enumerate(list_of_stereo_scripts):
            if n == 0:
                launch_jobs = f"stereo{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = f"{launch_jobs} && stereo{n}=$(sbatch --parsable {run})"

        os.system(launch_jobs)


if __name__ == "__main__":
    main()
