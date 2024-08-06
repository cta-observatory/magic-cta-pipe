"""
This script facilitates the usage of
"magic_calib_to_dl1.py". This script is more like a
"manager" that organizes the analysis process by:
1) Creating the necessary directories and subdirectories.
2) Generating all the bash script files that convert the
MAGIC files from Calibrated (`_Y_`) to DL1.
3) Launching these jobs in the IT container.

Notice that in this stage we only use MAGIC data.
No LST data is used here.

Standard usage:
$ dl1_production (-t analysis_type) (-c config_file.yaml)
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

__all__ = [
    "config_file_gen",
    "lists_and_bash_gen_MAGIC",
    "directories_generator_real",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def config_file_gen(target_dir, noise_value, source_name, config_gen):

    """
    Here we create the configuration file needed for transforming DL0 into DL1

    Parameters
    ----------
    target_dir : path
        Directory to store the results
    noise_value : list
        List of the noise correction values for LST
    source_name : str
        Name of the target source
    config_gen : dict
        Dictionary of the entries of the general configuration file
    """
    config_file = config_gen["general"]["base_config_file"]
    if config_file == "":
        config_file = resource_file("config.yaml")
    with open(
        config_file, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)
    LST_config = config_dict["LST"]
    MAGIC_config = config_dict["MAGIC"]

    
    conf = {
        "mc_tel_ids": config_gen["mc_tel_ids"],
        "LST": LST_config,
        "MAGIC": MAGIC_config,
    }
    
    file_name = f"{target_dir}/v{__version__}/{source_name}/config_DL0_to_DL1.yaml"
    with open(file_name, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)



def lists_and_bash_gen_MAGIC(
    target_dir, telescope_ids, MAGIC_runs, source, env_name, cluster
):

    """
    Below we create bash scripts that link the MAGIC data paths to each subdirectory and process them from Calibrated to Dl1

    Parameters
    ----------
    target_dir : str
        Directory to store the results
    telescope_ids : list
        List of the telescope IDs (set by the user)
    MAGIC_runs : array
        MAGIC dates and runs to be processed
    source : str
        Name of the target
    env_name : str
        Name of the environment
    cluster : str
        Cluster system
    """
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return
    process_name = source
    lines = slurm_lines(
        queue="short",
        job_name=process_name,
        out_name=f"{target_dir}/v{__version__}/{source}/DL1/slurm-linkMAGIC-%x.%j",
    )

    with open(f"{source}_linking_MAGIC_data_paths.sh", "w") as f:
        f.writelines(lines)
        for i in MAGIC_runs:
            for magic in [1, 2]:
                # if 1 then magic is second from last, if 2 then last
                if telescope_ids[magic - 3] > 0:
                    lines = [
                        f'export IN1=/fefs/onsite/common/MAGIC/data/M{magic}/event/Calibrated/{i[0].replace("_","/")}\n',
                        f"export OUT1={target_dir}/v{__version__}/{source}/DL1/M{magic}/{i[0]}/{i[1]}/logs \n",
                        f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_cal.txt\n\n",
                    ]
                    f.writelines(lines)

    for magic in [1, 2]:
        # if 1 then magic is second from last, if 2 then last
        if telescope_ids[magic - 3] > 0:
            for i in MAGIC_runs:
                number_of_nodes = glob.glob(
                    f'/fefs/onsite/common/MAGIC/data/M{magic}/event/Calibrated/{i[0].replace("_","/")}/*{i[1]}.*_Y_*.root'
                )
                number_of_nodes = len(number_of_nodes) - 1
                if number_of_nodes < 0:
                    continue
                slurm = slurm_lines(
                    queue="short",
                    job_name=process_name,
                    array=number_of_nodes,
                    mem="2g",
                    out_name=f"{target_dir}/v{__version__}/{source}/DL1/M{magic}/{i[0]}/{i[1]}/logs/slurm-%x.%A_%a",
                )
                rc = rc_lines(
                    store="$SAMPLE ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID}",
                    out="$OUTPUTDIR/logs/list",
                )
                lines = (
                    slurm
                    + [
                        f"export OUTPUTDIR={target_dir}/v{__version__}/{source}/DL1/M{magic}/{i[0]}/{i[1]}\n",
                        "SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_cal.txt))\n",
                        "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                        "export LOG=$OUTPUTDIR/logs/real_0_1_task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                        f"conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/v{__version__}/{source}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
                    ]
                    + rc
                )
                with open(
                    f"{source}_MAGIC-" + "I" * magic + f"_cal_to_dl1_run_{i[1]}.sh",
                    "w",
                ) as f:
                    f.writelines(lines)


def directories_generator_real(
    target_dir, telescope_ids, MAGIC_runs, source_name
):
    """
    Here we create all subdirectories for a given workspace and target name.

    Parameters
    ----------
    target_dir : str
        Directory to store the results
    telescope_ids : list
        List of the telescope IDs (set by the user)
    MAGIC_runs : array
        MAGIC dates and runs to be processed
    source_name : str
        Name of the target source
    """

    
    os.makedirs(f"{target_dir}/v{__version__}/{source_name}/DL1", exist_ok=True)
    dl1_dir = str(f"{target_dir}/v{__version__}/{source_name}/DL1")
    
    ###########################################
    # MAGIC
    ###########################################
    for i in MAGIC_runs:
        for magic in [1, 2]:
            if telescope_ids[magic - 3] > 0:
                os.makedirs(f"{dl1_dir}/M{magic}/{i[0]}/{i[1]}/logs", exist_ok=True)




def main():

    """
    Main function
    """

    # Here we are simply collecting the parameters from the command line, as input file, output directory, and configuration file

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

    telescope_ids = list(config["mc_tel_ids"].values())
    env_name = config["general"]["env_name"]

    
    focal_length = config["general"]["focal_length"]
    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    cluster = config["general"]["cluster"]
    target_dir = Path(config["directories"]["workspace_dir"])
    source_list = []
    if source_in is None:
        source_list = joblib.load("list_sources.dat")

    else:
        source_list.append(source)
    noise_value = [0, 0, 0]
    
    

    for source_name in source_list:
        

        MAGIC_runs_and_dates = f"{source_name}_MAGIC_runs.txt"
        MAGIC_runs = np.genfromtxt(
            MAGIC_runs_and_dates, dtype=str, delimiter=",", ndmin=2
        )  # READ LIST OF DATES AND RUNS: format table where each line is like "2020_11_19,5093174"

        # TODO: fix here above
        print("*** Converting Calibrated into DL1 data ***")
        print(f"Process name: {source_name}")
        print(
            f"To check the jobs submitted to the cluster, type: squeue -n {source_name}"
        )

        directories_generator_real(
            str(target_dir), telescope_ids, MAGIC_runs, source_name
        )  # Here we create all the necessary directories in the given workspace and collect the main directory of the target
        config_file_gen(
            target_dir, noise_value, source_name, config
        )  # TODO: fix here

        # Below we run the analysis on the MAGIC data

        lists_and_bash_gen_MAGIC(
            target_dir,
            telescope_ids,
            MAGIC_runs,
            source_name,
            env_name,
            cluster,
        )  # MAGIC real data
        if (telescope_ids[-2] > 0) or (telescope_ids[-1] > 0):
            list_of_MAGIC_runs = glob.glob(f"{source_name}_MAGIC-*.sh")
            if len(list_of_MAGIC_runs) < 1:
                logger.warning(
                    "No bash script has been produced. Please check the provided MAGIC_runs.txt and the MAGIC calibrated data"
                )
                continue

            launch_jobs = f"linking=$(sbatch --parsable {source_name}_linking_MAGIC_data_paths.sh)"
            for n, run in enumerate(list_of_MAGIC_runs):
                launch_jobs = f"{launch_jobs} && RES{n}=$(sbatch --parsable --dependency=afterany:$linking {run})"

            os.system(launch_jobs)


if __name__ == "__main__":
    main()
