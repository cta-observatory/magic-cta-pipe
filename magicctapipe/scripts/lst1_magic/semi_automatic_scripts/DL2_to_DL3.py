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
import yaml

from magicctapipe import __version__
from magicctapipe.io import resource_file

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configuration_DL3(target_dir, source_name, config_file):
    """
    This function creates the configuration file needed for the DL2 to DL3 conversion

    Parameters
    ----------
    ids: list
        list of telescope IDs
    target_dir: str
        Path to the working directory
    target_coords:
        sorce coordinates
    source: str
        source name
    """

    if config_file == "":
        config_file = resource_file("config.yaml")

    with open(
        config_file, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)
    DL3_config = config_dict["dl2_to_dl3"]
    DL3_config["source_name"] = source_name
    # DL3_config['source_ra']=ra
    # DL3_config['source_dec']=dec
    conf = {
        "mc_tel_ids": config_dict["mc_tel_ids"],
        "dl2_to_dl3": DL3_config,
    }

    conf_dir = f"{target_dir}/v{__version__}/{source_name}"
    os.makedirs(conf_dir, exist_ok=True)

    file_name = f"{conf_dir}/config_DL3.yaml"

    with open(file_name, "w") as f:

        yaml.dump(conf, f, default_flow_style=False)


def DL2_to_DL3(target_dir, source, env_name):
    """
    This function creates the bash scripts to run lst1_magic_dl2_to_dl3.py on the real data.

    Parameters
    ----------
    target_dir: str
        Path to the working directory
    source: str
        source name
    env_name: str
        conda enviroment name
    """

    target_dir = str(target_dir)

    os.makedirs(target_dir + f"/v{__version__}/{source}/DL3/logs", exist_ok=True)

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
            print("nsb = ", nsb)
            period = file.split("/")[-1].split("_")[0]
            print("period = ", period)

            IRF_dir = f"/fefs/aswg/LST1MAGIC/mc/IRF/{period}/NSB{nsb}/GammaTest/v01.2/g_dyn_0.9_th_glo_0.2/dec_2276/"

            process_name = source
            output = target_dir + f"/v{__version__}/{source}/DL3"

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
                    f"conda run -n {env_name} lst1_magic_dl2_to_dl3 --input-file-dl2 $SAMPLE --input-dir-irf {IRF_dir} --output-dir {output} --config-file {target_dir}/config_DL3.yaml >$LOG 2>&1\n\n",
                ]
                + rc
            )
            with open(
                f'{source}_DL3_{nsb}_{period}_{night.split("/")[-1]}.sh', "w"
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

    telescope_ids = list(config["mc_tel_ids"].values())
    target_dir = Path(config["directories"]["workspace_dir"])

    print("***** Generating file config_DL3.yaml...")
    print("***** This file can be found in ", target_dir)

    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    env_name = config["general"]["env_name"]
    config_file = config["general"]["base_config_file"]
    cluster = config["general"]["cluster"]

    # cp the .txt files from DL1 stereo anaysis to be used again.
    DL1stereo_Nihts = np.sort(
        glob.glob(f"{target_dir}/v{__version__}/{source}/DL1Stereo/Merged/*")
    )
    for night in DL1stereo_Nihts:
        File_list = glob.glob(f"{night}/logs/ST*.txt")
        night_date = night.split("/")[-1]
        print("night date ", night_date)
        for file in File_list:
            cp_dir = f"{target_dir}/v{__version__}/{source}/DL2/{night_date}"
            os.system(f"cp {file} {cp_dir}")

    if source_in is None:
        source_list = joblib.load("list_sources.dat")
    else:
        source_list = [source]
    for source_name in source_list:
        configuration_DL3(telescope_ids, target_dir, target_coords, source_name)
        DL2_to_DL3(target_dir, source_name, env_name)
        list_of_stereo_scripts = np.sort(glob.glob(f"{source_name}_DL3*.sh"))
        print(list_of_stereo_scripts)
        if len(list_of_stereo_scripts) < 1:
            logger.warning("No bash scripts for real data")
            continue
        launch_jobs = ""
        for n, run in enumerate(list_of_stereo_scripts):
            launch_jobs += (" && " if n > 0 else "") + f"sbatch {run}"
        os.system(launch_jobs)


if __name__ == "__main__":
    main()
