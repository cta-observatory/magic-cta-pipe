"""
This script creates the bashscripts necessary to apply "lst1_magic_dl2_to_dl3.py"
to the DL2. It also creates new subdirectories associated with
the data level 3.


Usage:
$ python new_DL2_to_DL3.py -c configuration_file.yaml

"""
import argparse
import os
import numpy as np
import glob
import yaml
import logging
import joblib
from pathlib import Path
from magicctapipe import __version__
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configuration_DL3(ids, target_dir, target_coords,source):
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

    f = open(str(target_dir) + "/config_DL3.yaml", "w")
    f.write(
        "mc_tel_ids:\n    LST-1: "
        + str(ids[0])
        + "\n    LST-2: "
        + str(ids[1])
        + "\n    LST-3: "
        + str(ids[2])
        + "\n    LST-4: "
        + str(ids[3])
        + "\n    MAGIC-I: "
        + str(ids[4])
        + "\n    MAGIC-II: "
        + str(ids[5])
        + "\n\n"
    )
    f.write(
        f'dl2_to_dl3:\n    interpolation_method: "linear"  # select "nearest", "linear" or "cubic"\n    interpolation_scheme: "cosZd" # select "cosZdAz" or "cosZd"\n    max_distance: "90. deg"\n    source_name: "{source}"\n    source_ra: "{target_coords[0]} deg" # used when the source name cannot be resolved\n    source_dec: "{target_coords[1]} deg" # used when the source name cannot be resolved\n\n'
    )

    f.close()


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
    if not os.path.exists(target_dir + f"/v{__version__}/{source}/DL3"):
        os.mkdir(target_dir + f"/v{__version__}/{source}/DL3")
    if not os.path.exists(target_dir + f"/v{__version__}/{source}/DL3/logs"):
        os.mkdir(target_dir  + f"/v{__version__}/{source}/DL3/logs")

    #Loop over all nights
    Nights_list = np.sort(glob.glob(f"{target_dir}/v{__version__}/{source}/DL2/*"))
    for night in Nights_list:
        #Loop over every run:
        File_list = np.sort(glob.glob(f"{night}/*.txt"))
        for file in File_list:
            with open(file, "r") as f: 
                runs = f.readlines()
                process_size = len(runs) - 1
                run_new = [run.split("DL1Stereo")[0] + "DL2" + run.split("DL1Stereo")[1].replace("dl1_stereo", "dl2").replace("/Merged", "") for run in runs]
                with open(file, "w") as g:
                    g.writelines(run_new)

            nsb = file.split("/")[-1].split("_")[-1][:3]
            print("nsb = ", nsb)
            period = file.split("/")[-1].split("_")[0]
            print("period = ", period)

            IRF_dir = (
                f"/fefs/aswg/LST1MAGIC/mc/IRF/{period}/NSB{nsb}/GammaTest/v01.2/g_dyn_0.9_th_glo_0.2/dec_2276/"
            )

            process_name = "DL3_" + target_dir.split("/")[-2:][1] + str(nsb)
            output = target_dir  + f"/v{__version__}/{source}/DL3"

            f = open(f'{source}_DL3_{nsb}_{period}_{night.split("/")[-1]}.sh', "w")
            f.write("#!/bin/sh\n\n")
            f.write("#SBATCH -p short\n")
            f.write("#SBATCH -J " + process_name + "\n")
            f.write("#SBATCH --mem=10g\n")
            f.write(f"#SBATCH --array=0-{process_size}%100\n")
            f.write("#SBATCH -N 1\n\n")
            f.write("ulimit -l unlimited\n")
            f.write("ulimit -s unlimited\n")
            f.write("ulimit -a\n\n")

            f.write(
                    f"SAMPLE_LIST=($(<{file}))\n"
                )
            f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")

            f.write(f'export LOG={output}/logs/DL3_{nsb}_{period}_{night.split("/")[-1]}.log\n')
            f.write(
                f"conda run -n {env_name} lst1_magic_dl2_to_dl3 --input-file-dl2 $SAMPLE --input-dir-irf {IRF_dir} --output-dir {output} --config-file {target_dir}/config_DL3.yaml >$LOG 2>&1\n\n"
            )

            f.close()


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

    target_coords = [
        config["data_selection"]["target_RA_deg"],
        config["data_selection"]["target_Dec_deg"],
    ]

    print("***** Generating file config_DL3.yaml...")
    print("***** This file can be found in ", target_dir)

    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    env_name = config["general"]["env_name"]
    cluster = config["general"]["cluster"]

    #cp the .txt files from DL1 stereo anaysis to be used again.
    DL1stereo_Nihts = np.sort(glob.glob(f"{target_dir}/v{__version__}/{source}/DL1Stereo/*"))
    for night in DL1stereo_Nihts:
        File_list = glob.glob(f"{night}/Merged/logs/ST*.txt")
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
        list_of_stereo_scripts = np.sort(glob.glob(f'{source_name}_DL3*.sh'))
        print(list_of_stereo_scripts)
        for n, run in enumerate(list_of_stereo_scripts):
            if n == 0:
                launch_jobs = f"stereo{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = (
                    f"{launch_jobs} && stereo{n}=$(sbatch --parsable {run})"
                )
        print(launch_jobs)
        os.system(launch_jobs)

if __name__ == "__main__":
    main()
