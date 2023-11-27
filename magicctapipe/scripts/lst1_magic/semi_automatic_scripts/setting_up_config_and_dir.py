"""
This script facilitates the usage of
"magic_calib_to_dl1.py". This script is more like a
"manager" that organizes the analysis process by:
1) Creating the necessary directories and subdirectories.
2) Generating all the bash script files that convert the
MAGIC files from DL0 to DL1.
3) Launching these jobs in the IT container.

Notice that in this stage we only use MAGIC data.
No LST data is used here.

Standard usage:
$ python setting_up_config_and_dir.py (-c config_file.yaml)
"""
import argparse
import glob
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml

from magicctapipe import __version__

__all__ = [
    "nsb_avg",
    "collect_nsb",
    "config_file_gen",
    "lists_and_bash_generator",
    "lists_and_bash_gen_MAGIC",
    "directories_generator",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

ST_list = ["ST0320A", "ST0319A", "ST0318A", "ST0317A", "ST0316A"]
ST_begin = ["2023_03_10", "2022_12_15", "2022_06_10", "2021_12_30", "2020_10_24"]
ST_end = [
    "2024_01_01",
    "2023_03_09",
    "2022_08_31",
    "2022_06_09",
    "2021_09_29",
]  # ST0320 ongoing -> 'service' end date


def nsb_avg(source, config, LST_list):

    """
    This function evaluates the average of the NSB levels that have been evaluated by LSTnsb_MC.py (one value per run).

    Parameters
    ----------
    source : str
        Source name
    config : str
        Config file
    LST_list : str
        Name of the file where the adopted LST runs are listed

    Returns
    -------
    continue_process : string
        If 'y', data processing will continue, otherwise it will be stopped
    nsb : double
        NSB value (average over the runs)
    """
    allfile = np.sort(
        glob.glob(f"{source}_LST_nsb_*.txt")
    )  # List with the names of all files containing the NSB values for each run
    if len(allfile) == 0:
        print(
            "Warning: no file (containing the NSB value) exists for any of the LST runs to be processed. Check the input list"
        )
        return
    noise = []
    for j in allfile:
        with open(j) as ff:
            line_str = ff.readline().rstrip("\n")
            line = float(line_str)
            noise.append(line)
    nsb = np.average(noise)
    std = np.std(noise)
    continue_process = "y"
    if std > 0.2:
        continue_process = input(
            f'The standard deviation of the NSB levels is {std}. We recommend using NSB-matching scripts always that the standard deviation of NSB is > 0.2. Would you like to continue the current analysis anyway? [only "y" or "n"]: '
        )
    delete_index = []
    for n, j in enumerate(allfile):
        run = j.split("_")[3].rstrip(".txt")
        if abs(noise[n] - nsb) > 3 * std:
            sigma_range = input(
                f'Run {run} has an NSB value of {noise[n]}, which is more than 3*sigma (i.e. {3*std}) away from the average (i.e. {nsb}). Would you like to continue the current analysis anyway? [only "y" or "n"]: '
            )
            if sigma_range != "y":
                return (sigma_range, 0)

            sigma_range = input(
                f'Would you like to keep this run (i.e. {run}) in the analysis? [only "y" or "n"]:'
            )
            if sigma_range != "y":
                delete_index.append(n)
                with open(LST_list, "r") as f:
                    lines = f.readlines()
                with open(LST_list, "w") as f:
                    for i in lines:
                        if not i.endswith(f"{run}\n"):
                            f.write(i)

    if len(delete_index) > 0:
        index = (
            delete_index.reverse()
        )  # Here we reverse the list of indexes associated with out-of-the-average NSB values, such that after deleting one element (below), the indexes of the array do not change.
        for k in index:
            np.delete(noise, k)

    nsb = np.average(noise)
    with open(config, "r") as f:
        lines = f.readlines()
    with open(config, "w") as f:
        for i in lines:
            if not i.startswith("nsb_value"):
                f.write(i)
        f.write(f"nsb_value: {nsb}\n")
    return (continue_process, nsb)


def collect_nsb(config):
    """
    Here we split the LST runs in NSB-wise .txt files

    Parameters
    ----------
    config : dict
        Configuration file
    """
    source = config["directories"]["target_name"]

    nsb = config["general"]["nsb"]
    for nsblvl in nsb:
        allfile = np.sort(glob.glob(f"{source}_LST_{nsblvl}_*.txt"))
        if len(allfile) == 0:
            continue
        for j in allfile:
            with open(j) as ff:
                line = ff.readline()
                with open(f"{source}_LST_{nsblvl}_.txt", "a+") as f:
                    f.write(f"{line.rstrip()}\n")


def config_file_gen(ids, target_dir, noise_value, NSB_match):

    """
    Here we create the configuration file needed for transforming DL0 into DL1

    Parameters
    ----------
    ids : list
        Telescope IDs
    target_dir : path
        Directory to store the results
    noise_value : list
        List of the noise correction values for LST
    NSB_match : bool
        If real data are matched to pre-processed MCs or not
    """

    """
    Here we create the configuration file needed for transforming DL0 into DL1

    Parameters
    ----------
    ids : list
        Telescope IDs
    target_dir : path
        Directory to store the results
    noise_value : list
        Extra noise in dim and bright pixels, Extra bias in dim pixels
    """
    config_file = "../config.yaml"
    with open(
        config_file, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)
    LST_config = config_dict["LST"]
    MAGIC_config = config_dict["MAGIC"]

    if not NSB_match:
        LST_config["increase_nsb"]["extra_noise_in_dim_pixels"] = {noise_value[0]}
        LST_config["increase_nsb"]["extra_bias_in_dim_pixels"] = {noise_value[2]}
        LST_config["increase_nsb"]["extra_noise_in_bright_pixels"] = {noise_value[1]}
    conf = {}
    conf["LST"] = LST_config

    conf["MAGIC"] = MAGIC_config

    with open(f"{target_dir}/config_DL0_to_DL1.yaml", "w") as f:
        lines = [
            "mc_tel_ids:",
            f"\n    LST-1: {ids[0]}",
            f"\n    LST-2: {ids[1]}",
            f"\n    LST-3: {ids[2]}",
            f"\n    LST-4: {ids[3]}",
            f"\n    MAGIC-I: {ids[4]}",
            f"\n    MAGIC-II: {ids[5]}",
            "\n",
        ]
        f.writelines(lines)
        yaml.dump(conf, f, default_flow_style=False)


def lists_and_bash_generator(
    particle_type, target_dir, MC_path, SimTel_version, focal_length, env_name
):

    """
    This function creates the lists list_nodes_gamma_complete.txt and list_folder_gamma.txt with the MC file paths.
    After that, it generates a few bash scripts to link the MC paths to each subdirectory.
    These bash scripts will be called later in the main() function below. This step will be skipped in case the MC path has not been provided (MC_path='')

    Parameters
    ----------
    particle_type : str
        Particle type (e.g., protons)
    target_dir : str
        Directory to store the results
    MC_path : str
        Path to the MCs DL0s
    SimTel_version : str
        Version of SimTel (used to produce MCs)
    focal_length : str
        Focal length to be used to process MCs (e.g., 'nominal')
    env_name : str
        Name of the environment
    """

    if MC_path == "":
        return

    process_name = target_dir.split("/")[-2:][1]

    list_of_nodes = glob.glob(f"{MC_path}/node*")
    with open(
        f"{target_dir}/list_nodes_{particle_type}_complete.txt", "w"
    ) as f:  # creating list_nodes_gammas_complete.txt
        for i in list_of_nodes:
            f.write(f"{i}/output_{SimTel_version}\n")

    with open(
        f"{target_dir}/list_folder_{particle_type}.txt", "w"
    ) as f:  # creating list_folder_gammas.txt
        for i in list_of_nodes:
            f.write(f'{i.split("/")[-1]}\n')

    ####################################################################################
    # bash scripts that link the MC paths to each subdirectory.
    ####################################################################################

    with open(f"linking_MC_{particle_type}_paths.sh", "w") as f:
        lines_of_config_file = [
            "#!/bin/sh\n\n",
            "#SBATCH -p short\n",
            f"#SBATCH -J {process_name}\n\n",
            "#SBATCH -N 1\n\n",
            "ulimit -l unlimited\n",
            "ulimit -s unlimited\n",
            "ulimit -a\n\n",
            "while read -r -u 3 lineA && read -r -u 4 lineB\n",
            "do\n",
            f"    cd {target_dir}/DL1/MC/{particle_type}\n",
            "    mkdir $lineB\n",
            "    cd $lineA\n",
            "    ls -lR *.gz |wc -l\n",
            f"    ls *.gz > {target_dir}/DL1/MC/{particle_type}/$lineB/list_dl0.txt\n",
            '    string=$lineA"/"\n',
            f"    export file={target_dir}/DL1/MC/{particle_type}/$lineB/list_dl0.txt\n\n",
            "    cat $file | while read line; do echo $string${line}"
            + f" >>{target_dir}/DL1/MC/{particle_type}/$lineB/list_dl0_ok.txt; done\n\n",
            '    echo "folder $lineB  and node $lineA"\n',
            f'done 3<"{target_dir}/list_nodes_{particle_type}_complete.txt" 4<"{target_dir}/list_folder_{particle_type}.txt"\n',
            "",
        ]
        f.writelines(lines_of_config_file)

    ################################################################################################################
    # bash script that applies lst1_magic_mc_dl0_to_dl1.py to all MC data files.
    ################################################################################################################

    number_of_nodes = glob.glob(f"{MC_path}/node*")
    number_of_nodes = len(number_of_nodes) - 1

    with open(f"linking_MC_{particle_type}_paths_r.sh", "w") as f:
        lines_of_config_file = [
            "#!/bin/sh\n\n",
            "#SBATCH -p xxl\n",
            f"#SBATCH -J {process_name}\n",
            f"#SBATCH --array=0-{number_of_nodes}%50\n",
            "#SBATCH --mem=10g\n",
            "#SBATCH -N 1\n\n",
            "ulimit -l unlimited\n",
            "ulimit -s unlimited\n",
            "ulimit -a\n",
            f"cd {target_dir}/DL1/MC/{particle_type}\n\n",
            f"export INF={target_dir}\n",
            f"SAMPLE_LIST=($(<$INF/list_folder_{particle_type}.txt))\n",
            "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
            "cd $SAMPLE\n\n",
            f"export LOG={target_dir}/DL1/MC/{particle_type}"
            + "/simtel_{$SAMPLE}_all.log\n",
            "cat list_dl0_ok.txt | while read line\n",
            "do\n",
            f"    cd {target_dir}/../\n",
            f"    conda run -n {env_name} lst1_magic_mc_dl0_to_dl1 --input-file $line --output-dir {target_dir}/DL1/MC/{particle_type}/$SAMPLE --config-file {target_dir}/config_DL0_to_DL1.yaml --focal_length_choice {focal_length}>>$LOG 2>&1\n\n",
            "done\n",
            "",
        ]
        f.writelines(lines_of_config_file)


def lists_and_bash_gen_MAGIC(
    target_dir, telescope_ids, MAGIC_runs, source, env_name, NSB_match
):

    """
    Below we create a bash script that links the the MAGIC data paths to each subdirectory.

    Parameters
    ----------
    target_dir : str
        Directory to store the results
    telescope_ids : list
        List of the telescope IDs (set by the user)
    MAGIC_runs : str
        MAGIC dates and runs to be processed
    source : str
        Name of the target
    env_name : str
        Name of the environment
    NSB_match : bool
        If real data are matched to pre-processed MCs or not
    """
    process_name = f'{target_dir.split("/")[-2:][1]}'
    lines = [
        "#!/bin/sh\n\n",
        "#SBATCH -p short\n",
        f"#SBATCH -J {process_name}\n",
        "#SBATCH -N 1\n\n",
        "ulimit -l unlimited\n",
        "ulimit -s unlimited\n",
        "ulimit -a\n",
    ]
    with open("linking_MAGIC_data_paths.sh", "w") as f:
        f.writelines(lines)
        if NSB_match:
            if (len(MAGIC_runs) == 2) and (len(MAGIC_runs[0]) == 10):
                MAGIC = MAGIC_runs

                MAGIC_runs = []
                MAGIC_runs.append(MAGIC)

            for i in MAGIC_runs:
                for p in range(len(ST_begin)):
                    if (
                        time.strptime(i[0], "%Y_%m_%d")
                        >= time.strptime(ST_begin[p], "%Y_%m_%d")
                    ) and (
                        time.strptime(i[0], "%Y_%m_%d")
                        <= time.strptime(ST_end[p], "%Y_%m_%d")
                    ):
                        if telescope_ids[-1] > 0:
                            lines = [
                                f'export IN1=/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}\n',
                                f"export OUT1={target_dir}/v{__version__}/DL1/{ST_list[p]}/M2/{i[0]}/{i[1]}/logs \n",
                                f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_dl0.txt\n",
                            ]
                            f.writelines(lines)

                        f.write("\n")
                        if telescope_ids[-2] > 0:
                            lines = [
                                f'export IN1=/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}\n',
                                f"export OUT1={target_dir}/v{__version__}/DL1/{ST_list[p]}/M1/{i[0]}/{i[1]}/logs \n",
                                f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_dl0.txt\n",
                            ]
                            f.writelines(lines)
        else:
            if telescope_ids[-1] > 0:
                for i in MAGIC_runs:
                    lines = [
                        f'export IN1=/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}\n',
                        f"export OUT1={target_dir}/DL1/Observations/M2/{i[0]}/{i[1]}\n",
                        f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_dl0.txt\n",
                    ]
                    f.writelines(lines)
            f.write("\n")
            if telescope_ids[-2] > 0:
                for i in MAGIC_runs:
                    lines = [
                        f'export IN1=/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}\n',
                        f"export OUT1={target_dir}/DL1/Observations/M1/{i[0]}/{i[1]}\n",
                        f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_dl0.txt\n",
                    ]
                    f.writelines(lines)
    if NSB_match:

        if (telescope_ids[-2] > 0) or (telescope_ids[-1] > 0):
            for i in MAGIC_runs:

                for p in range(len(ST_begin)):
                    if (
                        time.strptime(i[0], "%Y_%m_%d")
                        >= time.strptime(ST_begin[p], "%Y_%m_%d")
                    ) and (
                        time.strptime(i[0], "%Y_%m_%d")
                        <= time.strptime(ST_end[p], "%Y_%m_%d")
                    ):

                        if telescope_ids[-1] > 0:
                            number_of_nodes = glob.glob(
                                f'/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}/*{i[1]}.*_Y_*.root'
                            )
                            number_of_nodes = len(number_of_nodes) - 1
                            if number_of_nodes < 0:
                                continue
                            lines = [
                                "#!/bin/sh\n\n",
                                "#SBATCH -p long\n",
                                f"#SBATCH -J {process_name}\n",
                                f"#SBATCH --array=0-{number_of_nodes}\n",
                                "#SBATCH -N 1\n\n",
                                "ulimit -l unlimited\n",
                                "ulimit -s unlimited\n",
                                "ulimit -a\n\n",
                                f"export OUTPUTDIR={target_dir}/v{__version__}/DL1/{ST_list[p]}/M2/{i[0]}/{i[1]}\n",
                                "SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_dl0.txt))\n",
                                "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                                "export LOG=$OUTPUTDIR/logs/real_0_1_task${SLURM_ARRAY_TASK_ID}.log\n",
                                f"time conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
                            ]
                            with open(
                                f"{source}_MAGIC-II_dl0_to_dl1_run_{i[1]}.sh", "w"
                            ) as f:
                                f.writelines(lines)

                        if telescope_ids[-2] > 0:
                            number_of_nodes = glob.glob(
                                f'/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}/*{i[1]}.*_Y_*.root'
                            )
                            number_of_nodes = len(number_of_nodes) - 1
                            if number_of_nodes < 0:
                                continue
                            lines = [
                                "#!/bin/sh\n\n",
                                "#SBATCH -p long\n",
                                f"#SBATCH -J {process_name}\n",
                                f"#SBATCH --array=0-{number_of_nodes}\n",
                                "#SBATCH -N 1\n\n",
                                "ulimit -l unlimited\n",
                                "ulimit -s unlimited\n",
                                "ulimit -a\n\n",
                                f"export OUTPUTDIR={target_dir}/v{__version__}/DL1/{ST_list[p]}/M1/{i[0]}/{i[1]}\n",
                                "SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_dl0.txt))\n",
                                "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                                "export LOG=$OUTPUTDIR/logs/real_0_1_task${SLURM_ARRAY_TASK_ID}.log\n",
                                f"time conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
                            ]
                            with open(
                                f"{source}_MAGIC-I_dl0_to_dl1_run_{i[1]}.sh", "w"
                            ) as f:
                                f.writelines(lines)
    else:
        if (telescope_ids[-2] > 0) or (telescope_ids[-1] > 0):
            for i in MAGIC_runs:
                if telescope_ids[-1] > 0:
                    number_of_nodes = glob.glob(
                        f'/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}/*{i[1]}.*_Y_*.root'
                    )
                    number_of_nodes = len(number_of_nodes) - 1

                    with open(f"{source}_MAGIC-II_dl0_to_dl1_run_{i[1]}.sh", "w") as f:
                        lines = [
                            "#!/bin/sh\n\n",
                            "#SBATCH -p long\n",
                            f"#SBATCH -J {process_name}\n",
                            f"#SBATCH --array=0-{number_of_nodes}\n",
                            "#SBATCH -N 1\n\n",
                            "ulimit -l unlimited\n",
                            "ulimit -s unlimited\n",
                            "ulimit -a\n\n",
                            f"export OUTPUTDIR={target_dir}/DL1/Observations/M2/{i[0]}/{i[1]}\n",
                            f"cd {target_dir}/../\n",
                            "SAMPLE_LIST=($(<$OUTPUTDIR/list_dl0.txt))\n",
                            "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                            "export LOG=$OUTPUTDIR/real_0_1_task${SLURM_ARRAY_TASK_ID}.log\n",
                            f"conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
                            "",
                        ]
                        f.writelines(lines)

                if telescope_ids[-2] > 0:
                    number_of_nodes = glob.glob(
                        f'/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}/*{i[1]}.*_Y_*.root'
                    )
                    number_of_nodes = len(number_of_nodes) - 1

                    with open(f"{source}_MAGIC-I_dl0_to_dl1_run_{i[1]}.sh", "w") as f:
                        lines = [
                            "#!/bin/sh\n\n",
                            "#SBATCH -p long\n",
                            f"#SBATCH -J {process_name}\n",
                            f"#SBATCH --array=0-{number_of_nodes}\n",
                            "#SBATCH -N 1\n\n",
                            "ulimit -l unlimited\n",
                            "ulimit -s unlimited\n",
                            "ulimit -a\n\n",
                            f"export OUTPUTDIR={target_dir}/DL1/Observations/M1/{i[0]}/{i[1]}\n",
                            f"cd {target_dir}/../\n",
                            "SAMPLE_LIST=($(<$OUTPUTDIR/list_dl0.txt))\n",
                            "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                            "export LOG=$OUTPUTDIR/real_0_1_task${SLURM_ARRAY_TASK_ID}.log\n",
                            f"conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
                            "",
                        ]
                        f.writelines(lines)


def directories_generator(target_dir, telescope_ids, MAGIC_runs, NSB_match):

    """
    Here we create all subdirectories for a given workspace and target name.

    Parameters
    ----------
    target_dir : str
        Directory to store the results
    telescope_ids : list
        List of the telescope IDs (set by the user)
    MAGIC_runs : str
        MAGIC dates and runs to be processed
    NSB_match : bool
        If real data are matched to pre-processed MCs or not
    """

    if NSB_match:
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        if not os.path.exists(f"{target_dir}/v{__version__}"):
            os.mkdir(f"{target_dir}/v{__version__}")
        if not os.path.exists(f"{target_dir}/v{__version__}/DL1"):
            os.mkdir(f"{target_dir}/v{__version__}/DL1")
        dl1_dir = str(f"{target_dir}/v{__version__}/DL1")
    else:
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            os.mkdir(f"{target_dir}/DL1")
            os.mkdir(f"{target_dir}/DL1/Observations")
            os.mkdir(f"{target_dir}/DL1/MC")
            os.mkdir(f"{target_dir}/DL1/MC/gammas")
            os.mkdir(f"{target_dir}/DL1/MC/gammadiffuse")
            os.mkdir(f"{target_dir}/DL1/MC/electrons")
            os.mkdir(f"{target_dir}/DL1/MC/protons")
            os.mkdir(f"{target_dir}/DL1/MC/helium")
        else:
            overwrite = input(
                f'MC directory for {target_dir.split("/")[-1]} already exists. Would you like to overwrite it? [only "y" or "n"]: '
            )
            if overwrite == "y":
                os.system(f"rm -r {target_dir}")
                os.mkdir(target_dir)
                os.mkdir(f"{target_dir}/DL1")
                os.mkdir(f"{target_dir}/DL1/Observations")
                os.mkdir(f"{target_dir}/DL1/MC")
                os.mkdir(f"{target_dir}/DL1/MC/gammas")
                os.mkdir(f"{target_dir}/DL1/MC/gammadiffuse")
                os.mkdir(f"{target_dir}/DL1/MC/electrons")
                os.mkdir(f"{target_dir}/DL1/MC/protons")
                os.mkdir(f"{target_dir}/DL1/MC/helium")
            else:
                print("Directory not modified.")

    ###########################################
    # MAGIC
    ###########################################
    if (len(MAGIC_runs) == 2) and (len(MAGIC_runs[0]) == 10):
        MAGIC = MAGIC_runs

        MAGIC_runs = []
        MAGIC_runs.append(MAGIC)
    if NSB_match:
        for i in MAGIC_runs:
            for p in range(len(ST_begin)):
                if (
                    time.strptime(i[0], "%Y_%m_%d")
                    >= time.strptime(ST_begin[p], "%Y_%m_%d")
                ) and (
                    time.strptime(i[0], "%Y_%m_%d")
                    <= time.strptime(ST_end[p], "%Y_%m_%d")
                ):
                    if telescope_ids[-1] > 0:
                        if not os.path.exists(f"{dl1_dir}/{ST_list[p]}"):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}")
                        if not os.path.exists(f"{dl1_dir}/{ST_list[p]}/M2"):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}/M2")
                        if not os.path.exists(f"{dl1_dir}/{ST_list[p]}/M2/{i[0]}"):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}/M2/{i[0]}")

                        if not os.path.exists(
                            f"{dl1_dir}/{ST_list[p]}/M2/{i[0]}/{i[1]}"
                        ):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}/M2/{i[0]}/{i[1]}")
                        if not os.path.exists(
                            f"{dl1_dir}/{ST_list[p]}/M2/{i[0]}/{i[1]}/logs"
                        ):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}/M2/{i[0]}/{i[1]}/logs")
                    if telescope_ids[-2] > 0:
                        if not os.path.exists(f"{dl1_dir}/{ST_list[p]}"):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}")
                        if not os.path.exists(f"{dl1_dir}/{ST_list[p]}/M1"):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}/M1")
                        if not os.path.exists(f"{dl1_dir}/{ST_list[p]}/M1/{i[0]}"):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}/M1/{i[0]}")

                        if not os.path.exists(
                            f"{dl1_dir}/{ST_list[p]}/M1/{i[0]}/{i[1]}"
                        ):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}/M1/{i[0]}/{i[1]}")
                        if not os.path.exists(
                            f"{dl1_dir}/{ST_list[p]}/M1/{i[0]}/{i[1]}/logs"
                        ):
                            os.mkdir(f"{dl1_dir}/{ST_list[p]}/M1/{i[0]}/{i[1]}/logs")
    else:
        if telescope_ids[-1] > 0:
            if not os.path.exists(f"{target_dir}/DL1/Observations/M2"):
                os.mkdir(f"{target_dir}/DL1/Observations/M2")
                for i in MAGIC_runs:
                    if not os.path.exists(f"{target_dir}/DL1/Observations/M2/{i[0]}"):
                        os.mkdir(f"{target_dir}/DL1/Observations/M2/{i[0]}")
                        os.mkdir(f"{target_dir}/DL1/Observations/M2/{i[0]}/{i[1]}")
                    else:
                        os.mkdir(f"{target_dir}/DL1/Observations/M2/{i[0]}/{i[1]}")

        if telescope_ids[-2] > 0:
            if not os.path.exists(f"{target_dir}/DL1/Observations/M1"):
                os.mkdir(f"{target_dir}/DL1/Observations/M1")
                for i in MAGIC_runs:
                    if not os.path.exists(f"{target_dir}/DL1/Observations/M1/{i[0]}"):
                        os.mkdir(f"{target_dir}/DL1/Observations/M1/{i[0]}")
                        os.mkdir(f"{target_dir}/DL1/Observations/M1/{i[0]}/{i[1]}")
                    else:
                        os.mkdir(f"{target_dir}/DL1/Observations/M1/{i[0]}/{i[1]}")


def main():

    """Here we read the config file and call the functions to generate the necessary directories, bash scripts and launching the jobs."""

    # Here we are simply collecting the parameters from the command line, as input file, output directory, and configuration file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-type",
        "-t",
        choices=["onlyMAGIC", "onlyMC"],
        dest="analysis_type",
        type=str,
        default="doEverything",
        help="You can type 'onlyMAGIC' or 'onlyMC' to run this script only on MAGIC or MC data, respectively.",
    )

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
    SimTel_version = config["general"]["SimTel_version"]
    env_name = config["general"]["env_name"]
    NSB_match = config["general"]["NSB_matching"]
    MAGIC_runs_and_dates = config["general"]["MAGIC_runs"]
    MAGIC_runs = np.genfromtxt(
        MAGIC_runs_and_dates, dtype=str, delimiter=","
    )  # READ LIST OF DATES AND RUNS: format table where each line is like "2020_11_19,5093174"
    target_dir = str(
        Path(config["directories"]["workspace_dir"])
        / config["directories"]["target_name"]
    )
    LST_runs_and_dates = config["general"]["LST_runs"]
    MC_gammas = str(Path(config["directories"]["MC_gammas"]))
    MC_electrons = str(Path(config["directories"]["MC_electrons"]))
    MC_helium = str(Path(config["directories"]["MC_helium"]))
    MC_protons = str(Path(config["directories"]["MC_protons"]))
    MC_gammadiff = str(Path(config["directories"]["MC_gammadiff"]))
    focal_length = config["general"]["focal_length"]
    source = config["directories"]["target_name"]
    noise_value = [0, 0, 0]
    if not NSB_match:
        running, nsb = nsb_avg(source, args.config_file, LST_runs_and_dates)
        if running != "y":
            print("OK... The script was terminated by the user choice.")
            return
        noisebright = 1.15 * pow(nsb, 1.115)
        biasdim = 0.358 * pow(nsb, 0.805)
        noise_value = [nsb, noisebright, biasdim]
    else:
        collect_nsb(config)

    print("*** Reducing DL0 to DL1 data***")
    print(f'Process name: {target_dir.split("/")[-2:][1]}')
    print(
        f'To check the jobs submitted to the cluster, type: squeue -n {target_dir.split("/")[-2:][1]}'
    )

    directories_generator(
        target_dir, telescope_ids, MAGIC_runs, NSB_match
    )  # Here we create all the necessary directories in the given workspace and collect the main directory of the target
    config_file_gen(telescope_ids, target_dir, noise_value, NSB_match)

    if not NSB_match:
        # Below we run the analysis on the MC data
        if (args.analysis_type == "onlyMC") or (args.analysis_type == "doEverything"):
            lists_and_bash_generator(
                "gammas", target_dir, MC_gammas, SimTel_version, focal_length, env_name
            )  # gammas
            lists_and_bash_generator(
                "electrons",
                target_dir,
                MC_electrons,
                SimTel_version,
                focal_length,
                env_name,
            )  # electrons
            lists_and_bash_generator(
                "helium", target_dir, MC_helium, SimTel_version, focal_length, env_name
            )  # helium
            lists_and_bash_generator(
                "protons",
                target_dir,
                MC_protons,
                SimTel_version,
                focal_length,
                env_name,
            )  # protons
            lists_and_bash_generator(
                "gammadiffuse",
                target_dir,
                MC_gammadiff,
                SimTel_version,
                focal_length,
                env_name,
            )  # gammadiffuse

            # Here we do the MC DL0 to DL1 conversion:
            list_of_MC = glob.glob("linking_MC_*s.sh")

            # os.system("RES=$(sbatch --parsable linking_MC_gammas_paths.sh) && sbatch --dependency=afterok:$RES MC_dl0_to_dl1.sh")

            for n, run in enumerate(list_of_MC):
                if n == 0:
                    launch_jobs_MC = f"linking{n}=$(sbatch --parsable {run}) && running{n}=$(sbatch --parsable --dependency=afterany:$linking{n} {run[0:-3]}_r.sh)"
                else:
                    launch_jobs_MC = f"{launch_jobs_MC} && linking{n}=$(sbatch --parsable {run}) && running{n}=$(sbatch --parsable --dependency=afterany:$linking{n} {run[0:-3]}_r.sh)"

            os.system(launch_jobs_MC)

    # Below we run the analysis on the MAGIC data
    if (
        (args.analysis_type == "onlyMAGIC")
        or (args.analysis_type == "doEverything")
        or (NSB_match)
    ):
        lists_and_bash_gen_MAGIC(
            target_dir, telescope_ids, MAGIC_runs, source, env_name, NSB_match
        )  # MAGIC real data
        if (telescope_ids[-2] > 0) or (telescope_ids[-1] > 0):
            list_of_MAGIC_runs = glob.glob(f"{source}_MAGIC-*.sh")
            if len(list_of_MAGIC_runs) < 1:
                print(
                    "Warning: no bash script has been produced. Please check the provided MAGIC_runs.txt and the MAGIC calibrated data"
                )
                return

            for n, run in enumerate(list_of_MAGIC_runs):
                if n == 0:
                    launch_jobs = f"linking=$(sbatch --parsable linking_MAGIC_data_paths.sh)  &&  RES{n}=$(sbatch --parsable --dependency=afterany:$linking {run})"
                else:
                    launch_jobs = f"{launch_jobs} && RES{n}=$(sbatch --parsable --dependency=afterany:$linking {run})"

            os.system(launch_jobs)


if __name__ == "__main__":
    main()
