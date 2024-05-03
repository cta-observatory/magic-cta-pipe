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

# import time
from pathlib import Path

import joblib
import numpy as np
import yaml

from magicctapipe import __version__
from magicctapipe.io import resource_file

__all__ = [
    "config_file_gen",
    "lists_and_bash_generator",
    "lists_and_bash_gen_MAGIC",
    "directories_generator",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def config_file_gen(ids, target_dir, noise_value, NSB_match, source_name):

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
    source_name : str
        Name of the target source
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

    config_file = resource_file("config.yaml")
    with open(
        config_file, "rb"
    ) as fc:  # "rb" mode opens the file in binary format for reading
        config_dict = yaml.safe_load(fc)
    LST_config = config_dict["LST"]
    MAGIC_config = config_dict["MAGIC"]

    if not NSB_match:
        LST_config["increase_nsb"]["extra_noise_in_dim_pixels"] = noise_value[0]
        LST_config["increase_nsb"]["extra_bias_in_dim_pixels"] = noise_value[2]
        LST_config["increase_nsb"]["extra_noise_in_bright_pixels"] = noise_value[1]
    conf = {}
    conf["LST"] = LST_config

    conf["MAGIC"] = MAGIC_config
    if not NSB_match:
        file_name = f"{target_dir}/{source_name}/config_DL0_to_DL1.yaml"
    else:
        file_name = f"{target_dir}/v{__version__}/{source_name}/config_DL0_to_DL1.yaml"
    with open(file_name, "w") as f:
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
    particle_type,
    target_dir,
    MC_path,
    SimTel_version,
    focal_length,
    env_name,
    source_name,
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
    source_name : str
        Name of the target source
    """

    if MC_path == "":
        return

    process_name = source_name

    list_of_nodes = glob.glob(f"{MC_path}/node*")
    with open(
        f"{target_dir}/{source_name}/list_nodes_{particle_type}_complete.txt", "w"
    ) as f:  # creating list_nodes_gammas_complete.txt
        for i in list_of_nodes:
            f.write(f"{i}/output_{SimTel_version}\n")

    with open(
        f"{target_dir}/{source_name}/list_folder_{particle_type}.txt", "w"
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
            "#SBATCH -n 1\n\n",
            f"#SBATCH --output={target_dir}/{source_name}/DL1/MC/{particle_type}/slurm-linkMC-%x.%j.out"
            f"#SBATCH --error={target_dir}/{source_name}/DL1/MC/{particle_type}/slurm-linkMC-%x.%j.err"
            "ulimit -l unlimited\n",
            "ulimit -s unlimited\n",
            "ulimit -a\n\n",
            "while read -r -u 3 lineA && read -r -u 4 lineB\n",
            "do\n",
            f"    cd {target_dir}/{source_name}/DL1/MC/{particle_type}\n",
            "    mkdir $lineB\n",
            "    cd $lineA\n",
            "    ls -lR *.gz |wc -l\n",
            f"    ls *.gz > {target_dir}/{source_name}/DL1/MC/{particle_type}/$lineB/list_dl0.txt\n",
            '    string=$lineA"/"\n',
            f"    export file={target_dir}/{source_name}/DL1/MC/{particle_type}/$lineB/list_dl0.txt\n\n",
            "    cat $file | while read line; do echo $string${line}"
            + f" >>{target_dir}/{source_name}/DL1/MC/{particle_type}/$lineB/list_dl0_ok.txt; done\n\n",
            '    echo "folder $lineB  and node $lineA"\n',
            f'done 3<"{target_dir}/{source_name}/list_nodes_{particle_type}_complete.txt" 4<"{target_dir}/{source_name}/list_folder_{particle_type}.txt"\n',
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
            f"#SBATCH --output={target_dir}/{source_name}/DL1/MC/{particle_type}/slurm-%x.%A_%a.out"
            f"#SBATCH --error={target_dir}/{source_name}/DL1/MC/{particle_type}/slurm-%x.%A_%a.err"
            "#SBATCH -n 1\n\n",
            "ulimit -l unlimited\n",
            "ulimit -s unlimited\n",
            "ulimit -a\n",
            f"cd {target_dir}/{source_name}/DL1/MC/{particle_type}\n\n",
            f"export INF={target_dir}/{source_name}\n",
            f"SAMPLE_LIST=($(<$INF/list_folder_{particle_type}.txt))\n",
            "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
            "cd $SAMPLE\n\n",
            f"export LOG={target_dir}/{source_name}/DL1/MC/{particle_type}"
            + "/simtel_{$SAMPLE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_all.log\n",
            "cat list_dl0_ok.txt | while read line\n",
            "do\n",
            f"    cd {target_dir}/{source_name}/../\n",
            f"    conda run -n {env_name} lst1_magic_mc_dl0_to_dl1 --input-file $line --output-dir {target_dir}/{source_name}/DL1/MC/{particle_type}/$SAMPLE --config-file {target_dir}/{source_name}/config_DL0_to_DL1.yaml --focal_length_choice {focal_length}>>$LOG 2>&1\n\n",
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
    MAGIC_runs : array
        MAGIC dates and runs to be processed
    source : str
        Name of the target
    env_name : str
        Name of the environment
    NSB_match : bool
        If real data are matched to pre-processed MCs or not
    """
    process_name = source
    lines = [
        "#!/bin/sh\n\n",
        "#SBATCH -p short\n",
        f"#SBATCH -J {process_name}\n",
        "#SBATCH -n 1\n\n",
        f"#SBATCH --output={target_dir}/v{__version__}/{source}/DL1/slurm-linkMAGIC-%x.%j.out"
        f"#SBATCH --error={target_dir}/v{__version__}/{source}/DL1/slurm-linkMAGIC-%x.%j.err"
        "ulimit -l unlimited\n",
        "ulimit -s unlimited\n",
        "ulimit -a\n",
    ]
    with open(f"{source}_linking_MAGIC_data_paths.sh", "w") as f:
        f.writelines(lines)
        if NSB_match:

            if (len(MAGIC_runs) == 2) and (len(MAGIC_runs[0]) == 10):
                MAGIC = MAGIC_runs

                MAGIC_runs = []
                MAGIC_runs.append(MAGIC)

            for i in MAGIC_runs:

                if telescope_ids[-1] > 0:
                    lines = [
                        f'export IN1=/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}\n',
                        f"export OUT1={target_dir}/v{__version__}/{source}/DL1/M2/{i[0]}/{i[1]}/logs \n",
                        f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_dl0.txt\n",
                    ]
                    f.writelines(lines)

                f.write("\n")
                if telescope_ids[-2] > 0:
                    lines = [
                        f'export IN1=/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}\n',
                        f"export OUT1={target_dir}/v{__version__}/{source}/DL1/M1/{i[0]}/{i[1]}/logs \n",
                        f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_dl0.txt\n",
                    ]
                    f.writelines(lines)
        else:
            if telescope_ids[-1] > 0:
                for i in MAGIC_runs:
                    lines = [
                        f'export IN1=/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}\n',
                        f"export OUT1={target_dir}/{source}/DL1/Observations/M2/{i[0]}/{i[1]}\n",
                        f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_dl0.txt\n",
                    ]
                    f.writelines(lines)
            f.write("\n")
            if telescope_ids[-2] > 0:
                for i in MAGIC_runs:
                    lines = [
                        f'export IN1=/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}\n',
                        f"export OUT1={target_dir}/{source}/DL1/Observations/M1/{i[0]}/{i[1]}\n",
                        f"ls $IN1/*{i[1][-2:]}.*_Y_*.root > $OUT1/list_dl0.txt\n",
                    ]
                    f.writelines(lines)
    if NSB_match:

        if (telescope_ids[-2] > 0) or (telescope_ids[-1] > 0):
            for i in MAGIC_runs:

                if telescope_ids[-1] > 0:
                    number_of_nodes = glob.glob(
                        f'/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/{i[0].split("_")[0]}/{i[0].split("_")[1]}/{i[0].split("_")[2]}/*{i[1]}.*_Y_*.root'
                    )
                    number_of_nodes = len(number_of_nodes) - 1
                    if number_of_nodes < 0:
                        continue
                    lines = [
                        "#!/bin/sh\n\n",
                        "#SBATCH -p short\n",
                        f"#SBATCH -J {process_name}\n",
                        f"#SBATCH --array=0-{number_of_nodes}\n",
                        "#SBATCH -n 1\n\n",
                        f"#SBATCH --output={target_dir}/v{__version__}/{source}/DL1/M2/{i[0]}/{i[1]}/logs/slurm-%x.%A_%a.out"
                        f"#SBATCH --error={target_dir}/v{__version__}/{source}/DL1/M2/{i[0]}/{i[1]}/logs/slurm-%x.%A_%a.err"
                        "#SBATCH --mem 2g\n\n",
                        "ulimit -l unlimited\n",
                        "ulimit -s unlimited\n",
                        "ulimit -a\n\n",
                        f"export OUTPUTDIR={target_dir}/v{__version__}/{source}/DL1/M2/{i[0]}/{i[1]}\n",
                        "SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_dl0.txt))\n",
                        "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                        "export LOG=$OUTPUTDIR/logs/real_0_1_task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                        f"conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/v{__version__}/{source}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
                    ]
                    with open(f"{source}_MAGIC-II_dl0_to_dl1_run_{i[1]}.sh", "w") as f:
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
                        "#SBATCH -p short\n",
                        f"#SBATCH -J {process_name}\n",
                        f"#SBATCH --array=0-{number_of_nodes}\n",
                        "#SBATCH -n 1\n\n",
                        f"#SBATCH --output={target_dir}/v{__version__}/{source}/DL1/M1/{i[0]}/{i[1]}/logs/slurm-%x.%A_%a.out"
                        f"#SBATCH --error={target_dir}/v{__version__}/{source}/DL1/M1/{i[0]}/{i[1]}/logs/slurm-%x.%A_%a.err"
                        "#SBATCH --mem 2g\n\n",
                        "ulimit -l unlimited\n",
                        "ulimit -s unlimited\n",
                        "ulimit -a\n\n",
                        f"export OUTPUTDIR={target_dir}/v{__version__}/{source}/DL1/M1/{i[0]}/{i[1]}\n",
                        "SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_dl0.txt))\n",
                        "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                        "export LOG=$OUTPUTDIR/logs/real_0_1_task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                        f"conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/v{__version__}/{source}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
                    ]
                    with open(f"{source}_MAGIC-I_dl0_to_dl1_run_{i[1]}.sh", "w") as f:
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
                            "#SBATCH -n 1\n\n",
                            f"#SBATCH --output={target_dir}/{source}/DL1/Observations/M2/{i[0]}/{i[1]}/slurm-%x.%A_%a.out"
                            f"#SBATCH --error={target_dir}/{source}/DL1/Observations/M2/{i[0]}/{i[1]}/slurm-%x.%A_%a.err"
                            "ulimit -l unlimited\n",
                            "ulimit -s unlimited\n",
                            "ulimit -a\n\n",
                            f"export OUTPUTDIR={target_dir}/{source}/DL1/Observations/M2/{i[0]}/{i[1]}\n",
                            f"cd {target_dir}/{source}/../\n",
                            "SAMPLE_LIST=($(<$OUTPUTDIR/list_dl0.txt))\n",
                            "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                            "export LOG=$OUTPUTDIR/real_0_1_task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                            f"conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/{source}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
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
                            "#SBATCH -n 1\n\n",
                            f"#SBATCH --output={target_dir}/{source}/DL1/Observations/M1/{i[0]}/{i[1]}/slurm-%x.%A_%a.out"
                            f"#SBATCH --error={target_dir}/{source}/DL1/Observations/M1/{i[0]}/{i[1]}/slurm-%x.%A_%a.err"
                            "ulimit -l unlimited\n",
                            "ulimit -s unlimited\n",
                            "ulimit -a\n\n",
                            f"export OUTPUTDIR={target_dir}/{source}/DL1/Observations/M1/{i[0]}/{i[1]}\n",
                            f"cd {target_dir}/{source}/../\n",
                            "SAMPLE_LIST=($(<$OUTPUTDIR/list_dl0.txt))\n",
                            "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n",
                            "export LOG=$OUTPUTDIR/real_0_1_task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log\n",
                            f"conda run -n {env_name} magic_calib_to_dl1 --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/{source}/config_DL0_to_DL1.yaml >$LOG 2>&1\n",
                            "",
                        ]
                        f.writelines(lines)


def directories_generator(
    target_dir, telescope_ids, MAGIC_runs, NSB_match, source_name
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
    NSB_match : bool
        If real data are matched to pre-processed MCs or not
    source_name : str
        Name of the target source
    """

    if NSB_match:
        if not os.path.exists(f"{target_dir}/v{__version__}"):
            os.mkdir(f"{target_dir}/v{__version__}")
        if not os.path.exists(f"{target_dir}/v{__version__}/{source_name}"):
            os.mkdir(f"{target_dir}/v{__version__}/{source_name}")
        if not os.path.exists(f"{target_dir}/v{__version__}/{source_name}/DL1"):
            os.mkdir(f"{target_dir}/v{__version__}/{source_name}/DL1")
        dl1_dir = str(f"{target_dir}/v{__version__}/{source_name}/DL1")
    else:
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            os.mkdir(f"{target_dir}/{source_name}/DL1")
            os.mkdir(f"{target_dir}/{source_name}/DL1/Observations")
            os.mkdir(f"{target_dir}/{source_name}/DL1/MC")
            os.mkdir(f"{target_dir}/{source_name}/DL1/MC/gammas")
            os.mkdir(f"{target_dir}/{source_name}/DL1/MC/gammadiffuse")
            os.mkdir(f"{target_dir}/{source_name}/DL1/MC/electrons")
            os.mkdir(f"{target_dir}/{source_name}/DL1/MC/protons")
            os.mkdir(f"{target_dir}/{source_name}/DL1/MC/helium")
        else:
            overwrite = input(
                f'MC directory for {target_dir.split("/")[-1]} already exists. Would you like to overwrite it? [only "y" or "n"]: '
            )
            if overwrite == "y":
                os.system(f"rm -r {target_dir}/{source_name}")
                os.mkdir(target_dir)
                os.mkdir(f"{target_dir}/{source_name}/DL1")
                os.mkdir(f"{target_dir}/{source_name}/DL1/Observations")
                os.mkdir(f"{target_dir}/{source_name}/DL1/MC")
                os.mkdir(f"{target_dir}/{source_name}/DL1/MC/gammas")
                os.mkdir(f"{target_dir}/{source_name}/DL1/MC/gammadiffuse")
                os.mkdir(f"{target_dir}/{source_name}/DL1/MC/electrons")
                os.mkdir(f"{target_dir}/{source_name}/DL1/MC/protons")
                os.mkdir(f"{target_dir}/{source_name}/DL1/MC/helium")
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

            if telescope_ids[-1] > 0:
                if not os.path.exists(f"{dl1_dir}"):
                    os.mkdir(f"{dl1_dir}")
                if not os.path.exists(f"{dl1_dir}/M2"):
                    os.mkdir(f"{dl1_dir}/M2")
                if not os.path.exists(f"{dl1_dir}/M2/{i[0]}"):
                    os.mkdir(f"{dl1_dir}/M2/{i[0]}")

                if not os.path.exists(f"{dl1_dir}/M2/{i[0]}/{i[1]}"):
                    os.mkdir(f"{dl1_dir}/M2/{i[0]}/{i[1]}")
                if not os.path.exists(f"{dl1_dir}/M2/{i[0]}/{i[1]}/logs"):
                    os.mkdir(f"{dl1_dir}/M2/{i[0]}/{i[1]}/logs")
            if telescope_ids[-2] > 0:
                if not os.path.exists(f"{dl1_dir}"):
                    os.mkdir(f"{dl1_dir}")
                if not os.path.exists(f"{dl1_dir}/M1"):
                    os.mkdir(f"{dl1_dir}/M1")
                if not os.path.exists(f"{dl1_dir}/M1/{i[0]}"):
                    os.mkdir(f"{dl1_dir}/M1/{i[0]}")

                if not os.path.exists(f"{dl1_dir}/M1/{i[0]}/{i[1]}"):
                    os.mkdir(f"{dl1_dir}/M1/{i[0]}/{i[1]}")
                if not os.path.exists(f"{dl1_dir}/M1/{i[0]}/{i[1]}/logs"):
                    os.mkdir(f"{dl1_dir}/M1/{i[0]}/{i[1]}/logs")
    else:
        if telescope_ids[-1] > 0:
            if not os.path.exists(f"{target_dir}/{source_name}/DL1/Observations/M2"):
                os.mkdir(f"{target_dir}/{source_name}/DL1/Observations/M2")
                for i in MAGIC_runs:
                    if not os.path.exists(
                        f"{target_dir}/{source_name}/DL1/Observations/M2/{i[0]}"
                    ):
                        os.mkdir(
                            f"{target_dir}/{source_name}/DL1/Observations/M2/{i[0]}"
                        )
                        os.mkdir(
                            f"{target_dir}/{source_name}/DL1/Observations/M2/{i[0]}/{i[1]}"
                        )
                    else:
                        os.mkdir(
                            f"{target_dir}/{source_name}/DL1/Observations/M2/{i[0]}/{i[1]}"
                        )

        if telescope_ids[-2] > 0:
            if not os.path.exists(f"{target_dir}/{source_name}/DL1/Observations/M1"):
                os.mkdir(f"{target_dir}/{source_name}/DL1/Observations/M1")
                for i in MAGIC_runs:
                    if not os.path.exists(
                        f"{target_dir}/{source_name}/DL1/Observations/M1/{i[0]}"
                    ):
                        os.mkdir(
                            f"{target_dir}/{source_name}/DL1/Observations/M1/{i[0]}"
                        )
                        os.mkdir(
                            f"{target_dir}/{source_name}/DL1/Observations/M1/{i[0]}/{i[1]}"
                        )
                    else:
                        os.mkdir(
                            f"{target_dir}/{source_name}/DL1/Observations/M1/{i[0]}/{i[1]}"
                        )


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

    # LST_runs_and_dates = config["general"]["LST_runs"]
    MC_gammas = str(Path(config["directories"]["MC_gammas"]))
    MC_electrons = str(Path(config["directories"]["MC_electrons"]))
    MC_helium = str(Path(config["directories"]["MC_helium"]))
    MC_protons = str(Path(config["directories"]["MC_protons"]))
    MC_gammadiff = str(Path(config["directories"]["MC_gammadiff"]))
    focal_length = config["general"]["focal_length"]
    source = config["data_selection"]["source_name_output"]

    source_list = []
    if source is not None:
        source_list = joblib.load("list_sources.dat")

    else:
        source_list.append(source)
    for source_name in source_list:
        target_dir = Path(config["directories"]["workspace_dir"])

        MAGIC_runs_and_dates = f"{source_name}_MAGIC_runs.txt"
        MAGIC_runs = np.genfromtxt(
            MAGIC_runs_and_dates, dtype=str, delimiter=","
        )  # READ LIST OF DATES AND RUNS: format table where each line is like "2020_11_19,5093174"

        noise_value = [0, 0, 0]
        if not NSB_match:
            nsb = config["general"]["NSB_MC"]

            noisebright = 1.15 * pow(nsb, 1.115)
            biasdim = 0.358 * pow(nsb, 0.805)
            noise_value = [nsb, noisebright, biasdim]

        # TODO: fix here above
        print("*** Converting DL0 into DL1 data ***")
        print(f"Process name: {source_name}")
        print(
            f"To check the jobs submitted to the cluster, type: squeue -n {source_name}"
        )
        print("This process will take about 10 min to run if the IT cluster is free.")

        directories_generator(
            target_dir, telescope_ids, MAGIC_runs, NSB_match, source_name
        )  # Here we create all the necessary directories in the given workspace and collect the main directory of the target
        config_file_gen(
            telescope_ids, target_dir, noise_value, NSB_match, source_name
        )  # TODO: fix here

        if not NSB_match:
            # Below we run the analysis on the MC data
            if (args.analysis_type == "onlyMC") or (
                args.analysis_type == "doEverything"
            ):
                lists_and_bash_generator(
                    "gammas",
                    target_dir,
                    MC_gammas,
                    SimTel_version,
                    focal_length,
                    env_name,
                    source_name,
                )  # gammas
                lists_and_bash_generator(
                    "electrons",
                    target_dir,
                    MC_electrons,
                    SimTel_version,
                    focal_length,
                    env_name,
                    source_name,
                )  # electrons
                lists_and_bash_generator(
                    "helium",
                    target_dir,
                    MC_helium,
                    SimTel_version,
                    focal_length,
                    env_name,
                    source_name,
                )  # helium
                lists_and_bash_generator(
                    "protons",
                    target_dir,
                    MC_protons,
                    SimTel_version,
                    focal_length,
                    env_name,
                    source_name,
                )  # protons
                lists_and_bash_generator(
                    "gammadiffuse",
                    target_dir,
                    MC_gammadiff,
                    SimTel_version,
                    focal_length,
                    env_name,
                    source_name,
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
                target_dir,
                telescope_ids,
                MAGIC_runs,
                source_name,
                env_name,
                NSB_match,
            )  # MAGIC real data
            if (telescope_ids[-2] > 0) or (telescope_ids[-1] > 0):
                list_of_MAGIC_runs = glob.glob(f"{source_name}_MAGIC-*.sh")
                if len(list_of_MAGIC_runs) < 1:
                    print(
                        "Warning: no bash script has been produced. Please check the provided MAGIC_runs.txt and the MAGIC calibrated data"
                    )
                    continue

                for n, run in enumerate(list_of_MAGIC_runs):
                    if n == 0:
                        launch_jobs = f"linking=$(sbatch --parsable {source_name}_linking_MAGIC_data_paths.sh)  &&  RES{n}=$(sbatch --parsable --dependency=afterany:$linking {run})"
                    else:
                        launch_jobs = f"{launch_jobs} && RES{n}=$(sbatch --parsable --dependency=afterany:$linking {run})"

                os.system(launch_jobs)


if __name__ == "__main__":
    main()
