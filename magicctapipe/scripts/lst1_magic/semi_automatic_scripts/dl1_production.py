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
    "lists_and_bash_generator",
    "lists_and_bash_gen_MAGIC",
    "directories_generator_real",
    "directories_generator_MC",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def config_file_gen(target_dir, NSB_match, source_name, config_file):

    """
    Here we create the configuration file needed for transforming DL0 into DL1

    Parameters
    ----------
    target_dir : path
        Directory to store the results
    NSB_match : bool
        If real data are matched to pre-processed MCs or not
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
        "LST": config_dict["LST"],
        "MAGIC": config_dict["MAGIC"],
    }
    if source_name == "MC":
        file_name = f"{target_dir}/v{__version__}/MC/config_DL0_to_DL1.yaml"
    else:
        file_name = f"{target_dir}/v{__version__}/{source_name}/config_DL0_to_DL1.yaml"
    with open(file_name, "w") as f:
        yaml.dump(conf, f, default_flow_style=False)


def lists_and_bash_generator(
    particle_type,
    target_dir,
    MC_path,
    focal_length,
    env_name,
    cluster,
):

    """
    This function creates the lists list_nodes_*_complete.txt and list_folder_*.txt with the MC file paths.
    After that, it generates a few bash scripts to link the MC paths to each subdirectory and to process them from DL0 to DL1.
    These bash scripts will be called later in the main() function below. This step will be skipped in case the MC path has not been provided (MC_path='')

    Parameters
    ----------
    particle_type : str
        Particle type (e.g., protons)
    target_dir : str
        Directory to store the results
    MC_path : str
        Path to the MCs DL0s
    focal_length : str
        Focal length to be used to process MCs (e.g., 'nominal')
    env_name : str
        Name of the environment
    cluster : str
        Cluster system
    """

    if MC_path == "":
        return
    print(f"running {particle_type} from {MC_path}")
    process_name = "MC"

    list_of_nodes = glob.glob(f"{MC_path}/node*")
    dir1 = f"{target_dir}/v{__version__}/MC"
    with open(
        f"{dir1}/logs/list_nodes_{particle_type}_complete.txt", "w"
    ) as f:  # creating list_nodes_gammas_complete.txt
        for i in list_of_nodes:
            out_list = glob.glob(f"{i}/output*")
            if len(out_list) == 0:
                logger.error(
                    f"No output file for node {i}, or the directory structure is not the usual one. Skipping..."
                )
                continue
            elif len(out_list) == 1:
                f.write(f"{out_list[0]}\n")
            else:
                output_index = input(
                    f"The available outputs are {out_list}, please provide the array index of the desired one:"
                )
                f.write(f"{out_list[output_index]}\n")

    with open(
        f"{dir1}/logs/list_folder_{particle_type}.txt", "w"
    ) as f:  # creating list_folder_gammas.txt
        for i in list_of_nodes:
            f.write(f'{i.split("/")[-1]}\n')

    ####################################################################################
    # bash scripts that link the MC paths to each subdirectory.
    ####################################################################################
    if cluster != "SLURM":
        logger.warning(
            "Automatic processing not implemented for the cluster indicated in the config file"
        )
        return
    with open(f"linking_MC_{particle_type}_paths.sh", "w") as f:
        slurm = slurm_lines(
            queue="short",
            job_name=process_name,
            out_name=f"{dir1}/DL1/{particle_type}/logs/slurm-linkMC-%x.%j",
        )
        lines_of_config_file = slurm + [
            "while read -r -u 3 lineA && read -r -u 4 lineB\n",
            "do\n",
            f"    cd {dir1}/DL1/{particle_type}\n",
            "    mkdir $lineB\n",
            "    cd $lineA\n",
            "    ls -lR *.gz |wc -l\n",
            f"    mkdir -p {dir1}/DL1/{particle_type}/$lineB/logs/\n",
            f"    ls *.gz > {dir1}/DL1/{particle_type}/$lineB/logs/list_dl0.txt\n",
            '    string=$lineA"/"\n',
            f"    export file={dir1}/DL1/{particle_type}/$lineB/logs/list_dl0.txt\n\n",
            "    cat $file | while read line; do echo $string${line}"
            + f" >>{dir1}/DL1/{particle_type}/$lineB/logs/list_dl0_ok.txt; done\n\n",
            '    echo "folder $lineB  and node $lineA"\n',
            f'done 3<"{dir1}/logs/list_nodes_{particle_type}_complete.txt" 4<"{dir1}/logs/list_folder_{particle_type}.txt"\n',
            "",
        ]
        f.writelines(lines_of_config_file)

    ################################################################################################################
    # bash script that applies lst1_magic_mc_dl0_to_dl1.py to all MC data files.
    ################################################################################################################

    number_of_nodes = glob.glob(f"{MC_path}/node*")
    number_of_nodes = len(number_of_nodes) - 1
    with open(f"linking_MC_{particle_type}_paths_r.sh", "w") as f:
        slurm = slurm_lines(
            queue="xxl",
            job_name=process_name,
            array=number_of_nodes,
            mem="10g",
            out_name=f"{dir1}/DL1/{particle_type}/logs/slurm-%x.%A_%a",
        )
        lines_of_config_file = slurm + [
            f"cd {dir1}/DL1/{particle_type}\n\n",
            f"export INF={dir1}/logs\n",
            f"SAMPLE_LIST=($(<$INF/list_folder_{particle_type}.txt))\n",
            "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
            "cd $SAMPLE\n\n",
            f"export LOG={dir1}/DL1/{particle_type}/logs/simtel_{{$SAMPLE}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}_all.log\n",
            "cat logs/list_dl0_ok.txt | while read line\n",
            "do\n",
            f"    cd {dir1}/../\n",
            f"    conda run -n {env_name} lst1_magic_mc_dl0_to_dl1 --input-file $line --output-dir {dir1}/DL1/{particle_type}/$SAMPLE --config-file {dir1}/config_DL0_to_DL1.yaml --focal_length_choice {focal_length}>>$LOG 2>&1\n\n",
            "done\n",
            "",
        ]
        f.writelines(lines_of_config_file)


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
        os.makedirs(f"{target_dir}/v{__version__}/{source_name}/DL1", exist_ok=True)
        dl1_dir = str(f"{target_dir}/v{__version__}/{source_name}/DL1")
    else:

        dl1_dir = str(f"{target_dir}/v{__version__}/{source_name}/DL1")
        if not os.path.exists(f"{target_dir}/v{__version__}/{source_name}"):
            os.makedirs(
                f"{target_dir}/v{__version__}/{source_name}/DL1",
                exist_ok=True,
            )

        else:
            overwrite = input(
                f'data directory for {target_dir.split("/")[-1]} already exists. Would you like to overwrite it? [only "y" or "n"]: '
            )
            if overwrite == "y":
                os.system(f"rm -r {target_dir}/v{__version__}/{source_name}")
                os.makedirs(
                    f"{target_dir}/v{__version__}/{source_name}/DL1",
                    exist_ok=True,
                )

            else:
                print("Directory not modified.")

    ###########################################
    # MAGIC
    ###########################################
    for i in MAGIC_runs:
        for magic in [1, 2]:
            if telescope_ids[magic - 3] > 0:
                os.makedirs(f"{dl1_dir}/M{magic}/{i[0]}/{i[1]}/logs", exist_ok=True)


def directories_generator_MC(target_dir, telescope_ids):

    """
    Here we create all subdirectories for a given workspace and target name.

    Parameters
    ----------
    target_dir : str
        Directory to store the results
    telescope_ids : list
        List of the telescope IDs (set by the user)
    """

    dir_list = [
        "gammas",
        "gammadiffuse",
        "electrons",
        "protons",
        "helium",
    ]
    if not os.path.exists(f"{target_dir}/v{__version__}/MC"):
        os.makedirs(f"{target_dir}/v{__version__}/MC/logs", exist_ok=True)
        os.makedirs(f"{target_dir}/v{__version__}/MC/DL1", exist_ok=True)
        for dir in dir_list:
            os.makedirs(
                f"{target_dir}/v{__version__}/MC/DL1/{dir}/logs",
                exist_ok=True,
            )
    else:
        overwrite = input(
            'MC directory already exists. Would you like to overwrite it? [only "y" or "n"]: '
        )
        if overwrite == "y":
            os.system(f"rm -r {target_dir}/v{__version__}/MC")
            os.makedirs(f"{target_dir}/v{__version__}/MC/logs", exist_ok=True)
            for dir in dir_list:
                os.makedirs(
                    f"{target_dir}/v{__version__}/MC/DL1/{dir}/logs",
                    exist_ok=True,
                )
        else:
            print("Directory not modified.")


def main():

    """
    Main function
    """

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
    NSB_match = config["general"]["NSB_matching"]
    config_file = config["general"]["base_config_file"]

    MC_gammas = config["directories"]["MC_gammas"]
    MC_electrons = config["directories"]["MC_electrons"]
    MC_helium = config["directories"]["MC_helium"]
    MC_protons = config["directories"]["MC_protons"]
    MC_gammadiff = config["directories"]["MC_gammadiff"]
    focal_length = config["general"]["focal_length"]
    source_in = config["data_selection"]["source_name_database"]
    source = config["data_selection"]["source_name_output"]
    cluster = config["general"]["cluster"]
    target_dir = Path(config["directories"]["workspace_dir"])

    if source_in is None:
        source_list = joblib.load("list_sources.dat")

    else:
        if source is None:
            source = source_in
        source_list = [source]
    
    if not NSB_match:
        # Below we run the analysis on the MC data
        if (args.analysis_type == "onlyMC") or (args.analysis_type == "doEverything"):
            directories_generator_MC(
                str(target_dir), telescope_ids
            )  # Here we create all the necessary directories in the given workspace and collect the main directory of the target
            config_file_gen(target_dir, NSB_match, "MC", config_file)  # TODO: fix here
            to_process = {
                "gammas": MC_gammas,
                "electrons": MC_electrons,
                "helium": MC_helium,
                "protons": MC_protons,
                "gammadiffuse": MC_gammadiff,
            }
            for particle in to_process.keys():
                lists_and_bash_generator(
                    particle,
                    target_dir,
                    to_process[particle],
                    focal_length,
                    env_name,
                    cluster,
                )
                list_of_MC = glob.glob(f"linking_MC_{particle}_*.sh")
                if len(list_of_MC) < 2:
                    logger.warning(
                        f"No bash script has been produced for processing {particle}"
                    )
                else:
                    launch_jobs_MC = f"linking=$(sbatch --parsable linking_MC_{particle}_paths.sh) && running=$(sbatch --parsable --dependency=afterany:$linking linking_MC_{particle}_paths_r.sh)"
                    os.system(launch_jobs_MC)
            # Here we do the MC DL0 to DL1 conversion:

    for source_name in source_list:
        if (
            (args.analysis_type == "onlyMAGIC")
            or (args.analysis_type == "doEverything")
            or (NSB_match)
        ):

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
                str(target_dir), telescope_ids, MAGIC_runs, NSB_match, source_name
            )  # Here we create all the necessary directories in the given workspace and collect the main directory of the target
            config_file_gen(
                target_dir, NSB_match, source_name, config_file
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
