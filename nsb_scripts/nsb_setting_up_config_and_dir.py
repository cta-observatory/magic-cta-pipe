####TODO: DIRECTORY STRUCTURE, ST match
"""
This script facilitates the usage of 
"magic_calib_to_dl1.py". This script is more like a
"manager" that organizes the analysis process by:
1) Creating the necessary directories and subdirectories.
2) Generatign all the bash script files that convert the
MAGIC files from DL0 to DL1.
3) Launching these jobs in the IT container.

Notice that in this stage we only use MAGIC data.
No LST data is used here.

Standard usage:
$ python setting_up_config_and_dir.py (-c config_file.yaml)



"""
import argparse
import os
import numpy as np
from magicctapipe import __version__
import glob
import time
import yaml
from pathlib import Path


ST_list = ["ST0320A", "ST0319A", "ST0318A", "ST0317A", "ST0316A"]
ST_begin = ["2023_03_10", "2022_12_15", "2022_06_10", "2021_12_30", "2020_10_24"]
ST_end = [
    "2024_01_01",
    "2023_03_09",
    "2022_08_31",
    "2022_06_09",
    "2021_09_29",
]  # ST0320 ongoing -> 'service' end date


def config_file_gen(ids, target_dir):
    """
    Here we create the configuration file needed for transforming DL0 data in DL1
    """

    f = open(target_dir + "/config_step1.yaml", "w")
    f.write("directories:\n    target: " + target_dir + "\n\n")
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
        'LST:\n    image_extractor:\n        type: "LocalPeakWindowSum"\n        window_shift: 4\n        window_width: 8\n\n'
    )
    f.write(
        "    increase_nsb:\n        use: true\n        extra_noise_in_dim_pixels: 1.27\n        extra_bias_in_dim_pixels: 0.665\n        transition_charge: 8\n        extra_noise_in_bright_pixels: 2.08\n\n"
    )
    f.write("    increase_psf:\n        use: false\n        fraction: null\n\n")
    f.write(
        "    tailcuts_clean:\n        picture_thresh: 8\n        boundary_thresh: 4\n        keep_isolated_pixels: false\n        min_number_picture_neighbors: 2\n\n"
    )
    f.write(
        "    time_delta_cleaning:\n        use: true\n        min_number_neighbors: 1\n        time_limit: 2\n\n"
    )
    f.write(
        "    dynamic_cleaning:\n        use: true\n        threshold: 267\n        fraction: 0.03\n\n    use_only_main_island: false\n\n"
    )

    f.write(
        'MAGIC:\n    image_extractor:\n        type: "SlidingWindowMaxSum"\n        window_width: 5\n        apply_integration_correction: false\n\n'
    )
    f.write("    charge_correction:\n        use: true\n        factor: 1.143\n\n")
    f.write(
        '    magic_clean:\n        use_time: true\n        use_sum: true\n        picture_thresh: 6\n        boundary_thresh: 3.5\n        max_time_off: 4.5\n        max_time_diff: 1.5\n        find_hotpixels: true\n        pedestal_type: "from_extractor_rndm"\n\n'
    )
    f.write(
        "    muon_ring:\n        thr_low: 25\n        tailcut: [12, 8]\n        ring_completeness_threshold: 25\n\n"
    )
    f.close()


def lists_and_bash_gen_MAGIC(scripts_dir, target_dir, telescope_ids, MAGIC_runs, source):
    """
    Below we create a bash script that links the MAGIC data paths to each subdirectory.
    """

    process_name = target_dir.split("/")[-2:][0] + target_dir.split("/")[-2:][1]

    f = open(f"{source}_linking_MAGIC_data_paths.sh", "w")
    f.write("#!/bin/sh\n\n")
    f.write("#SBATCH -p short\n")
    f.write("#SBATCH -J " + process_name + "\n")
    f.write("#SBATCH -N 1\n\n")
    f.write("ulimit -l unlimited\n")
    f.write("ulimit -s unlimited\n")
    f.write("ulimit -a\n")

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
                time.strptime(i[0], "%Y_%m_%d") <= time.strptime(ST_end[p], "%Y_%m_%d")
            ):
                if telescope_ids[-1] > 0:
                    f.write(
                        "export IN1=/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/"
                        + i[0].split("_")[0]
                        + "/"
                        + i[0].split("_")[1]
                        + "/"
                        + i[0].split("_")[2]
                        + "\n"
                    )
                    f.write(
                        "export OUT1="
                        + target_dir
                        + f"/v{__version__}"
                        + "/DL1/"
                        + ST_list[p]
                        + "/M2/"
                        + i[0]
                        + "/"
                        + i[1]
                        + "/logs \n"
                    )
                    f.write(
                        "ls $IN1/*" + i[1][-2:] + ".*_Y_*.root > $OUT1/list_dl0.txt\n"
                    )

                f.write("\n")
                if telescope_ids[-2] > 0:
                    f.write(
                        "export IN1=/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/"
                        + i[0].split("_")[0]
                        + "/"
                        + i[0].split("_")[1]
                        + "/"
                        + i[0].split("_")[2]
                        + "\n"
                    )
                    f.write(
                        "export OUT1="
                        + target_dir
                        + f"/v{__version__}"
                        + "/DL1/"
                        + ST_list[p]
                        + "/M1/"
                        + i[0]
                        + "/"
                        + i[1]
                        + "/logs \n"
                    )
                    f.write(
                        "ls $IN1/*" + i[1][-2:] + ".*_Y_*.root > $OUT1/list_dl0.txt\n"
                    )

    f.close()

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
                            "/fefs/onsite/common/MAGIC/data/M2/event/Calibrated/"
                            + i[0].split("_")[0]
                            + "/"
                            + i[0].split("_")[1]
                            + "/"
                            + i[0].split("_")[2]
                            + f"/*{i[1]}.*_Y_*.root"
                        )
                        number_of_nodes = len(number_of_nodes) - 1
                        if number_of_nodes < 0:
                            continue
                        f = open(f"{source}_MAGIC-II_dl0_to_dl1_run_{i[1]}.sh", "w")
                        f.write("#!/bin/sh\n\n")
                        f.write("#SBATCH -p long\n")
                        f.write("#SBATCH -J " + process_name + "\n")
                        f.write("#SBATCH --array=0-" + str(number_of_nodes) + "\n")
                        f.write("#SBATCH -N 1\n\n")
                        f.write("ulimit -l unlimited\n")
                        f.write("ulimit -s unlimited\n")
                        f.write("ulimit -a\n\n")

                        f.write(
                            "export OUTPUTDIR="
                            + target_dir
                            + f"/v{__version__}"
                            + "/DL1/"
                            + ST_list[p]
                            + "/M2/"
                            + i[0]
                            + "/"
                            + i[1]
                            + "\n"
                        )

                        f.write("SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_dl0.txt))\n")
                        f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n")

                        f.write(
                            "export LOG=$OUTPUTDIR/logs/real_0_1_task${SLURM_ARRAY_TASK_ID}.log\n"
                        )
                        f.write(
                            f"conda run -n magic-lst python {scripts_dir}/magic_calib_to_dl1.py --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file "
                            + target_dir
                            + "/config_step1.yaml >$LOG 2>&1\n"
                        )
                        f.close()

                    if telescope_ids[-2] > 0:
                        number_of_nodes = glob.glob(
                            "/fefs/onsite/common/MAGIC/data/M1/event/Calibrated/"
                            + i[0].split("_")[0]
                            + "/"
                            + i[0].split("_")[1]
                            + "/"
                            + i[0].split("_")[2]
                            + f"/*{i[1]}.*_Y_*.root"
                        )
                        number_of_nodes = len(number_of_nodes) - 1
                        if number_of_nodes < 0:
                            continue
                        f = open(f"{source}_MAGIC-I_dl0_to_dl1_run_{i[1]}.sh", "w")
                        f.write("#!/bin/sh\n\n")
                        f.write("#SBATCH -p long\n")
                        f.write("#SBATCH -J " + process_name + "\n")
                        f.write("#SBATCH --array=0-" + str(number_of_nodes) + "\n")
                        f.write("#SBATCH -N 1\n\n")
                        f.write("ulimit -l unlimited\n")
                        f.write("ulimit -s unlimited\n")
                        f.write("ulimit -a\n\n")

                        f.write(
                            "export OUTPUTDIR="
                            + target_dir
                            + f"/v{__version__}"
                            + "/DL1/"
                            + ST_list[p]
                            + "/M1/"
                            + i[0]
                            + "/"
                            + i[1]
                            + "\n"
                        )
                        f.write("cd " + target_dir + "/../\n")
                        f.write("SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_dl0.txt))\n")
                        f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n\n")

                        f.write(
                            "export LOG=$OUTPUTDIR/logs/real_0_1_task${SLURM_ARRAY_TASK_ID}.log\n"
                        )
                        f.write(
                            f"conda run -n magic-lst python {scripts_dir}/magic_calib_to_dl1.py --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file "
                            + target_dir
                            + "/config_step1.yaml >$LOG 2>&1\n"
                        )
                        f.close()


def directories_generator(target_dir, telescope_ids, MAGIC_runs):
    """
    Here we create all subdirectories for a given workspace and target name.
    """



    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not os.path.exists(target_dir + f"/v{__version__}"):
        os.mkdir(target_dir + f"/v{__version__}")
    if not os.path.exists(target_dir + f"/v{__version__}" + "/DL1"):
        os.mkdir(target_dir + f"/v{__version__}" + "/DL1")
    dl1_dir = str(target_dir + f"/v{__version__}" + "/DL1")

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
                time.strptime(i[0], "%Y_%m_%d") <= time.strptime(ST_end[p], "%Y_%m_%d")
            ):
                if telescope_ids[-1] > 0:
                    if not os.path.exists(dl1_dir + f"/{ST_list[p]}"):
                        os.mkdir(dl1_dir + f"/{ST_list[p]}")
                    if not os.path.exists(dl1_dir + f"/{ST_list[p]}/M2"):
                        os.mkdir(dl1_dir + f"/{ST_list[p]}/M2")
                    if not os.path.exists(dl1_dir + f"/{ST_list[p]}/M2/" + i[0]):
                        os.mkdir(dl1_dir + f"/{ST_list[p]}/M2/" + i[0])

                    if not os.path.exists(
                        dl1_dir + f"/{ST_list[p]}/M2/" + i[0] + "/" + i[1]
                    ):
                        os.mkdir(dl1_dir + f"/{ST_list[p]}/M2/" + i[0] + "/" + i[1])
                    if not os.path.exists(
                        dl1_dir + f"/{ST_list[p]}/M2/" + i[0] + "/" + i[1] + "/logs"
                    ):
                        os.mkdir(
                            dl1_dir + f"/{ST_list[p]}/M2/" + i[0] + "/" + i[1] + "/logs"
                        )
                if telescope_ids[-2] > 0:
                    if not os.path.exists(dl1_dir + f"/{ST_list[p]}"):
                        os.mkdir(dl1_dir + f"/{ST_list[p]}")
                    if not os.path.exists(dl1_dir + f"/{ST_list[p]}/M1"):
                        os.mkdir(dl1_dir + f"/{ST_list[p]}/M1")
                    if not os.path.exists(dl1_dir + f"/{ST_list[p]}/M1/" + i[0]):
                        os.mkdir(dl1_dir + f"/{ST_list[p]}/M1/" + i[0])

                    if not os.path.exists(
                        dl1_dir + f"/{ST_list[p]}/M1/" + i[0] + "/" + i[1]
                    ):
                        os.mkdir(dl1_dir + f"/{ST_list[p]}/M1/" + i[0] + "/" + i[1])
                    if not os.path.exists(
                        dl1_dir + f"/{ST_list[p]}/M1/" + i[0] + "/" + i[1] + "/logs"
                    ):
                        os.mkdir(
                            dl1_dir + f"/{ST_list[p]}/M1/" + i[0] + "/" + i[1] + "/logs"
                        )


def main():
    """Here we read the config file and call the functions to generate the necessary directories, bash scripts and launching the jobs."""

    # Here we are simply collecting the parameters from the command line, as input file, output directory, and configuration file

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

    MAGIC_runs_and_dates = config["general"]["MAGIC_runs"]
    MAGIC_runs = np.genfromtxt(
        MAGIC_runs_and_dates, dtype=str, delimiter=","
    )  # READ LIST OF DATES AND RUNS: format table where each line is like "2020_11_19,5093174"
    target_dir = str(
        Path(config["directories"]["workspace_dir"])
        / config["directories"]["target_name"]
    )
    source=config["directories"]["target_name"]
    scripts_dir = str(Path(config["directories"]["scripts_dir"]))

    print("*** Reducing DL0 to DL1 data***")
    print(
        "Process name: ", target_dir.split("/")[-2:][0] + target_dir.split("/")[-2:][1]
    )
    print(
        "To check the jobs submitted to the cluster, type: squeue -n",
        target_dir.split("/")[-2:][0] + target_dir.split("/")[-2:][1],
    )

    directories_generator(
        target_dir, telescope_ids, MAGIC_runs
    )  # Here we create all the necessary directories in the given workspace and collect the main directory of the target
    config_file_gen(telescope_ids, target_dir)

    lists_and_bash_gen_MAGIC(
        scripts_dir, target_dir, telescope_ids, MAGIC_runs, source
    )  # MAGIC real data
    # If there are MAGIC data, we convert them from DL0 to DL1 here:
    if (telescope_ids[-2] > 0) or (telescope_ids[-1] > 0):
        list_of_MAGIC_runs = glob.glob(f"{source}_MAGIC-*.sh")
        if len(list_of_MAGIC_runs) < 1:
            return
        for n, run in enumerate(list_of_MAGIC_runs):
            if n == 0:
                launch_jobs = f"linking=$(sbatch --parsable {source}_linking_MAGIC_data_paths.sh)  &&  RES{n}=$(sbatch --parsable --dependency=afterany:$linking {run})"
            else:
                launch_jobs = launch_jobs + f" && RES{n}=$(sbatch --parsable {run})"

        os.system(launch_jobs)


if __name__ == "__main__":
    main()

# sbatch linking_MC_gammas_paths_r.sh && sbatch linking_MC_protons_paths_r.sh && sbatch linking_MC_gammadiffuse_paths_r.sh
