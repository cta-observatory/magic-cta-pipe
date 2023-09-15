"""
This scripts facilitates the usage of the script
"lst1_magic_dl2_to_dl3.py". This script is
more like a "maneger" that organizes the analysis
process by:
1) Creating the bash scripts for looking for
applying "lst1_magic_dl2_to_dl3.py"
2) Creating the subdirectories for the DL3
data files.


Usage:
$ python DL2_to_DL3.py

"""
import argparse
import os
import numpy as np
import glob
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configuration_DL3(ids, target_dir, target_coords):
    """
    This function creates the configuration file needed for the DL2 to DL3 conversion

    Parameters
    ----------
    ids: list
        list of telescope IDs
    target_dir: str
        Path to the working directory
    """

    target_name = target_dir.split("/")[-1]

    f = open(target_dir + "/config_DL3.yaml", "w")
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
        f'dl2_to_dl3:\n    interpolation_method: "linear"  # select "nearest", "linear" or "cubic"\n    interpolation_scheme: "cosZdAz" # select "cosZdAz" or "cosZd"\n    max_distance: "45. deg"\n    source_name: "{target_name}"\n    source_ra: "{target_coords[0]} deg" # used when the source name cannot be resolved\n    source_dec: "{target_coords[1]} deg" # used when the source name cannot be resolved\n\n'
    )

    f.close()


def DL2_to_DL3(scripts_dir, target_dir, nsb):
    """
    This function creates the bash scripts to run lst1_magic_dl2_to_dl3.py on the real data.

    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """

    if not os.path.exists(target_dir + "/DL3"):
        os.mkdir(target_dir + "/DL3")

    process_name = "DL3_" + target_dir.split("/")[-2:][1] + str(nsb)
    output = target_dir + "/DL3"
    nights = np.sort(glob.glob(target_dir + f"/DL2/Observations/{nsb}/*"))
    IRF_dir = (
        "/fefs/aswg/workspace/elisa.visentin/MAGIC_LST_analysis/PG1553_nsb/IRF/"
        + str(nsb)
    )  # IRF direcctory

    for night in nights:
        listOfDL2files = np.sort(glob.glob(night + "/Merged/*.h5"))

        f = open(f'DL3_{nsb}_{night.split("/")[-1]}.sh', "w")
        f.write("#!/bin/sh\n\n")
        f.write("#SBATCH -p short\n")
        f.write("#SBATCH -J " + process_name + "\n")
        f.write("#SBATCH --mem=10g\n")
        f.write("#SBATCH -N 1\n\n")
        f.write("ulimit -l unlimited\n")
        f.write("ulimit -s unlimited\n")
        f.write("ulimit -a\n\n")

        for DL2_file in listOfDL2files:
            f.write(f'export LOG={output}/DL3_{DL2_file.split("/")[-1]}.log\n')
            f.write(
                f"conda run -n magic-lst python {scripts_dir}/lst1_magic_dl2_to_dl3.py --input-file-dl2 {DL2_file} --input-dir-irf {IRF_dir} --output-dir {output} --config-file {target_dir}/config_DL3.yaml >$LOG 2>&1\n\n"
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

    target_dir = str(
        Path(config["directories"]["workspace_dir"])
        / config["directories"]["target_name"]
    )
    scripts_dir = str(Path(config["directories"]["scripts_dir"]))

    target_coords = [
        config["general"]["target_RA_deg"],
        config["general"]["target_Dec_deg"],
    ]

    print("***** Generating file config_DL3.yaml...")
    print("***** This file can be found in ", target_dir)
    configuration_DL3(telescope_ids, target_dir, target_coords)
    source = config["directories"]["target_name"]
    listnsb = np.sort(glob.glob(f"{source}_LST_*_.txt"))
    nsb = []
    for f in listnsb:
        nsb.append(f.split("_")[2])

    print("nsb", nsb)
    for nsblvl in nsb:
        print("***** Generating bashscripts for DL2-DL3 conversion...")
        DL2_to_DL3(scripts_dir, target_dir, nsblvl)

        print("***** Running lst1_magic_dl2_to_dl3.py in the DL2 real data files...")
        print("Process name: DL3_" + target_dir.split("/")[-2:][1] + str(nsblvl))
        print(
            "To check the jobs submitted to the cluster, type: squeue -n DL3_"
            + target_dir.split("/")[-2:][1]
            + str(nsblvl)
        )

        # Below we run the bash scripts to perform the DL1 to DL2 cnoversion:
        list_of_DL2_to_DL3_scripts = np.sort(glob.glob(f"DL3_{nsblvl}_2*.sh"))
        if len(list_of_DL2_to_DL3_scripts) < 1:
            continue
        for n, run in enumerate(list_of_DL2_to_DL3_scripts):
            if n == 0:
                launch_jobs = f"dl3{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = (
                    launch_jobs
                    + f" && dl3{n}=$(sbatch --parsable --dependency=afterany:$dl3{n-1} {run})"
                )

        os.system(launch_jobs)


if __name__ == "__main__":
    main()
