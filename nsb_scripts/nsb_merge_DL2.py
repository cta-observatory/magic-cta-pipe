"""
This script apply the "lst1_magic_create_irf.py"
to the DL2 MC gamma files and saves the results at:
WorkingDirectory/IRF

Usage:
$ python IRF.py

"""
import argparse
import os
import numpy as np
import glob
import yaml
import logging
from pathlib import Path
from magicctapipe import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def MergeDL2(scripts_dir, target_dir, nsb, source):
    """
    This function creates the bash scripts to run merge_hdf_files.py in all DL2 subruns.

    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """

    process_name = "DL3_" + target_dir.split("/")[-2:][1]
    ST_list = [
        os.path.basename(x)
        for x in glob.glob(f"{target_dir}/v{__version__}/DL2/*")
    ]
    print(ST_list)
    for p in ST_list:

        DL2_dir = target_dir+f"/v{__version__}/DL2/"+ str(p)+ "/NSB"+ str(nsb)
        list_of_nights = np.sort(glob.glob(DL2_dir + "/20*"))
        print(DL2_dir)

        f = open(f"{source}_MergeDL2_{nsb}_{p}.sh", "w")
        f.write("#!/bin/sh\n\n")
        f.write("#SBATCH -p short\n")
        f.write("#SBATCH -J " + process_name + "\n")
        f.write("#SBATCH -N 1\n\n")
        f.write("ulimit -l unlimited\n")
        f.write("ulimit -s unlimited\n")
        f.write("ulimit -a\n\n")

        for night in list_of_nights:
            if not os.path.exists(night + "/Merged"):
                os.mkdir(night + "/Merged")
            if not os.path.exists(night + "/Merged/logs"):
                os.mkdir(night + "/Merged/logs")
            f.write(f"export LOG={night}/Merged/logs/merge.log\n")
            f.write(
                f"conda run -n magic-lst python {scripts_dir}/merge_hdf_files.py --input-dir {night} --output-dir {night}/Merged --run-wise >$LOG 2>&1\n"
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

    target_dir = str(
        Path(config["directories"]["workspace_dir"])
        / config["directories"]["target_name"]
    )

    scripts_dir = str(Path(config["directories"]["scripts_dir"]))
    source = config["directories"]["target_name"]

    listnsb = np.sort(glob.glob(f"{source}_LST_*_.txt"))
    nsb = []
    for f in listnsb:
        nsb.append(f.split("_")[2])

    print("nsb", nsb)
    for nsblvl in nsb:
        print("***** Merging DL2 files run-wise...")
        MergeDL2(scripts_dir, target_dir, nsblvl, source)

    # Below we run the bash scripts to perform the DL1 to DL2 cnoversion:
    list_of_DL1_to_2_scripts = np.sort(glob.glob(f"{source}_MergeDL2_*.sh"))
    print(list_of_DL1_to_2_scripts)
    if len(list_of_DL1_to_2_scripts) < 1:
        return
    for n, run in enumerate(list_of_DL1_to_2_scripts):
        if n == 0:
            launch_jobs = f"conversion{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = (
                launch_jobs
                + f" && conversion{n}=$(sbatch --parsable --dependency=afterany:$conversion{n-1} {run})"
            )

    # print(launch_jobs)
    os.system(launch_jobs)


if __name__ == "__main__":
    main()
