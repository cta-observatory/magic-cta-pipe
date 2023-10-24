"""
This script creates the bashscripts necessary to apply "lst1_magic_dl1_stereo_to_dl2.py"
to the DL1 stereo data (real and MC). It also creates new subdirectories associated with
the data level 2. The DL2 files are saved at:
WorkingDirectory/DL2/
and in the subdirectories therein.

Usage:
$ python DL1_to_DL2.py

"""
import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import yaml
from magicctapipe import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def DL1_to_2(scripts_dir, target_dir, nsb, source, config, env_name):
    """
    This function creates the bash scripts to run lst1_magic_dl1_stereo_to_dl2.py.

    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """
    process_name = "DL2_" + target_dir.split("/")[-2:][1] + str(nsb)
    if not os.path.exists(target_dir + f"/v{__version__}/DL2"):
        os.mkdir(target_dir + f"/v{__version__}/DL2")

    ST_list = [
        os.path.basename(x)
        for x in glob.glob(f"{target_dir}/v{__version__}/DL1CoincidentStereo/*")
    ]
    for p in ST_list:
        print("period", p)
        if not os.path.exists(f"{target_dir}/v{__version__}/DL2/" + str(p)):
            os.mkdir(f"{target_dir}/v{__version__}/DL2/" + str(p))

        if (
            not os.path.exists(
                f"{target_dir}/v{__version__}/DL2/" + str(p) + "/NSB" + str(nsb)
            )
        ) and (
            os.path.exists(
                f"{target_dir}/v{__version__}/DL1CoincidentStereo/"
                + str(p)
                + "/NSB"
                + str(nsb)
            )
        ):
            os.mkdir(f"{target_dir}/v{__version__}/DL2/" + str(p) + "/NSB" + str(nsb))
        data_files_dir = (
            target_dir
            + f"/v{__version__}/DL1CoincidentStereo/"
            + str(p)
            + "/NSB"
            + str(nsb)
        )
        RFs_dir = (
            f"/fefs/aswg/workspace/elisa.visentin/MAGIC_LST_analysis/{source}/RF/"
            + str(p)
            + "/NSB"
            + str(nsb)
        )  # then, RFs saved somewhere (as Julian's ones)
        listOfDL1nights = np.sort(glob.glob(data_files_dir + "/*"))
        print(data_files_dir)
        for night in listOfDL1nights:
            output = (
                target_dir + f'/v{__version__}/DL2/{p}/NSB{nsb}/{night.split("/")[-1]}'
            )
            if not os.path.exists(output):
                os.mkdir(output)
            if not os.path.exists(output + "/logs"):
                os.mkdir(output + "/logs")
            listOfDL1Files = np.sort(glob.glob(night + "/*.h5"))
            np.savetxt(
                output + "/logs/list_of_DL1_stereo_files.txt", listOfDL1Files, fmt="%s"
            )
            process_size = len(listOfDL1Files) - 1
            if process_size < 0:
                continue
            with open(f'{source}_DL1_to_DL2_{nsb}_{night.split("/")[-1]}.sh', "w") as f:
                f.write("#!/bin/sh\n\n")
                f.write("#SBATCH -p long\n")
                f.write("#SBATCH -J " + process_name + "\n")
                f.write(f"#SBATCH --array=0-{process_size}%100\n")
                f.write("#SBATCH --mem=30g\n")
                f.write("#SBATCH -N 1\n\n")
                f.write("ulimit -l unlimited\n")
                f.write("ulimit -s unlimited\n")
                f.write("ulimit -a\n\n")

                f.write(
                    f"SAMPLE_LIST=($(<{output}/logs/list_of_DL1_stereo_files.txt))\n"
                )
                f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
                f.write(
                    f"export LOG={output}"
                    + "/logs/DL1_to_DL2_${SLURM_ARRAY_TASK_ID}.log\n"
                )
                f.write(
                    f"conda run -n {env_name} python {scripts_dir}/lst1_magic_dl1_stereo_to_dl2.py --input-file-dl1 $SAMPLE --input-dir-rfs {RFs_dir} --output-dir {output} --config-file {scripts_dir}/{config} >$LOG 2>&1\n\n"
                )


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
    env_name = config["general"]["env_name"]
    scripts_dir = str(Path(config["directories"]["scripts_dir"]))
    source = config["directories"]["target_name"]
    listnsb = np.sort(glob.glob(f"{source}_LST_*_.txt"))
    nsb = []
    for f in listnsb:
        nsb.append(f.split("_")[2])

    print("nsb", nsb)
    for nsblvl in nsb:
        print("***** Generating bashscripts for DL2...")
        DL1_to_2(scripts_dir, target_dir, nsblvl, source, args.config_file, env_name)

        print("***** Running lst1_magic_dl1_stereo_to_dl2.py in the DL1 data files...")
        print("Process name: DL2_" + target_dir.split("/")[-2:][1] + str(nsblvl))
        print(
            "To check the jobs submitted to the cluster, type: squeue -n DL2_"
            + target_dir.split("/")[-2:][1]
            + str(nsblvl)
        )

        # Below we run the bash scripts to perform the DL1 to DL2 cnoversion:
        list_of_DL1_to_2_scripts = np.sort(
            glob.glob(f"{source}_DL1_to_DL2_{nsblvl}*.sh")
        )
        if len(list_of_DL1_to_2_scripts) < 1:
            continue
        print(list_of_DL1_to_2_scripts)
        for n, run in enumerate(list_of_DL1_to_2_scripts):
            if n == 0:
                launch_jobs = f"dl2{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = (
                    launch_jobs
                    + f" && dl2{n}=$(sbatch --parsable --dependency=afterany:$dl2{n-1} {run})"
                )

        # print(launch_jobs)
        os.system(launch_jobs)


if __name__ == "__main__":
    main()
