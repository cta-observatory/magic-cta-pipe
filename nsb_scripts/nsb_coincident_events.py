"""
This scripts facilitates the usage of the script
"lst1_magic_event_coincidence.py". This script is
more like a "manager" that organizes the analysis
process by:
1) Creating the bash scripts for looking for
coincidence events between MAGIC and LST in each
night.
2) Creating the subdirectories for the coincident
event files.


Usage:
$ python coincident_events.py (-c config_file.yaml)


"""
from pathlib import Path
import argparse
import os
from datetime import timedelta
from datetime import date as dtdt
import numpy as np
from magicctapipe import __version__
import glob
import yaml
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configfile_coincidence(ids, target_dir):
    """
    This function creates the configuration file needed for the event coincidence step

    Parameters
    ----------
    ids: list
        list of telescope IDs
    target_dir: str
        Path to the working directory
    """

    f = open(target_dir + "/config_coincidence.yaml", "w")
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
        'event_coincidence:\n    timestamp_type_lst: "dragon_time"  # select "dragon_time", "tib_time" or "ucts_time"\n    pre_offset_search: true\n    n_pre_offset_search_events: 100\n    window_half_width: "300 ns"\n'
    )
    f.write('    time_offset:\n        start: "-10 us"\n        stop: "0 us"\n')
    f.close()


def linking_bash_lst(scripts_dir, target_dir, LST_runs, nsb, date, source):
    """
    This function links the LST data paths to the working directory and creates bash scripts.
    Parameters
    ----------
    scripts_dir: str
        Path to the scripts directory
    target_dir: str
        Path to the working directory
    LST_runs: matrix of strings
        This matrix is imported from config_general.yaml and tells the function where to find the LST data and link them to our working directory
    nsb: int
        NSB level
    date:
        Array of lists [date run] for all the LST runs (no NSB splitting)
    """

    coincidence_DL1_dir = target_dir + f"/v{__version__}"
    if not os.path.exists(coincidence_DL1_dir + "/DL1Coincident/"):
        os.mkdir(f"{coincidence_DL1_dir}/DL1Coincident")
    ST_list = [
        os.path.basename(x) for x in glob.glob(f"{target_dir}/v{__version__}/DL1/*")
    ]

    if (len(LST_runs) == 2) and (len(LST_runs[0]) == 10):
        LST = LST_runs

        LST_runs = []
        LST_runs.append(LST)

    if (len(date) == 2) and (len(date[0]) == 10):
        dt = date
        date = []
        date.append(dt)

    for p in ST_list:
        MAGIC_DL1_dir = target_dir + f"/v{__version__}" + "/DL1/" + p
        if not os.path.exists(coincidence_DL1_dir + "/DL1Coincident/" + str(p)):
            os.mkdir(f"{coincidence_DL1_dir}/DL1Coincident/{p}")
        dates = [os.path.basename(x) for x in glob.glob(f"{MAGIC_DL1_dir}/M1/*")]
        for d in dates:
            Y_M = int(d.split("_")[0])
            M_M = int(d.split("_")[1])
            D_M = int(d.split("_")[2])

            day_MAGIC = dtdt(Y_M, M_M, D_M)

            delta = timedelta(days=1)
            for i in LST_runs:
                Y_L = i[0].split("_")[0]
                M_L = i[0].split("_")[1]
                D_L = i[0].split("_")[2]
                day_LST = dtdt(int(Y_L), int(M_L), int(D_L))
                if day_MAGIC == day_LST + delta:
                    if not os.path.exists(
                        coincidence_DL1_dir
                        + "/DL1Coincident/"
                        + str(p)
                        + "/NSB"
                        + str(nsb)
                    ):
                        os.mkdir(
                            f"{coincidence_DL1_dir}/DL1Coincident/{p}"
                            + "/NSB"
                            + str(nsb)
                        )

                    lstObsDir = (
                        i[0].split("_")[0] + i[0].split("_")[1] + i[0].split("_")[2]
                    )

                    inputdir = f"/fefs/aswg/data/real/DL1/{lstObsDir}/v0.9/tailcut84"
                    if not os.path.exists(
                        coincidence_DL1_dir
                        + "/DL1Coincident/"
                        + str(p)
                        + "/NSB"
                        + str(nsb)
                        + "/"
                        + lstObsDir
                    ):
                        os.mkdir(
                            f"{coincidence_DL1_dir}/DL1Coincident/{p}"
                            + "/NSB"
                            + str(nsb)
                            + "/"
                            + lstObsDir
                        )
                    if not os.path.exists(
                        coincidence_DL1_dir
                        + "/DL1Coincident/"
                        + str(p)
                        + "/NSB"
                        + str(nsb)
                        + "/"
                        + lstObsDir
                        + "/logs"
                    ):
                        os.mkdir(
                            f"{coincidence_DL1_dir}/DL1Coincident/{p}"
                            + "/NSB"
                            + str(nsb)
                            + "/"
                            + lstObsDir
                            + "/logs"
                        )

                    outputdir = (
                        f"{coincidence_DL1_dir}/DL1Coincident/{p}"
                        + "/NSB"
                        + str(nsb)
                        + "/"
                        + lstObsDir
                    )
                    list_of_subruns = np.sort(
                        glob.glob(f"{inputdir}/dl1*Run*{i[1]}*.*.h5")
                    )
                    if os.path.exists(f"{outputdir}/logs/list_LST.txt"):
                        with open(
                            f"{outputdir}/logs/list_LST.txt", "a"
                        ) as LSTdataPathFile:
                            for subrun in list_of_subruns:
                                LSTdataPathFile.write(
                                    subrun + "\n"
                                )  # If this files already exists, simply append the new information
                    else:
                        f = open(
                            f"{outputdir}/logs/list_LST.txt", "w"
                        )  # If the file list_LST.txt does not exist, it will be created here
                        for subrun in list_of_subruns:
                            f.write(subrun + "\n")
                        f.close()

                    if not os.path.exists(outputdir + "/logs/list_LST.txt"):
                        continue
                    process_size = (
                        len(
                            np.genfromtxt(outputdir + "/logs/list_LST.txt", dtype="str")
                        )
                        - 1
                    )

                    if process_size < 0:
                        continue
                    f = open(
                        f"{source}_LST_coincident_{nsb}_{outputdir.split('/')[-1]}.sh",
                        "w",
                    )
                    f.write("#!/bin/sh\n\n")
                    f.write("#SBATCH -p short\n")
                    f.write(
                        "#SBATCH -J "
                        + target_dir.split("/")[-2:][1]
                        + "_coincidence"
                        + str(nsb)
                        + "\n"
                    )
                    f.write(f"#SBATCH --array=0-{process_size}\n")
                    f.write("#SBATCH --mem=30g\n")
                    f.write("#SBATCH -N 1\n\n")
                    f.write("ulimit -l unlimited\n")
                    f.write("ulimit -s unlimited\n")
                    f.write("ulimit -a\n\n")

                    f.write(
                        f"export INM={MAGIC_DL1_dir}/Merged/Merged_{str(Y_M).zfill(4)}_{str(M_M).zfill(2)}_{str(D_M).zfill(2)}\n"
                    )
                    f.write(f"export OUTPUTDIR={outputdir}\n")
                    f.write("SAMPLE_LIST=($(<$OUTPUTDIR/logs/list_LST.txt))\n")
                    f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
                    f.write(
                        "export LOG=$OUTPUTDIR/logs/coincidence_${SLURM_ARRAY_TASK_ID}.log\n"
                    )
                    f.write(
                        f"conda run -n magic-lst python {scripts_dir}/lst1_magic_event_coincidence.py --input-file-lst $SAMPLE --input-dir-magic $INM --output-dir $OUTPUTDIR --config-file {target_dir}/config_coincidence.yaml >$LOG 2>&1"
                    )
                    f.close()


def main():
    """
    Here we read the config file and call the functions defined above.
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
    source = config["directories"]["target_name"]
    print("***** Generating file config_coincidence.yaml...")
    print("***** This file can be found in ", target_dir)
    configfile_coincidence(telescope_ids, target_dir)
    nsb = config["general"]["nsb"]
    runs_all = config["general"]["LST_runs"]
    date = np.genfromtxt(runs_all, dtype=str, delimiter=",")
    for nsblvl in nsb:
        try:
            LST_runs = np.genfromtxt(
                f"{source}_LST_{nsblvl}_.txt", dtype=str, delimiter=","
            )

            print("***** Linking the paths to LST data files...")

            print("***** Generating the bashscript...")
            # bash_coincident(scripts_dir, target_dir, nsblvl)
            linking_bash_lst(
                scripts_dir, target_dir, LST_runs, nsblvl, date, source
            )  # linking the data paths to current working directory

            print("***** Submitting processess to the cluster...")
            print(
                "Process name: "
                + target_dir.split("/")[-2:][1]
                + "_coincidence"
                + str(nsblvl)
            )
            print(
                "To check the jobs submitted to the cluster, type: squeue -n "
                + target_dir.split("/")[-2:][1]
                + "_coincidence"
                + str(nsblvl)
            )

            # Below we run the bash scripts to find the coincident events
            list_of_coincidence_scripts = np.sort(
                glob.glob(f"{source}_LST_coincident_{nsblvl}*.sh")
            )
            if len(list_of_coincidence_scripts) < 1:
                continue
            for n, run in enumerate(list_of_coincidence_scripts):
                if n == 0:
                    launch_jobs = f"coincidence{n}=$(sbatch --parsable {run})"
                else:
                    launch_jobs = (
                        launch_jobs + f" && coincidence{n}=$(sbatch --parsable {run})"
                    )

            # print(launch_jobs)
            os.system(launch_jobs)

        except OSError as exc:
            print(exc)


if __name__ == "__main__":
    main()
