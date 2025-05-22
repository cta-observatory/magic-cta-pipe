"""
This script does checks of status of jobs based on the log files generated during the execution.
It also does accounting of memory and CPU usage
It loads the config_auto_MCP file to figure out what files it should look for and processes source name and time range
For the moment it ignores date_list and skip_*_runs

It can also update the h5 file with the list of runs to process
"""
import glob
import json
import os
import re
from datetime import datetime, timedelta
from subprocess import PIPE, run

import numpy as np
import pandas as pd
import yaml

from magicctapipe import __version__
from magicctapipe.utils import auto_MCP_parser

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
ENDC = "\033[0m"

__all__ = ["run_shell"]


def run_shell(command):
    """
    Simple function to extract the output of a command run in a shell

    Parameters
    ----------
    command : str
        Command to be executed

    Returns
    ----------
    list
        List of lines returned by the program
    """
    result = run(command, stdout=PIPE, stderr=PIPE, shell=True, universal_newlines=True)
    return result.stdout


def main():
    """
    Function counts the number of jobs that should have been submitted, and checks the output of the logs to see how many finished successfully, and how many failed.
    """
    parser = auto_MCP_parser()

    parser.add_argument(
        "--data-level",
        "-d",
        dest="data_level",
        type=str,
        default="DL1/M1",
        help="Data level to be checked",
    )

    parser.add_argument(
        "--version",
        "-v",
        dest="version",
        type=str,
        default=__version__,
        help="MCP version (used for subdirectory name)",
    )

    parser.add_argument(
        "--no-accounting",
        action="store_true",
        help="No CPU/Memory usage check (faster)",
    )

    parser.add_argument(
        "--run-list-file",
        "-r",
        dest="run_list",
        type=str,
        default=None,
        help="h5 file with run list",
    )

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    if args.run_list is not None:
        try:
            h5key = "joint_obs"
            run_key = "LST1_run"
            ismagic = False
            for magic in [1, 2]:
                if args.data_level[-2:] == f"M{magic}":
                    h5key = f"MAGIC{magic}/runs_M{magic}"
                    run_key = "Run ID"
                    ismagic = True

            h5runs = pd.read_hdf(args.run_list, key=h5key)
        except (FileNotFoundError, KeyError):
            print(f"Cannot open {h5key} in  {args.run_list}")
            exit(1)

        rc_col = "DL1_rc" if ismagic else args.data_level + "_rc"

        if rc_col not in h5runs.keys():
            h5runs[rc_col] = "{}"
            h5runs[rc_col + "_all"] = None

        rc_dicts = {}
        for rrun, dct in np.array(h5runs[[run_key, rc_col]]):
            rc_dicts[rrun] = json.loads(dct)

    # TODO: those variables will be needed when more features are implemented
    source_out = config["data_selection"]["source_name_output"]
    timerange = config["data_selection"]["time_range"]

    # skip_LST = config["data_selection"]["skip_LST_runs"]
    # skip_MAGIC = config["data_selection"]["skip_MAGIC_runs"]
    work_dir = config["directories"]["workspace_dir"]

    print(f"Checking progress of jobs stored in {work_dir}")
    if source_out is None:
        source_out = "*"

    indir = f"{work_dir}/v{args.version}/{source_out}/{args.data_level}"

    dirs = [
        x.replace("/logs", "")
        for x in (
            sorted(
                glob.glob(f"{indir}/[0-9]*/[0-9]*/logs")
                + glob.glob(f"{indir}/[0-9]*/logs")
                + glob.glob(f"{indir}/logs")
            )
        )
    ]

    if dirs == []:
        versions = [x.split("/v")[-1] for x in glob.glob(f"{work_dir}/v*")]
        print("Error, no directories found")
        print(f"for path {work_dir} found in {args.config_file} this is available")
        print(f"Versions {versions}")

        print(
            "Supported data types: DL1/M1, DL1/M2, DL1/Merged, DL1Coincident, DL1Stereo, DL1Stereo/Merged, DL2, DL3"
        )
        exit(1)

    if timerange:
        timemin = str(config["data_selection"]["min"])
        timemax = str(config["data_selection"]["max"])
        timemin = datetime.strptime(timemin, "%Y_%m_%d")
        timemax = datetime.strptime(timemax, "%Y_%m_%d")
    if "DL1/" in args.data_level:
        tdelta = timedelta(days=1)
        timemin = timemin + tdelta
        timemax = timemax + tdelta

    all_todo = 0
    all_return = 0
    all_good = 0
    all_cpu = []
    all_mem = []
    total_time = 0
    all_jobs = []

    for dir in dirs:

        this_date_str = re.sub(f".+/{args.data_level}/", "", dir)
        this_date_str = re.sub(r"\D", "", this_date_str.split("/")[0])
        this_date = datetime.strptime(this_date_str, "%Y%m%d")
        if timerange and (this_date < timemin or this_date > timemax):
            continue

        print(dir)
        list_dl0 = ""
        ins = ["list_dl0.txt", "list_LST.txt", "list_coin.txt", "list_cal.txt"]

        for file in ins:
            if os.path.exists(f"{dir}/logs/{file}"):
                list_dl0 = f"{dir}/logs/{file}"

        if list_dl0 != "":
            with open(list_dl0, "r") as fp:
                this_todo = len(fp.readlines())
        elif args.data_level in ["DL2", "DL3"]:

            files = glob.glob(f"{dir}/logs/ST*.txt")
            this_todo = 0
            if len(files) == 0:
                print(f"{RED}No ST* files {ENDC}")
            else:
                for f in files:
                    with open(f, "r") as fp:
                        this_todo += len(fp.readlines())

        else:
            print(f"{RED}No {ins} files {ENDC}")
            this_todo = 0

        list_return = glob.glob(f"{dir}/logs/list_*_return.log")

        this_good = 0
        this_cpu = []
        this_mem = []
        this_return = 0
        try:
            for list in list_return:
                with open(list, "r") as fp:
                    returns = fp.readlines()
                    this_return += len(returns)
                    for line in returns:
                        line = line.split()
                        file_in = line[0]
                        slurm_id = f"{line[1]}_{line[2]}" if len(line) == 4 else line[1]
                        rc = line[-1]

                        if args.run_list is not None:
                            if ismagic:
                                run_subrun = file_in.split("/")[-1].split("_")[2]
                                this_run = int(run_subrun.split(".")[0])
                                this_subrun = int(run_subrun.split(".")[1])
                            else:
                                filename = file_in.split("/")[-1]
                                this_run = filename.split(".")[1].replace("Run", "")
                                this_subrun = int(filename.split(".")[2])

                            rc_dicts[this_run][str(this_subrun)] = rc

                        if rc == "0":
                            this_good += 1
                            # now check accounting
                            if not args.no_accounting:
                                out = run_shell(
                                    f'sacct --format="JobID,CPUTime,MaxRSS" --units=M  -j {slurm_id}| tail -1'
                                ).split()
                                if len(out) == 3:
                                    _, cpu, mem = out
                                elif (
                                    len(out) == 2
                                ):  # MaxRSS sometimes is missing in the output
                                    cpu = out[1]
                                    mem = None
                                else:
                                    print("Unexpected sacct output: {out}")
                                if cpu is not None:
                                    hh, mm, ss = (int(x) for x in str(cpu).split(":"))
                                    delta = timedelta(
                                        days=hh // 24,
                                        hours=hh % 24,
                                        minutes=mm,
                                        seconds=ss,
                                    )
                                    if slurm_id not in all_jobs:
                                        total_time += delta.total_seconds() / 3600
                                        all_jobs += [slurm_id]
                                    np.append(this_cpu, delta)
                                if mem is not None and mem.endswith("M"):
                                    this_mem.append(float(mem[0:-1]))
                                else:
                                    print("Memory usage information is missing")
                        else:
                            print(f"file {file_in} failed with error {rc}")
                    if len(this_cpu) > 0:
                        all_cpu += this_cpu
                        all_mem += this_mem
                        this_cpu = np.array(this_cpu)
                        this_mem = np.array(this_mem)
                        mem_info = (
                            f"memory [M]: median={np.median(this_mem)}, max={this_mem.max()}"
                            if len(this_mem)
                            else ""
                        )
                        print(
                            f"CPU: median={np.median(this_cpu)}, max={this_cpu.max()}; {mem_info}"
                        )

        except IOError:
            print(f"{RED}File {list_return} is missing{ENDC}")
            this_return = 0

        all_todo += this_todo
        all_return += this_return
        all_good += this_good
        if this_good < this_return:
            status = RED  # there are errors in processing
        elif this_return < this_todo:
            status = YELLOW  # waiting for jobs to finish (or lost jobs)
        else:
            status = GREEN  # all done and ready

        if this_todo > 0:
            print(
                f"{status}to do: {this_todo}, finished: {this_return}, no errors: {this_good}{ENDC}"
            )

    print("\nSUMMARY")
    if all_good < all_return:
        status = RED  # there are errors in processing
    elif all_return < all_todo:
        status = YELLOW  # waiting for jobs to finish (or lost jobs)
    else:
        status = GREEN  # all done and ready

    if all_todo > 0:
        print(
            f"{status}to do: {all_todo}, finished: {all_return}, no errors: {all_good}{ENDC}"
        )

    if len(all_cpu) > 0:
        all_cpu = np.array(all_cpu)
        all_mem = np.array(all_mem)
        max_mem = all_mem.max() if len(all_mem) else np.nan
        max_cpu = all_cpu.max() if len(all_cpu) else np.nan
        print(
            f"CPU: median={np.median(all_cpu)}, max={max_cpu}, total={total_time:.2f} CPU hrs; memory [M]: median={np.median(all_mem)}, max={max_mem}"
        )

    if args.run_list is not None:
        print("Updating the database")
        for rrun in rc_dicts.keys():
            idx = h5runs[run_key] == rrun
            h5runs.loc[idx, rc_col] = json.dumps(rc_dicts[rrun])
            if ismagic:
                all_subruns = np.array(h5runs[idx]["number of subruns"])[0]
            else:
                all_subruns = len(rc_dicts[rrun])
            good_subruns = sum(np.array(list(rc_dicts[rrun].values())) == "0")
            isgood = np.logical_and(good_subruns == all_subruns, good_subruns > 0)
            h5runs.loc[idx, rc_col + "_all"] = isgood

        # fixme: for DL1/M[12] files since htere are two dataframes in the file, we need to append it
        # and this causes increase in the file size every time the file is updated
        h5runs.to_hdf(args.run_list, key=h5key, mode="r+")


if __name__ == "__main__":
    main()
