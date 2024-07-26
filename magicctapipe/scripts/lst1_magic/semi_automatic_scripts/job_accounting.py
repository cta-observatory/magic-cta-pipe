"""
This script does checks of status of jobs based on the log files generated during the execution.
It also does accounting of memory and CPU usage
It loads the config_general file to figure out what files it should look for and processes source name and time range
For the moment it ignores date_list and skip_*_runs
"""
import argparse
import glob
import os
import re
from datetime import datetime, timedelta
from subprocess import PIPE, run

import numpy as np
import yaml

from magicctapipe import __version__

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
ENDC = "\033[0m"


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_general.yaml",
        help="Path to a configuration file config_general.yaml",
    )

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

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

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
    
    if args.data_level == 'MergedStereo':
        dirs = sorted(glob.glob(f'{work_dir}/v{args.version}/{source_out}/DL1Stereo/[0-9]*/Merged'))
    
    else: 
        dirs = sorted(
            glob.glob(f"{indir}/[0-9]*/[M0-9]*")
            + glob.glob(f"{indir}/Merged_[0-9]*")
            + glob.glob(f"{indir}/" + "[0-9]" * 8)
        )
    
    if dirs == []:
        versions = [x.split("/v")[-1] for x in glob.glob(f"{work_dir}/v*")]
        print("Error, no directories found")
        print(f"for path {work_dir} found in {args.config_file} this is available")
        print(f"Versions {versions}")

        print(
            "Supported data types: DL1/M1, DL1/M2, DL1/Merged, DL1Coincident, DL1Stereo, MergedStereo"
        )
        exit(1)

    if timerange:
        timemin = str(config["data_selection"]["min"])
        timemax = str(config["data_selection"]["max"])
        timemin = datetime.strptime(timemin, "%Y_%m_%d")
        timemax = datetime.strptime(timemax, "%Y_%m_%d")

    all_todo = 0
    all_return = 0
    all_good = 0
    all_cpu = []
    all_mem = []
    total_time = 0
    all_jobs = []
    for dir in dirs:
        if args.data_level == 'MergedStereo':
            this_date=dir.split('/')[-2]
        else:
            this_date = re.sub(f".+/{args.data_level}/", "", dir)
            this_date = re.sub(r"\D", "", this_date.split("/")[0])
        this_date = datetime.strptime(this_date, "%Y%m%d")
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
        else:
            print(f"{RED}No {ins} files {ENDC}")
            this_todo = 0

        list_return = f"{dir}/logs/list_return.log"
        this_good = 0
        this_cpu = []
        this_mem = []
        try:
            with open(list_return, "r") as fp:
                returns = fp.readlines()
                this_return = len(returns)
                for line in returns:
                    line = line.split()
                    file_in = line[0]
                    slurm_id = f"{line[1]}_{line[2]}" if len(line) == 4 else line[1]
                    rc = line[-1]
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
                                    days=hh // 24, hours=hh % 24, minutes=mm, seconds=ss
                                )
                                if slurm_id not in all_jobs:
                                    total_time += delta.total_seconds() / 3600
                                    all_jobs += [slurm_id]
                                this_cpu.append(delta)
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
        print(
            f"CPU: median={np.median(all_cpu)}, max={all_cpu.max()}, total={total_time:.2f} CPU hrs; memory [M]: median={np.median(all_mem)}, max={all_mem.max()}"
        )


if __name__ == "__main__":
    main()
