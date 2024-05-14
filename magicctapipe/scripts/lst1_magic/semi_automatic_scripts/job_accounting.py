"""
This script does checks of status of jobs based on the log files generated during the execution.
It also does accounting of memory and CPU usage
"""
import argparse
import glob
from datetime import timedelta
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
    # source_in = config["data_selection"]["source_name_database"]
    # source_out = config["data_selection"]["source_name_output"]
    # timerange = config["data_selection"]["time_range"]
    # skip_LST = config["data_selection"]["skip_LST_runs"]
    # skip_MAGIC = config["data_selection"]["skip_MAGIC_runs"]
    NSB_matching = config["general"]["NSB_matching"]
    work_dir = config["directories"]["workspace_dir"]

    print(f"Checking progress of jobs stored in {work_dir}")
    dirs = sorted(glob.glob(f"{work_dir}/v{args.version}/*/{args.data_level}/*/*"))
    if dirs == []:
        versions = [x.split("/v")[-1] for x in glob.glob(f"{work_dir}/v*")]
        print("Error, no directories found")
        print(f"for path {work_dir} found in {args.config_file} this is available")
        print(f"Versions {versions}")
        tag = "" if NSB_matching else "/Observations"
        print(f"Supported data types: DL1{tag}/M1, DL1{tag}/M2")
        exit(1)

    all_todo = 0
    all_return = 0
    all_good = 0
    all_cpu = []
    all_mem = []
    for dir in dirs:
        print(dir)
        # fixme list_dl0.txt is only available for DL1/M[12] processing
        list_dl0 = f"{dir}/logs/list_dl0.txt"
        try:
            with open(list_dl0, "r") as fp:
                this_todo = len(fp.readlines())
        except IOError:
            print(f"{RED}File {list_dl0} is missing{ENDC}")
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
                    file_in, slurm_id, task_id, rc = line.split()
                    if rc == "0":
                        this_good += 1
                        # now check accounting
                        if not args.no_accounting:
                            out = run_shell(
                                f'sacct --format="JobID,CPUTime,MaxRSS" --units=M  -j {slurm_id}_{task_id}| tail -1'
                            )
                            _, cpu, mem = out.split()
                            hh, mm, ss = (int(x) for x in str(cpu).split(":"))
                            delta = timedelta(
                                days=hh // 24, hours=hh % 24, minutes=mm, seconds=ss
                            )
                            this_cpu.append(delta)
                            this_mem.append(float(mem[0:-1]))
                    else:
                        print(f"file {file_in} failed with error {rc}")
                if len(this_cpu) > 0:
                    all_cpu += this_cpu
                    all_mem += this_mem
                    this_cpu = np.array(this_cpu)
                    this_mem = np.array(this_mem)
                    print(
                        f"CPU: median={np.median(this_cpu)}, max={this_cpu.max()}; memory [M]: median={np.median(this_mem)}, max={this_mem.max()}"
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
            f"CPU: median={np.median(all_cpu)}, max={all_cpu.max()}; memory [M]: median={np.median(all_mem)}, max={all_mem.max()}"
        )


if __name__ == "__main__":
    main()
