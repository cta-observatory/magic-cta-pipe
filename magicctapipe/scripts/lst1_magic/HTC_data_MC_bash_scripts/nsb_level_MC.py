"""
Bash scripts to run LSTnsb.py on all the LST runs by using parallel jobs
Usage: python nsb_level.py (-c config.yaml)
"""

import argparse
import glob
import logging
import numpy as np
import os
import yaml

__all__=['bash_scripts']

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def sub_file(run_nr,run, source):
    print(source)

    print(run)
    with open(f'nsb_{run_nr}.sub','w') as f:
        lines=[
            f'executable= nsb_lvl_{run_nr}.sh\n',
            f'log=nsb_lvl_{run_nr}.log\n',
            'request_cpus   = 1',
            'request_memory = 5G',
            'request_disk   = 5G',
            f'error=err.nsb_lvl_{run_nr}\n',
            f'output=out.nsb_lvl_{run_nr}\n',
            'should_transfer_files = yes\n',
            f'queue',
        ]
        f.writelines(lines)
def bash_scripts(run_nr, run, config, source, env_name):
    '''Here we create the bash scripts (one per LST run)
    Parameters
    ----------
    run: str
        LST date and run number
    config: str
        Configuration file
    source: str
        Source name
    env_name:str
        Name of the environment

    '''
    lines = [
        "#!/bin/sh\n\n",
        "source /storage/gpfs_data/ctalocal/evisentin/conda\n",
        "mambafg\n",
        "echo $pwd\n",
        f"time mamba run -p  {env_name} LSTnsb_MC -c {config} -i {run} > {source}_nsblog_{run_nr}.log 2>&1 \n\n",
    ]
    with open(f"nsb_lvl_{run_nr}.sh", "w") as f:
        f.writelines(lines)


def main():
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
    source = config["directories"]["target_name"]
    runs = config["general"]["LST_runs"]
    env_name = config["general"]["env_name"]

    with open(str(runs), "r") as LSTfile:
        run = LSTfile.readlines()

    print("***** Generating bashscripts...")
    for n,i in enumerate(run):
        i = i.rstrip()
        bash_scripts(n,i, args.config_file, source, env_name)
        sub_file(n,i, source)
   
    

        launch_jobs = f"condor_submit -name sn-02.cr.cnaf.infn.it -spool nsb_{n}.sub"

  
        os.system(launch_jobs)


if __name__ == "__main__":
    main()
