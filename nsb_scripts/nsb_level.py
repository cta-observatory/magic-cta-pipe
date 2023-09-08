import argparse
import os
import numpy as np
import glob
import yaml
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def bash_scripts(run,config):
    f = open(f'run_{run}.sh','w')
    f.write('#!/bin/sh\n\n')
    f.write('#SBATCH -p long\n')
    f.write('#SBATCH -J '+'nsb'+'\n')
    
    f.write('#SBATCH -N 1\n\n')
    f.write('ulimit -l unlimited\n')
    f.write('ulimit -s unlimited\n')
    f.write('ulimit -a\n\n')
            
    
    
    
    f.write(f'conda run -n  magic-lst python LSTnsb.py -c {config} -i {run} > nsblog_{run}.log 2>&1 \n\n')
    f.close()
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
    with open(args.config_file, "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
        print(f)
    runs = config["general"]["LST_runs"]
    print(runs)
    with open(str(runs), 'r') as LSTfile:
    
        run=LSTfile.readlines()
        print(run)
    
    for i in run:
        print(i)
        i=i.rstrip()
        bash_scripts(i,args.config_file)
    list_of_bash_scripts = np.sort(glob.glob(f"run_*.sh"))
    print(list_of_bash_scripts)   
    for n,run in enumerate(list_of_bash_scripts):
        if n == 0:
            launch_jobs =  f"nsb{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = launch_jobs + f" && nsb{n}=$(sbatch --parsable --dependency=afterany:$nsb{n-1} {run})"
    
      #print(launch_jobs)
    os.system(launch_jobs)
if __name__ == "__main__":
    main()
