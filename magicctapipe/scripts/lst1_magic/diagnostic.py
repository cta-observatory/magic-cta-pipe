"""
This script simply submit the job created by "lst_magic_diagnostic.py"
to the cluster. The results are several diagnostic plots saved at your
working directory.

Usage:
$ python diagnostic.py

"""

import os
import numpy as np
import glob
import yaml
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
    
def diagnostic_plots(target_dir):

    """
    This function runs the script lst_magic_diagnostic.py in the DL2 MC and real data files.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """
    
    process_name = "diagnostic_"+target_dir.split("/")[-2:][1]
    
    f = open("diagnostic_plots.sh","w")
    f.write('#!/bin/sh\n\n')
    f.write('#SBATCH -p long\n')
    f.write('#SBATCH -J '+process_name+'\n')
    f.write('#SBATCH --mem=30g\n')
    f.write('#SBATCH -N 1\n\n')
    f.write('ulimit -l unlimited\n')
    f.write('ulimit -s unlimited\n')
    f.write('ulimit -a\n\n')
    
    f.write(f'export LOG={target_dir}/diagnostic.log\n')
    f.write(f'conda run -n magic-lst python lst_magic_diagnostic.py >$LOG 2>&1\n')        
    
    f.close()


def main():

    """
    Here we read the config_general.yaml file and call the functions defined above.
    """
    
    with open("config_general.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    
   
    print("***** Running lst_magic_diagnostic.py in the DL2 MC and real data files...")
    print("Process name: diagnostic_"+target_dir.split("/")[-2:][1])
    print("To check the jobs submitted to the cluster, type: squeue -n diagnostic_"+target_dir.split("/")[-2:][1])
    diagnostic_plots(target_dir)
    
    #Below we run the bash scripts to perform the DL1 to DL2 cnoversion:
    list_of_DL2_to_3_scripts = np.sort(glob.glob("diagnostic_*.sh"))
    
    for n,run in enumerate(list_of_DL2_to_3_scripts):
        if n == 0:
            launch_jobs =  f"dl3{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = launch_jobs + f" && dl3{n}=$(sbatch --parsable --dependency=afterany:$dl3{n-1} {run})"
    
    os.system(launch_jobs)

if __name__ == "__main__":
    main()

