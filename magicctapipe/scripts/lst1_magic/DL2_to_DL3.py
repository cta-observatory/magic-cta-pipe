"""
Usage:
$ python DL2_to_DL3.py

"""

import os
import numpy as np
import glob
import yaml
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def configuration_DL3(ids, target_dir):
    
    """
    This function creates the configuration file needed for the DL2 to DL3 conversion
    
    Parameters
    ----------
    ids: list
        list of telescope IDs
    target_dir: str
        Path to the working directory
    """
    
    f = open(target_dir+'/config_DL3.yaml','w')
    f.write("mc_tel_ids:\n    LST-1: "+str(ids[0])+"\n    LST-2: "+str(ids[1])+"\n    LST-3: "+str(ids[2])+"\n    LST-4: "+str(ids[3])+"\n    MAGIC-I: "+str(ids[4])+"\n    MAGIC-II: "+str(ids[5])+"\n\n")
    f.write('dl2_to_dl3:\n    interpolation_method: "linear"  # select "nearest", "linear" or "cubic"\n    source_name: "Crab"\n    source_ra: null  #"83.633083 deg"  used when the source name cannot be resolved\n    source_dec: null  #"22.0145 deg" used when the source name cannot be resolved\n\n')
    
    f.close()
    
        

def DL2_to_3(target_dir):
    
    """
    This function creates the bash scripts to run lst1_magic_dl2_to_dl3.py on the real data.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """
    
    if not os.path.exists(target_dir+"/DL3"):
        os.mkdir(target_dir+"/DL3")
        
    process_name = "DL3_"+target_dir.split("/")[-2:][1]
    output = target_dir+"/DL3"
    nights = np.sort(glob.glob(target_dir+"/DL2/Observations/*"))
    IRF_dir = target_dir+"/IRF"
    
    for night in nights:
        listOfDL2files = np.sort(glob.glob(night+"/*.h5"))
        
        process_size = len(listOfDL2files) - 1
        
        f = open(f'DL3_{night.split("/")[-1]}.sh','w')
        f.write('#!/bin/sh\n\n')
        f.write('#SBATCH -p short\n')
        f.write('#SBATCH -J '+process_name+'\n')
        f.write(f"#SBATCH --array=0-{process_size}%50\n")
        f.write('#SBATCH -N 1\n\n')
        f.write('ulimit -l unlimited\n')
        f.write('ulimit -s unlimited\n')
        f.write('ulimit -a\n\n')

        for DL2_file in listOfDL2files:
            f.write(f'export LOG={output}/{DL2_file.split("/")[-1][:-3]}_DL3.log\n')
            f.write(f'conda run -n magic-lst python lst1_magic_dl2_to_dl3.py --input-file-dl2 {DL2_file} --input-dir-irf {IRF_dir} --output-dir {output} --config-file {target_dir}/config_DL3.yaml >$LOG 2>&1\n\n')

        f.close()
    

def main():

    """
    Here we read the config_general.yaml file and call the functions defined above.
    """
    
    with open("config_general.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    telescope_ids = list(config["mc_tel_ids"].values())
    
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    
    print("***** Generating file config_DL3.yaml...")
    print("***** This file can be found in ",target_dir)
    configuration_DL3(telescope_ids, target_dir)
    
    
    print("***** Generating bashscripts for DL2-DL3 conversion...")
    DL2_to_3(target_dir)
    
    print("***** Running lst1_magic_dl2_to_dl3.py in the DL2 real data files...")
    print("Process name: DL3_"+target_dir.split("/")[-2:][1])
    print("To check the jobs submitted to the cluster, type: squeue -n DL3_"+target_dir.split("/")[-2:][1])
    
    #Below we run the bash scripts to perform the DL1 to DL2 cnoversion:
    list_of_DL2_to_3_scripts = np.sort(glob.glob("DL3_*.sh"))
    
    for n,run in enumerate(list_of_DL2_to_3_scripts):
        if n == 0:
            launch_jobs =  f"dl3{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = launch_jobs + f" && dl3{n}=$(sbatch --parsable --dependency=afterany:$dl3{n-1} {run})"
    
    os.system(launch_jobs)

if __name__ == "__main__":
    main()


    
    
    
