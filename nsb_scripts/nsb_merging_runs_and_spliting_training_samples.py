"""
This script split the proton MC data sample into "train"
and "test", deletes possible failed runs (only those files
that end up with a size < 1 kB), and generates the bash 
scripts to merge the data files calling the script "merge_hdf_files.py"
in the follwoing order:

MAGIC:
1) Merge the subruns into runs for M1 and M2 individually.
2) Merge the runs of M1 and M2 into M1-M2 runs.
3) Merge all the M1-M2 runs for a given night.
Workingdir/DL1/Observations/Merged 

MC:
1) Merges all MC runs in a node and save them at
Workingdir/DL1/MC/PARTICLE/Merged 


Usage:
$ python merging_runs_and_spliting_training_samples.py

"""
import argparse
import os
import numpy as np
import glob
import yaml
import logging


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def merge(scripts_dir, target_dir, identification, MAGIC_runs):
    
    """
    This function creates the bash scripts to run merge_hdf_files.py in all MAGIC subruns.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    identification: str
        Tells which batch to create. Options: subruns, M1M2, nights
    MAGIC_runs: matrix of strings
        This matrix is imported from config_general.yaml and tells the function where to find the data and where to put the merged files
    """
    
    process_name = "merging_"+target_dir.split("/")[-2:][1]
    
    MAGIC_DL1_dir = target_dir+"/DL1/Observations"
    if os.path.exists(MAGIC_DL1_dir+"/M1") & os.path.exists(MAGIC_DL1_dir+"/M2"):
        if not os.path.exists(MAGIC_DL1_dir+"/Merged"):
            os.mkdir(MAGIC_DL1_dir+"/Merged")
    print('directory',MAGIC_DL1_dir)    
    f = open(f"Merge_{identification}.sh","w")
    f.write('#!/bin/sh\n\n')
    f.write('#SBATCH -p short\n')
    f.write('#SBATCH -J '+process_name+'\n')
    f.write('#SBATCH -N 1\n\n')
    f.write('ulimit -l unlimited\n')
    f.write('ulimit -s unlimited\n')
    f.write('ulimit -a\n\n')
    
    if identification == "0_subruns":
        if os.path.exists(MAGIC_DL1_dir+"/M1"):
            for i in MAGIC_runs:
                if not os.path.exists(MAGIC_DL1_dir+f"/Merged/{i[0]}"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}")   #Creating a merged directory for the respective night
                if not os.path.exists(MAGIC_DL1_dir+f"/Merged/{i[0]}/{i[1]}"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]}") #Creating a merged directory for the respective run
                f.write(f'conda run -n magic-lst python {scripts_dir}/merge_hdf_files.py --input-dir {MAGIC_DL1_dir}/M1/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} \n')
                    
        if os.path.exists(MAGIC_DL1_dir+"/M2"):
            for i in MAGIC_runs:
                if not os.path.exists(MAGIC_DL1_dir+f"/Merged/{i[0]}"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}")   #Creating a merged directory for the respective night
                if not os.path.exists(MAGIC_DL1_dir+f"/Merged/{i[0]}/{i[1]}"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]}") #Creating a merged directory for the respective run
                f.write(f'conda run -n magic-lst python {scripts_dir}/merge_hdf_files.py --input-dir {MAGIC_DL1_dir}/M2/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} \n')
    
    elif identification == "1_M1M2":
        if os.path.exists(MAGIC_DL1_dir+"/M1") & os.path.exists(MAGIC_DL1_dir+"/M2"):
            for i in MAGIC_runs:
                if not os.path.exists(MAGIC_DL1_dir+f"/Merged/{i[0]}/Merged"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}/Merged") 
                f.write(f'conda run -n magic-lst python {scripts_dir}/merge_hdf_files.py --input-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/Merged --run-wise \n')        
    else:
        for i in MAGIC_runs:
            if not os.path.exists(MAGIC_DL1_dir+f"/Merged/Merged_{i[0]}"):
                os.mkdir(f"{MAGIC_DL1_dir}/Merged/Merged_{i[0]}")  #Creating a merged directory for each night
            f.write(f'conda run -n magic-lst python {scripts_dir}/merge_hdf_files.py --input-dir {MAGIC_DL1_dir}/Merged/{i[0]}/Merged --output-dir {MAGIC_DL1_dir}/Merged/Merged_{i[0]} \n')
    
    
    f.close()
   
def main():

    """
    Here we read the config_general.yaml file, split the pronton sample into "test" and "train", and merge the MAGIC files.
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
    with open(args.config_file, "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    scripts_dir=config["directories"]["scripts_dir"]
    
    MAGIC_runs_and_dates = config["general"]["MAGIC_runs"]
    MAGIC_runs = np.genfromtxt(MAGIC_runs_and_dates,dtype=str,delimiter=',')
    
    if (len(MAGIC_runs)==2) and (len(MAGIC_runs[0])==10):
        
        MAGIC=MAGIC_runs

        MAGIC_runs=[]
        MAGIC_runs.append(MAGIC)


   
    print("***** Generating merge bashscripts...")
    merge(scripts_dir,target_dir, "0_subruns", MAGIC_runs) #generating the bash script to merge the subruns
    merge(scripts_dir,target_dir, "1_M1M2", MAGIC_runs) #generating the bash script to merge the M1 and M2 runs
    merge(scripts_dir,target_dir, "2_nights", MAGIC_runs) #generating the bash script to merge all runs per night
    

    
    
    print("***** Running merge_hdf_files.py in the MAGIC data files...")
    print("Process name: merging_"+target_dir.split("/")[-2:][1])
    print("To check the jobs submitted to the cluster, type: squeue -n merging_"+target_dir.split("/")[-2:][1])
    
    #Below we run the bash scripts to merge the MAGIC files
    list_of_merging_scripts = np.sort(glob.glob("Merge_*.sh"))
    
    for n,run in enumerate(list_of_merging_scripts):
        if n == 0:
            launch_jobs =  f"merging{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = launch_jobs + f" && merging{n}=$(sbatch --parsable --dependency=afterany:$merging{n-1} {run})"
    
    #print(launch_jobs)
    os.system(launch_jobs)

if __name__ == "__main__":
    main()


    
    
    
    
    
    
