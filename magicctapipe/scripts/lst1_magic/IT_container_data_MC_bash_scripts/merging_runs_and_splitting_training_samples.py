"""
This script splits the proton MC data sample into "train"
and "test", deletes possible failed runs (only those files
that end up with a size < 1 kB), and generates the bash 
scripts to merge the data files calling the script "merge_hdf_files.py"
in the following order:

MAGIC:
1) Merge the subruns into runs for M1 and M2 individually.
2) Merge the runs of M1 and M2 into M1-M2 runs.
3) Merge all the M1-M2 runs for a given night.
Workingdir/DL1/Observations/Merged 

MC:
1) Merges all MC runs in a node and save them at
Workingdir/DL1/MC/PARTICLE/Merged 


Usage:
$ python merging_runs_and_splitting_training_samples.py

If you want to merge only the MAGIC or only the MC data,
you can do as follows:

Only MAGIC:
$ python merging_runs_and_splitting_training_samples.py --analysis-type onlyMAGIC

Only MC:
$ python merging_runs_and_splitting_training_samples.py --analysis-type onlyMC
"""

import os
import numpy as np
import glob
import yaml
import logging
from tqdm import tqdm
from pathlib import Path
import argparse

__all__=["cleaning", "split_train_test", "merge", "mergeMC"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def cleaning(list_of_nodes, target_dir):
    
    """
    This function looks for failed runs in each node and remove them.
    
    Parameters
    ----------
    target_dir: str
        Path to the target directory.
    list_of_nodes: array of str
        List of nodes where the function will look for failed runs.
    """
    
    for i in tqdm(range(len(list_of_nodes)), desc="Cleaning failed runs"):
        os.chdir(list_of_nodes[i])
        os.system('find . -type f -name "*.h5" -size -1k -delete')
    
    os.chdir(f"{target_dir}/../")
    print("Cleaning done.")

def split_train_test(target_dir, train_fraction):
    
    """
    This function splits the MC proton sample in 2, i.e. the "test" and the "train" subsamples.
    It generates 2 subdirectories in the directory .../DL1/MC/protons named "test" and "train" and creates sub-sub-directories with the names of all nodes.
    For each node sub-sub-directory we move 80% of the .h5 files (if it is in the "test" subdirectory) or 20% of the .h5 files (if it is in the "train" subdirectory).
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    train_fraction: float
        Fraction of proton MC files to be used in the training RF dataset
    """
    
    proton_dir = f"{target_dir}/DL1/MC/protons"
    
    if not os.path.exists(f"{proton_dir}/train"):
        os.mkdir(f"{proton_dir}/train")
    if not os.path.exists(f"{proton_dir}/../protons_test"):
        os.mkdir(f"{proton_dir}/../protons_test")
    
    list_of_dir = np.sort(glob.glob(f'{proton_dir}/node*{os.path.sep}'))
    
    for directory in tqdm(range(len(list_of_dir))):   #tqdm allows us to print a progessbar in the terminal
        if not os.path.exists(f"{proton_dir}/train/{list_of_dir[directory].split('/')[-2]}"):
            os.mkdir(f"{proton_dir}/train/{list_of_dir[directory].split('/')[-2]}")
        if not os.path.exists(f"{proton_dir}/../protons_test/{list_of_dir[directory].split('/')[-2]}"):
            os.mkdir(f'{proton_dir}/../protons_test/{list_of_dir[directory].split("/")[-2]}')
        list_of_runs = np.sort(glob.glob(f'{proton_dir}/{list_of_dir[directory].split("/")[-2]}/*.h5'))
        split_percent = int(len(list_of_runs)*train_fraction)
        for j in list_of_runs[0:split_percent]:
            os.system(f"mv {j} {proton_dir}/train/{list_of_dir[directory].split('/')[-2]}")
        
        os.system(f"cp {list_of_dir[directory]}*.txt {proton_dir}/train/{list_of_dir[directory].split('/')[-2]}")
        os.system(f"mv {list_of_dir[directory]}*.txt {proton_dir}/../protons_test/{list_of_dir[directory].split('/')[-2]}")
        os.system(f"mv {list_of_dir[directory]}*.h5 {proton_dir}/../protons_test/{list_of_dir[directory].split('/')[-2]}")
        os.system(f"rm -r {list_of_dir[directory]}")

def merge(target_dir, identification, MAGIC_runs, env_name):
    
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
    
    process_name = f"merging_{target_dir.split('/')[-2:][1]}"
    
    MAGIC_DL1_dir = f"{target_dir}/DL1/Observations"
    if os.path.exists(f"{MAGIC_DL1_dir}/M1") & os.path.exists(f"{MAGIC_DL1_dir}/M2"):
        if not os.path.exists(f"{MAGIC_DL1_dir}/Merged"):
            os.mkdir(f"{MAGIC_DL1_dir}/Merged")
    
    with open(f"Merge_MAGIC_{identification}.sh","w") as f:
        f.write('#!/bin/sh\n\n')
        f.write('#SBATCH -p short\n')
        f.write(f'#SBATCH -J {process_name}\n')
        f.write('#SBATCH -N 1\n\n')
        f.write('ulimit -l unlimited\n')
        f.write('ulimit -s unlimited\n')
        f.write('ulimit -a\n\n')
        
        if identification == "0_subruns":
            if os.path.exists(f"{MAGIC_DL1_dir}/M1"):
                for i in MAGIC_runs:
                    if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i[0]}"):
                        os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}")   #Creating a merged directory for the respective night
                    if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]}"):
                        os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]}") #Creating a merged directory for the respective run
                    f.write(f'conda run -n {env_name} merge_hdf_files --input-dir {MAGIC_DL1_dir}/M1/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} \n')
                        
            if os.path.exists(f"{MAGIC_DL1_dir}/M2"):
                for i in MAGIC_runs:
                    if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i[0]}"):
                        os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}")   #Creating a merged directory for the respective night
                    if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]}"):
                        os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]}") #Creating a merged directory for the respective run
                    f.write(f'conda run -n {env_name} merge_hdf_files --input-dir {MAGIC_DL1_dir}/M2/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} \n')
        
        elif identification == "1_M1M2":
            if os.path.exists(f"{MAGIC_DL1_dir}/M1") & os.path.exists(f"{MAGIC_DL1_dir}/M2"):
                for i in MAGIC_runs:
                    if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/{i[0]}/Merged"):
                        os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}/Merged") 
                    f.write(f'conda run -n {env_name} merge_hdf_files --input-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/Merged --run-wise \n')        
        else:
            for i in MAGIC_runs:
                if not os.path.exists(f"{MAGIC_DL1_dir}/Merged/Merged_{i[0]}"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/Merged_{i[0]}")  #Creating a merged directory for each night
                f.write(f'conda run -n {env_name} merge_hdf_files --input-dir {MAGIC_DL1_dir}/Merged/{i[0]}/Merged --output-dir {MAGIC_DL1_dir}/Merged/Merged_{i[0]} \n')
        
    
    
    

def mergeMC(target_dir, identification, env_name):
    
    """
    This function creates the bash scripts to run merge_hdf_files.py in all MC runs.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    identification: str
        Tells which batch to create. Options: protons, gammadiffuse
    """
    
    process_name = f"merging_{target_dir.split('/')[-2:][1]}"
    
    MC_DL1_dir = f"{target_dir}/DL1/MC"
    if not os.path.exists(f"{MC_DL1_dir}/{identification}/Merged"):
        os.mkdir(f"{MC_DL1_dir}/{identification}/Merged")
    
    if identification == "protons":
        list_of_nodes = np.sort(glob.glob(f"{MC_DL1_dir}/{identification}/train/node*"))
    else: 
        list_of_nodes = np.sort(glob.glob(f"{MC_DL1_dir}/{identification}/node*"))
    
    np.savetxt(f"{MC_DL1_dir}/{identification}/list_of_nodes.txt",list_of_nodes, fmt='%s')
    
        
    process_size = len(list_of_nodes) - 1
       
    cleaning(list_of_nodes, target_dir) #This will delete the (possibly) failed runs.
        
    with open(f"Merge_MC_{identification}.sh","w") as f:
        lines_bash_file = [
        '#!/bin/sh\n\n',
        '#SBATCH -p short\n',
        f'#SBATCH -J {process_name}n',
        f"#SBATCH --array=0-{process_size}%50\n",
        '#SBATCH --mem=7g\n',
        '#SBATCH -N 1\n\n',
        'ulimit -l unlimited\n',
        'ulimit -s unlimited\n',
        'ulimit -a\n\n',
        f"SAMPLE_LIST=($(<{MC_DL1_dir}/{identification}/list_of_nodes.txt))\n",
        "SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n",
        f'export LOG={MC_DL1_dir}/{identification}/Merged'+'/merged_${SLURM_ARRAY_TASK_ID}.log\n',
        f'conda run -n {env_name} merge_hdf_files --input-dir $SAMPLE --output-dir {MC_DL1_dir}/{identification}/Merged >$LOG 2>&1\n'
        ]
        f.writelines(lines_bash_file)
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

    parser.add_argument(
        "--analysis-type",
        "-t",
        choices=['onlyMAGIC', 'onlyMC'],
        dest="analysis_type",
        type=str,
        default="doEverything",
        help="You can type 'onlyMAGIC' or 'onlyMC' to run this script only on MAGIC or MC data, respectively.",
    )

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    
    target_dir = f'{Path(config["directories"]["workspace_dir"])}/{config["directories"]["target_name"]}'
    
    MAGIC_runs_and_dates = config["general"]["MAGIC_runs"]
    MAGIC_runs = np.genfromtxt(MAGIC_runs_and_dates,dtype=str,delimiter=',')
    
    train_fraction = float(config["general"]["proton_train_fraction"])
   
    env_name = config["general"]["env_name"]
    
    #Below we run the analysis on the MC data
    if (args.analysis_type=='onlyMC') or (args.analysis_type=='doEverything'):
        #Here we slice the proton MC data into "train" and "test" (but first we check if the directory already exists):
        if not os.path.exists(f"{target_dir}/DL1/MC/protons_test"):
            print("***** Splitting protons into 'train' and 'test' datasets...")
            split_train_test(target_dir, train_fraction)
    
        print("***** Generating merge_MC bashscripts...")
        mergeMC(target_dir, "protons", env_name) #generating the bash script to merge the files
        mergeMC(target_dir, "gammadiffuse", env_name) #generating the bash script to merge the files
        mergeMC(target_dir, "gammas", env_name) #generating the bash script to merge the files 
        mergeMC(target_dir, "protons_test", env_name)

        print("***** Running merge_hdf_files.py on the MC data files...")

        #Below we run the bash scripts to merge the MC files
        list_of_merging_scripts = np.sort(glob.glob("Merge_MC_*.sh"))

        for n,run in enumerate(list_of_merging_scripts):
            if n == 0:
                launch_jobs =  f"merging{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = f"{launch_jobs} && merging{n}=$(sbatch --parsable --dependency=afterany:$merging{n-1} {run})"
        
        os.system(launch_jobs)


    #Below we run the analysis on the MAGIC data
    if (args.analysis_type=='onlyMAGIC') or (args.analysis_type=='doEverything'): 
        print("***** Generating merge_MAGIC bashscripts...")
        merge(target_dir, "0_subruns", MAGIC_runs, env_name) #generating the bash script to merge the subruns
        merge(target_dir, "1_M1M2", MAGIC_runs, env_name) #generating the bash script to merge the M1 and M2 runs
        merge(target_dir, "2_nights", MAGIC_runs, env_name) #generating the bash script to merge all runs per night
        
        print("***** Running merge_hdf_files.py on the MAGIC data files...")
        
        #Below we run the bash scripts to merge the MAGIC files
        list_of_merging_scripts = np.sort(glob.glob("Merge_MAGIC_*.sh"))
        
        for n,run in enumerate(list_of_merging_scripts):
            if n == 0:
                launch_jobs =  f"merging{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = f"{launch_jobs} && merging{n}=$(sbatch --parsable --dependency=afterany:$merging{n-1} {run})"
        
        os.system(launch_jobs)
    
    print(f"Process name: merging_{target_dir.split('/')[-2:][1]}")
    print(f"To check the jobs submitted to the cluster, type: squeue -n merging_{target_dir.split('/')[-2:][1]}")
    
if __name__ == "__main__":
    main()


    
    
    
    
    
    
