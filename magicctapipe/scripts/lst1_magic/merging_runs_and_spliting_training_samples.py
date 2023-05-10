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

import os
import numpy as np
import glob
import yaml
import logging
from tqdm import tqdm

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
    
    os.chdir(target_dir+"/../")
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
    
    proton_dir = target_dir+"/DL1/MC/protons"
    
    if not os.path.exists(proton_dir+"/train"):
        os.mkdir(proton_dir+"/train")
    if not os.path.exists(proton_dir+"/../protons_test"):
        os.mkdir(proton_dir+"/../protons_test")
    
    list_of_dir = np.sort(glob.glob(proton_dir+'/node*' + os.path.sep))
    
    for directory in tqdm(range(len(list_of_dir))):   #tqdm allows us to print a progessbar in the terminal
        if not os.path.exists(proton_dir+"/train/"+list_of_dir[directory].split("/")[-2]):
            os.mkdir(proton_dir+"/train/"+list_of_dir[directory].split("/")[-2])
        if not os.path.exists(proton_dir+"/../protons_test/"+list_of_dir[directory].split("/")[-2]):
            os.mkdir(proton_dir+"/../protons_test/"+list_of_dir[directory].split("/")[-2])
        list_of_runs = np.sort(glob.glob(proton_dir+"/"+list_of_dir[directory].split("/")[-2]+"/*.h5"))
        split_percent = int(len(list_of_runs)*train_fraction)
        for j in list_of_runs[0:split_percent]:
            os.system(f"mv {j} {proton_dir}/train/"+list_of_dir[directory].split("/")[-2])
        
        os.system(f"cp {list_of_dir[directory]}*.txt "+proton_dir+"/train/"+list_of_dir[directory].split("/")[-2])
        os.system(f"mv {list_of_dir[directory]}*.txt "+proton_dir+"/../protons_test/"+list_of_dir[directory].split("/")[-2])
        os.system(f"mv {list_of_dir[directory]}*.h5 "+proton_dir+"/../protons_test/"+list_of_dir[directory].split("/")[-2])
        os.system(f"rm -r {list_of_dir[directory]}")

def merge(target_dir, identification, MAGIC_runs):
    
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
                f.write(f'conda run -n magic-lst python merge_hdf_files.py --input-dir {MAGIC_DL1_dir}/M1/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} \n')
                    
        if os.path.exists(MAGIC_DL1_dir+"/M2"):
            for i in MAGIC_runs:
                if not os.path.exists(MAGIC_DL1_dir+f"/Merged/{i[0]}"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}")   #Creating a merged directory for the respective night
                if not os.path.exists(MAGIC_DL1_dir+f"/Merged/{i[0]}/{i[1]}"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]}") #Creating a merged directory for the respective run
                f.write(f'conda run -n magic-lst python merge_hdf_files.py --input-dir {MAGIC_DL1_dir}/M2/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} \n')
    
    elif identification == "1_M1M2":
        if os.path.exists(MAGIC_DL1_dir+"/M1") & os.path.exists(MAGIC_DL1_dir+"/M2"):
            for i in MAGIC_runs:
                if not os.path.exists(MAGIC_DL1_dir+f"/Merged/{i[0]}/Merged"):
                    os.mkdir(f"{MAGIC_DL1_dir}/Merged/{i[0]}/Merged") 
                f.write(f'conda run -n magic-lst python merge_hdf_files.py --input-dir {MAGIC_DL1_dir}/Merged/{i[0]}/{i[1]} --output-dir {MAGIC_DL1_dir}/Merged/{i[0]}/Merged --run-wise \n')        
    else:
        for i in MAGIC_runs:
            if not os.path.exists(MAGIC_DL1_dir+f"/Merged/Merged_{i[0]}"):
                os.mkdir(f"{MAGIC_DL1_dir}/Merged/Merged_{i[0]}")  #Creating a merged directory for each night
            f.write(f'conda run -n magic-lst python merge_hdf_files.py --input-dir {MAGIC_DL1_dir}/Merged/{i[0]}/Merged --output-dir {MAGIC_DL1_dir}/Merged/Merged_{i[0]} \n')
    
    
    f.close()
    

def mergeMC(target_dir, identification):
    
    """
    This function creates the bash scripts to run merge_hdf_files.py in all MC runs.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    identification: str
        Tells which batch to create. Options: protons, gammadiffuse
    """
    
    process_name = "merging_"+target_dir.split("/")[-2:][1]
    
    MC_DL1_dir = target_dir+"/DL1/MC"
    if not os.path.exists(MC_DL1_dir+f"/{identification}/Merged"):
        os.mkdir(MC_DL1_dir+f"/{identification}/Merged")
    
    if identification == "protons":
        list_of_nodes = np.sort(glob.glob(MC_DL1_dir+f"/{identification}/train/node*"))
    else: 
        list_of_nodes = np.sort(glob.glob(MC_DL1_dir+f"/{identification}/node*"))
    
    np.savetxt(MC_DL1_dir+f"/{identification}/list_of_nodes.txt",list_of_nodes, fmt='%s')
    
        
    process_size = len(list_of_nodes) - 1
       
    cleaning(list_of_nodes, target_dir) #This will delete the (possibly) failed runs.
        
    f = open(f"Merge_{identification}.sh","w")
    f.write('#!/bin/sh\n\n')
    f.write('#SBATCH -p short\n')
    f.write('#SBATCH -J '+process_name+'\n')
    f.write(f"#SBATCH --array=0-{process_size}%50\n")
    f.write('#SBATCH --mem=7g\n')
    f.write('#SBATCH -N 1\n\n')
    f.write('ulimit -l unlimited\n')
    f.write('ulimit -s unlimited\n')
    f.write('ulimit -a\n\n')
    
    f.write(f"SAMPLE_LIST=($(<{MC_DL1_dir}/{identification}/list_of_nodes.txt))\n")
    f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
    f.write(f'export LOG={MC_DL1_dir}/{identification}/Merged'+'/merged_${SLURM_ARRAY_TASK_ID}.log\n')
    f.write(f'conda run -n magic-lst python merge_hdf_files.py --input-dir $SAMPLE --output-dir {MC_DL1_dir}/{identification}/Merged >$LOG 2>&1\n')        
    
    f.close()
    
    
def main():

    """
    Here we read the config_general.yaml file, split the pronton sample into "test" and "train", and merge the MAGIC files.
    """
    
    
    with open("config_general.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    
    MAGIC_runs_and_dates = config["general"]["MAGIC_runs"]
    MAGIC_runs = np.genfromtxt(MAGIC_runs_and_dates,dtype=str,delimiter=',')
    
    train_fraction = float(config["general"]["proton_train"])
    
    
    #Here we slice the proton MC data into "train" and "test":
    print("***** Spliting protons into 'train' and 'test' datasets...")
    split_train_test(target_dir, train_fraction)
    
    print("***** Generating merge bashscripts...")
    merge(target_dir, "0_subruns", MAGIC_runs) #generating the bash script to merge the subruns
    merge(target_dir, "1_M1M2", MAGIC_runs) #generating the bash script to merge the M1 and M2 runs
    merge(target_dir, "2_nights", MAGIC_runs) #generating the bash script to merge all runs per night
    
    print("***** Generating mergeMC bashscripts...")
    mergeMC(target_dir, "protons") #generating the bash script to merge the files
    mergeMC(target_dir, "gammadiffuse") #generating the bash script to merge the files
    mergeMC(target_dir, "gammas") #generating the bash script to merge the files 
    mergeMC(target_dir, "protons_test")
    
    
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


    
    
    
    
    
    
