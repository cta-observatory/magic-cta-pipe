"""
Usage:
$ python coincident_events.py

"""

import os
import numpy as np
import glob
import yaml
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def configfile_coincidence(ids, target_dir):
    
    """
    This function creates the configuration file needed for the event coincidence step
    
    Parameters
    ----------
    ids: list
        list of telescope IDs
    target_dir: str
        Path to the working directory
    """
    
    f = open(target_dir+'/config_coincidence.yaml','w')
    f.write("mc_tel_ids:\n    LST-1: "+str(ids[0])+"\n    LST-2: "+str(ids[1])+"\n    LST-3: "+str(ids[2])+"\n    LST-4: "+str(ids[3])+"\n    MAGIC-I: "+str(ids[4])+"\n    MAGIC-II: "+str(ids[5])+"\n\n")
    f.write('event_coincidence:\n    timestamp_type_lst: "dragon_time"  # select "dragon_time", "tib_time" or "ucts_time"\n    window_half_width: "300 ns"\n')
    f.write('    time_offset:\n        start: "-10 us"\n        stop: "0 us"\n')  
    f.close()
    

def linking_lst(target_dir, LST_runs):
    
    """
    This function links the LST data paths to the working directory. This is a preparation step required for running lst1_magic_event_coincidence.py 
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    LST_runs: matrix of strings
        This matrix is imported from config_general.yaml and tells the function where to find the LST data and link them to our working directory
    """
    
    
    coincidence_DL1_dir = target_dir+"/DL1/Observations"
    if not os.path.exists(coincidence_DL1_dir+"/Coincident"):
        os.mkdir(f"{coincidence_DL1_dir}/Coincident")
    
    for i in LST_runs:
        lstObsDir = i[0].split("_")[0]+i[0].split("_")[1]+i[0].split("_")[2]
        inputdir = f'/fefs/aswg/data/real/DL1/{lstObsDir}/v0.9/tailcut84'
        outputdir = f'{coincidence_DL1_dir}/Coincident/{lstObsDir}'
        list_of_subruns = np.sort(glob.glob(f"{inputdir}/dl1*Run*{i[1]}*.*.h5"))
        if os.path.exists(f"{outputdir}/list_LST.txt"):
            with open(f"{outputdir}/list_LST.txt", "a") as LSTdataPathFile:
                for subrun in list_of_subruns:
                    LSTdataPathFile.write(subrun+"\n") #If this files already exists, simply append the new information
        else:
            os.mkdir(outputdir)
            f = open(f"{outputdir}/list_LST.txt", "w") #If the file list_LST.txt does not exist, it will be created here
            for subrun in list_of_subruns:
                f.write(subrun+"\n")
            f.close()
        
     
def bash_coincident(target_dir):

    """
    This function generates the bashscript for running the coincidence analysis.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """

    process_name = target_dir.split("/")[-2:][1]
    
    listOfNightsLST = np.sort(glob.glob(target_dir+"/DL1/Observations/Coincident/*"))
    listOfNightsMAGIC = np.sort(glob.glob(target_dir+"/DL1/Observations/Merged/Merged*"))
    
    for nightMAGIC,nightLST in zip(listOfNightsMAGIC,listOfNightsLST):
        process_size = len(np.genfromtxt(nightLST+"/list_LST.txt",dtype="str"))
        
        f = open(f"LST_coincident_{nightLST.split('/')[-1]}.sh","w")
        f.write("#!/bin/sh\n\n")
        f.write("#SBATCH -p short\n")
        f.write("#SBATCH -J "+process_name+"_coincidence\n")
        f.write(f"#SBATCH --array=0-{process_size}%50\n")
        f.write("#SBATCH -N 1\n\n")
        f.write("ulimit -l unlimited\n")
        f.write("ulimit -s unlimited\n")
        f.write("ulimit -a\n\n") 

        f.write(f"export INM={nightMAGIC}\n")
        f.write(f"export OUTPUTDIR={nightLST}\n")
        f.write("SAMPLE_LIST=($(<$OUTPUTDIR/list_LST.txt))\n")
        f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
        f.write("export LOG=$OUTPUTDIR/coincidence_${SLURM_ARRAY_TASK_ID}.log\n")
        f.write(f"conda run -n magic-lst1 python lst1_magic_event_coincidence.py --input-file-lst $SAMPLE --input-dir-magic $INM --output-dir $OUTPUTDIR --config-file {target_dir}/config_coincidence.yaml >$LOG 2>&1")
        f.close()
        


def main():

    """
    Here we read the config_general.yaml file and call the functions defined above.
    """
    
    
    with open("config_general.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    telescope_ids = list(config["mc_tel_ids"].values())
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    
    LST_runs_and_dates = config["general"]["LST_runs"]
    LST_runs = np.genfromtxt(LST_runs_and_dates,dtype=str,delimiter=',')
    
    print("***** Generating file config_coincidence.yaml...")
    print("***** This file can be found in ",target_dir)
    configfile_coincidence(telescope_ids,target_dir)
    
        
    print("***** Linking the paths to LST data files...")
    linking_lst(target_dir, LST_runs) #linking the data paths to current working directory
    
    
    print("***** Generating the bashscript...")
    bash_coincident(target_dir)
    
    
    print("***** Submitting processess to the cluster...")
    print("Process name: "+target_dir.split("/")[-2:][1]+"_coincidence")
    print("To check the jobs submitted to the cluster, type: squeue -n "+target_dir.split("/")[-2:][1]+"_coincidence")
    
    #Below we run the bash scripts to find the coincident events
    list_of_coincidence_scripts = np.sort(glob.glob("LST_coincident*.sh"))
    
    for n,run in enumerate(list_of_coincidence_scripts):
        if n == 0:
            launch_jobs =  f"coincidence{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = launch_jobs + f" && coincidence{n}=$(sbatch --parsable --dependency=afterany:$coincidence{n-1} {run})"
    
    #print(launch_jobs)
    os.system(launch_jobs)

if __name__ == "__main__":
    main()


    
    
    
    
    
    
