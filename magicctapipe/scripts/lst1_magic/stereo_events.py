"""
This scripts generates and runs the bashscripts
to compute the stereo parameters of DL1 MC and 
Coincident MAGIC+LST data files. 

Usage:
$ python stereo_events.py

"""

import os
import numpy as np
import glob
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def configfile_stereo(ids, target_dir):
    
    """
    This function creates the configuration file needed for the event stereo step
    
    Parameters
    ----------
    ids: list
        list of telescope IDs
    target_dir: str
        Path to the working directory
    """
    
    f = open(target_dir+'/config_stereo.yaml','w')
    f.write("mc_tel_ids:\n    LST-1: "+str(ids[0])+"\n    LST-2: "+str(ids[1])+"\n    LST-3: "+str(ids[2])+"\n    LST-4: "+str(ids[3])+"\n    MAGIC-I: "+str(ids[4])+"\n    MAGIC-II: "+str(ids[5])+"\n\n")
    f.write('stereo_reco:\n    quality_cuts: "(intensity > 50) & (width > 0)"\n    theta_uplim: "6 arcmin"\n')
    f.close()
    
     
def bash_stereo(target_dir):

    """
    This function generates the bashscript for running the stereo analysis.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """

    process_name = target_dir.split("/")[-2:][1]

    if not os.path.exists(target_dir+"/DL1/Observations/Coincident_stereo"):
        os.mkdir(target_dir+"/DL1/Observations/Coincident_stereo")
        
    listOfNightsLST = np.sort(glob.glob(target_dir+"/DL1/Observations/Coincident/*"))
    
    for nightLST in listOfNightsLST:
        stereoDir = target_dir+"/DL1/Observations/Coincident_stereo/"+nightLST.split('/')[-1]
        if not os.path.exists(stereoDir):
            os.mkdir(stereoDir)
        
        os.system(f"ls {nightLST}/*LST*.h5 >  {nightLST}/list_coin.txt")  #generating a list with the DL1 coincident data files.
        process_size = len(np.genfromtxt(nightLST+"/list_coin.txt",dtype="str")) - 1
        
        f = open(f"StereoEvents_{nightLST.split('/')[-1]}.sh","w")
        f.write("#!/bin/sh\n\n")
        f.write("#SBATCH -p short\n")
        f.write("#SBATCH -J "+process_name+"_stereo\n")
        f.write(f"#SBATCH --array=0-{process_size}%100\n")
        f.write("#SBATCH -N 1\n\n")
        f.write("ulimit -l unlimited\n")
        f.write("ulimit -s unlimited\n")
        f.write("ulimit -a\n\n") 
        
        f.write(f"export INPUTDIR={nightLST}\n")
        f.write(f"export OUTPUTDIR={stereoDir}\n")
        f.write("SAMPLE_LIST=($(<$INPUTDIR/list_coin.txt))\n")
        f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
        f.write("export LOG=$OUTPUTDIR/stereo_${SLURM_ARRAY_TASK_ID}.log\n")
        f.write(f"conda run -n magic-lst python lst1_magic_stereo_reco.py --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_stereo.yaml >$LOG 2>&1")
        f.close()

def bash_stereoMC(target_dir, identification):

    """
    This function generates the bashscript for running the stereo analysis.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    identification: str
        Particle name. Options: protons, gammadiffuse
    """

    process_name = target_dir.split("/")[-2:][1]

    if not os.path.exists(target_dir+f"/DL1/MC/{identification}/Merged/StereoMerged"):
        os.mkdir(target_dir+f"/DL1/MC/{identification}/Merged/StereoMerged")
    
    inputdir = target_dir+f"/DL1/MC/{identification}/Merged"
    
    os.system(f"ls {inputdir}/dl1*.h5 >  {inputdir}/list_coin.txt")  #generating a list with the DL1 coincident data files.
    process_size = len(np.genfromtxt(inputdir+"/list_coin.txt",dtype="str")) - 1
    
    f = open(f"StereoEvents_{identification}.sh","w")
    f.write("#!/bin/sh\n\n")
    f.write("#SBATCH -p xxl\n")
    f.write("#SBATCH -J "+process_name+"_stereo\n")
    f.write(f"#SBATCH --array=0-{process_size}%100\n")
    f.write('#SBATCH --mem=30g\n')
    f.write("#SBATCH -N 1\n\n")
    f.write("ulimit -l unlimited\n")
    f.write("ulimit -s unlimited\n")
    f.write("ulimit -a\n\n")
    
    f.write(f"export INPUTDIR={inputdir}\n")
    f.write(f"export OUTPUTDIR={inputdir}/StereoMerged\n")
    f.write("SAMPLE_LIST=($(<$INPUTDIR/list_coin.txt))\n")
    f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
    f.write("export LOG=$OUTPUTDIR/stereo_${SLURM_ARRAY_TASK_ID}.log\n")
    f.write(f"conda run -n magic-lst python lst1_magic_stereo_reco.py --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_stereo.yaml >$LOG 2>&1")
    f.close()





def main():

    """
    Here we read the config_general.yaml file and call the functions defined above.
    """
    
    
    with open("config_general.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    
    target_dir = str(Path(config["directories"]["workspace_dir"]))+"/"+config["directories"]["target_name"]
    telescope_ids = list(config["mc_tel_ids"].values())
    
    print("***** Generating file config_stereo.yaml...")
    print("***** This file can be found in ",target_dir)
    configfile_stereo(telescope_ids, target_dir)
    
    print("***** Generating the bashscript...")
    bash_stereo(target_dir)
    
    print("***** Generating the bashscript for MCs...")
    bash_stereoMC(target_dir,"gammadiffuse")
    bash_stereoMC(target_dir,"gammas")
    bash_stereoMC(target_dir,"protons")
    bash_stereoMC(target_dir,"protons_test")
    
    print("***** Submitting processes to the cluster...")
    print("Process name: "+target_dir.split("/")[-2:][1]+"_stereo")
    print("To check the jobs submitted to the cluster, type: squeue -n "+target_dir.split("/")[-2:][1]+"_stereo")
    
    #Below we run the bash scripts to find the stereo events
    list_of_stereo_scripts = np.sort(glob.glob("StereoEvents_*.sh"))
    
    for n,run in enumerate(list_of_stereo_scripts):
        if n == 0:
            launch_jobs =  f"stereo{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = launch_jobs + f" && stereo{n}=$(sbatch --parsable --dependency=afterany:$stereo{n-1} {run})"
    
    #print(launch_jobs)
    os.system(launch_jobs)

if __name__ == "__main__":
    main()


    
    
    
    
    
    
