"""
This scripts generates and runs the bashscripts
to compute the stereo parameters of DL1 MC and 
Coincident MAGIC+LST data files. 

Usage:
$ python stereo_events.py

If you want to compute the stereo parameters only the real data or only the MC data,
you can do as follows:

Only real data:
$ python stereo_events.py --analysis-type onlyReal

Only MC:
$ python stereo_events.py --analysis-type onlyMC

"""

import os
import numpy as np
import glob
import yaml
import logging
from pathlib import Path
import argparse

__all__=['configfile_stereo', 'bash_stereo', 'bash_stereoMC']

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
    
    with open(f'{target_dir}/config_stereo.yaml','w') as f:
        f.write(f"mc_tel_ids:\n    LST-1: {ids[0]}\n    LST-2: {ids[1]}\n    LST-3: {ids[2]}\n    LST-4: {ids[3]}\n    MAGIC-I: {ids[4]}\n    MAGIC-II: {ids[5]}\n\n")
        f.write('stereo_reco:\n    quality_cuts: "(intensity > 50) & (width > 0)"\n    theta_uplim: "6 arcmin"\n')
      
    
     
def bash_stereo(target_dir, env_name):

    """
    This function generates the bashscript for running the stereo analysis.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """

    process_name = target_dir.split("/")[-2:][1]

    if not os.path.exists(f"{target_dir}/DL1/Observations/Coincident_stereo"):
        os.mkdir(f"{target_dir}/DL1/Observations/Coincident_stereo")
        
    listOfNightsLST = np.sort(glob.glob(f"{target_dir}/DL1/Observations/Coincident/*"))
    
    for nightLST in listOfNightsLST:
        stereoDir = f"{target_dir}/DL1/Observations/Coincident_stereo/{nightLST.split('/')[-1]}"
        if not os.path.exists(stereoDir):
            os.mkdir(stereoDir)
        
        os.system(f"ls {nightLST}/*LST*.h5 >  {nightLST}/list_coin.txt")  #generating a list with the DL1 coincident data files.
        process_size = len(np.genfromtxt(f"{nightLST}/list_coin.txt",dtype="str")) - 1
        
        with open(f"StereoEvents_real_{nightLST.split('/')[-1]}.sh","w") as f:
            f.write("#!/bin/sh\n\n")
            f.write("#SBATCH -p short\n")
            f.write(f"#SBATCH -J {process_name}_stereo\n")
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
            f.write(f"conda run -n {env_name} lst1_magic_stereo_reco --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_stereo.yaml >$LOG 2>&1")
            

def bash_stereoMC(target_dir, identification, env_name):

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

    if not os.path.exists(f"{target_dir}/DL1/MC/{identification}/Merged/StereoMerged"):
        os.mkdir(f"{target_dir}/DL1/MC/{identification}/Merged/StereoMerged")
    
    inputdir = f"{target_dir}/DL1/MC/{identification}/Merged"
    
    os.system(f"ls {inputdir}/dl1*.h5 >  {inputdir}/list_coin.txt")  #generating a list with the DL1 coincident data files.
    process_size = len(np.genfromtxt(f"{inputdir}/list_coin.txt",dtype="str")) - 1
    
    with open(f"StereoEvents_MC_{identification}.sh","w") as f:
        f.write("#!/bin/sh\n\n")
        f.write("#SBATCH -p xxl\n")
        f.write(f"#SBATCH -J {process_name}_stereo\n")
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
        f.write(f"conda run -n {env_name} lst1_magic_stereo_reco --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_stereo.yaml >$LOG 2>&1")
       


def main():

    """
    Here we read the config_general.yaml file and call the functions defined above.
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
        choices=['onlyReal', 'onlyMC'],
        dest="analysis_type",
        type=str,
        default="doEverything",
        help="You can type 'onlyReal' or 'onlyMC' to run this script only on real or MC data, respectively.",
    )

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    target_dir = f'{Path(config["directories"]["workspace_dir"])}/{config["directories"]["target_name"]}'


    env_name = config["general"]["env_name"]

    telescope_ids = list(config["mc_tel_ids"].values())
    
    print("***** Generating file config_stereo.yaml...")
    print("***** This file can be found in ",target_dir)
    configfile_stereo(telescope_ids, target_dir)
    
    #Below we run the analysis on the MC data
    if (args.analysis_type=='onlyMC') or (args.analysis_type=='doEverything'):
        print("***** Generating the bashscript for MCs...")
        bash_stereoMC(target_dir,"gammadiffuse", env_name)
        bash_stereoMC(target_dir,"gammas", env_name)
        bash_stereoMC(target_dir,"protons", env_name)
        bash_stereoMC(target_dir,"protons_test", env_name)

        list_of_stereo_scripts = np.sort(glob.glob("StereoEvents_MC_*.sh"))

        for n,run in enumerate(list_of_stereo_scripts):
            if n == 0:
                launch_jobs =  f"stereo{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = f"{launch_jobs} && stereo{n}=$(sbatch --parsable --dependency=afterany:$stereo{n-1} {run})"
        
        #print(launch_jobs)
        os.system(launch_jobs)
    
    #Below we run the analysis on the real data
    if (args.analysis_type=='onlyReal') or (args.analysis_type=='doEverything'): 
        print("***** Generating the bashscript for real data...")
        bash_stereo(target_dir, env_name)
        
        list_of_stereo_scripts = np.sort(glob.glob("StereoEvents_real_*.sh"))
    
        for n,run in enumerate(list_of_stereo_scripts):
            if n == 0:
                launch_jobs =  f"stereo{n}=$(sbatch --parsable {run})"
            else:
                launch_jobs = f"{launch_jobs} && stereo{n}=$(sbatch --parsable --dependency=afterany:$stereo{n-1} {run})"
        
        #print(launch_jobs)
        os.system(launch_jobs)

    print("***** Submitting processes to the cluster...")
    print(f"Process name: {target_dir.split('/')[-2:][1]}_stereo")
    print(f"To check the jobs submitted to the cluster, type: squeue -n {target_dir.split('/')[-2:][1]}_stereo")    

if __name__ == "__main__":
    main()
    
