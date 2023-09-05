"""
This scripts generates and runs the bashscripts
to compute the stereo parameters of DL1 MC and 
Coincident MAGIC+LST data files. 

Usage:
$ python stereo_events.py

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
    
     
def bash_stereo(scripts_dir, target_dir, nsb):

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
    
    if not os.path.exists(target_dir+"/DL1/Observations/Coincident_stereo/"+str(nsb)):
        os.mkdir(target_dir+"/DL1/Observations/Coincident_stereo/"+str(nsb))
    
    listOfNightsLST = np.sort(glob.glob(target_dir+f"/DL1/Observations/Coincident/{nsb}/*"))
    
    for nightLST in listOfNightsLST:
        stereoDir = target_dir+"/DL1/Observations/Coincident_stereo/"+str(nsb)+"/"+nightLST.split('/')[-1]
        
        if not os.listdir(f'{nightLST}'):
          print('skip ', nightLST)
          continue
        os.system(f"ls {nightLST}/*LST*.h5 >  {nightLST}/list_coin_{nsb}.txt")  #generating a list with the DL1 coincident data files.
        process_size = len(np.genfromtxt(nightLST+f"/list_coin_{nsb}.txt",dtype="str")) - 1

        if not os.path.exists(stereoDir):
            os.mkdir(stereoDir)
        
        f = open(f"StereoEvents_{nsb}_{nightLST.split('/')[-1]}.sh","w")
        f.write("#!/bin/sh\n\n")
        f.write("#SBATCH -p short\n")
        f.write("#SBATCH -J "+process_name+"_stereo"+str(nsb)+"\n")
        f.write(f"#SBATCH --array=0-{process_size}%100\n")
        f.write("#SBATCH -N 1\n\n")
        f.write("ulimit -l unlimited\n")
        f.write("ulimit -s unlimited\n")
        f.write("ulimit -a\n\n") 
        
        f.write(f"export INPUTDIR={nightLST}\n")
        f.write(f"export OUTPUTDIR={stereoDir}\n")
        f.write(f"SAMPLE_LIST=($(<$INPUTDIR/list_coin_{nsb}.txt))\n")
        f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
        f.write("export LOG=$OUTPUTDIR/stereo_${SLURM_ARRAY_TASK_ID}.log\n")
        f.write(f"conda run -n magic-lst python {scripts_dir}/lst1_magic_stereo_reco.py --input-file $SAMPLE --output-dir $OUTPUTDIR --config-file {target_dir}/config_stereo.yaml >$LOG 2>&1")
        f.close()



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

    args = parser.parse_args()
    with open(args.config_file, "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)

        
    
    
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    scripts_dir=config["directories"]["scripts_dir"]
    telescope_ids = list(config["mc_tel_ids"].values())
    
    print("***** Generating file config_stereo.yaml...")
    print("***** This file can be found in ",target_dir)
    configfile_stereo(telescope_ids, target_dir)
    listnsb = np.sort(glob.glob("LST_*_.txt"))
    nsb=[]
    for f in listnsb:
        nsb.append(f.split('_')[1])
    
    print('nsb', nsb)
    for nsblvl in nsb:
      
        print("***** Generating the bashscript...")
        bash_stereo(scripts_dir, target_dir, nsblvl)
    

    
        print("***** Submitting processess to the cluster...")
        print("Process name: "+target_dir.split("/")[-2:][1]+"_stereo"+str(nsblvl))
        print("To check the jobs submitted to the cluster, type: squeue -n "+target_dir.split("/")[-2:][1]+"_stereo"+str(nsblvl))
    
        #Below we run the bash scripts to find the stereo events
        list_of_stereo_scripts = np.sort(glob.glob(f"StereoEvents_{nsblvl}*.sh"))
    
        for n,run in enumerate(list_of_stereo_scripts):
          if n == 0:
            launch_jobs =  f"stereo{n}=$(sbatch --parsable {run})"
          else:
            launch_jobs = launch_jobs + f" && stereo{n}=$(sbatch --parsable --dependency=afterany:$stereo{n-1} {run})"
    
        
        os.system(launch_jobs)

if __name__ == "__main__":
    main()


    
    
    
    
    
    
