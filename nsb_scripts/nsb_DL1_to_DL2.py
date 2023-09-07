"""
This script creates the bashscripts necessary to apply "lst1_magic_dl1_stereo_to_dl2.py"
to the DL1 stereo data (real and MC). It also creates new subdirectories associated with
the data level 2. The DL2 files are saved at:
WorkingDirectory/DL2/
and in the subdirectories therein.

Usage:
$ python DL1_to_DL2.py

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

def DL1_to_2(scripts_dir,target_dir, nsb, config):
    
    """
    This function creates the bash scripts to run lst1_magic_dl1_stereo_to_dl2.py.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """
    
    if not os.path.exists(target_dir+"/DL2"):
        os.mkdir(target_dir+"/DL2")
    
    if not os.path.exists(target_dir+"/DL2/Observations"):
        os.mkdir(target_dir+"/DL2/Observations")
    if not os.path.exists(target_dir+"/DL2/Observations/"+str(nsb)):
        os.mkdir(target_dir+"/DL2/Observations/"+str(nsb))

        
    process_name = "DL2_"+target_dir.split("/")[-2:][1]+str(nsb)
    data_files_dir = target_dir+"/DL1/Observations/Coincident_stereo/"+str(nsb)
    RFs_dir = "/fefs/aswg/workspace/elisa.visentin/MAGIC_LST_analysis/PG1553_nsb/RF/"+str(nsb)   #then, RFs saved somewhere (as Julian's ones)
    listOfDL1nights = np.sort(glob.glob(data_files_dir+"/*"))
    print(data_files_dir)
    for night in listOfDL1nights:
        output = target_dir+f'/DL2/Observations/{nsb}/{night.split("/")[-1]}'
        if not os.path.exists(output):
            os.mkdir(output)        
        
        listOfDL1Files = np.sort(glob.glob(night+"/*.h5"))
        np.savetxt(night+"/list_of_DL1_stereo_files.txt",listOfDL1Files, fmt='%s')
        process_size = len(listOfDL1Files) - 1
        
        f = open(f'DL1_to_DL2_{nsb}_{night.split("/")[-1]}.sh','w')
        f.write('#!/bin/sh\n\n')
        f.write('#SBATCH -p long\n')
        f.write('#SBATCH -J '+process_name+'\n')
        f.write(f"#SBATCH --array=0-{process_size}%100\n")
        f.write('#SBATCH --mem=30g\n')
        f.write('#SBATCH -N 1\n\n')
        f.write('ulimit -l unlimited\n')
        f.write('ulimit -s unlimited\n')
        f.write('ulimit -a\n\n')
            
        f.write(f"SAMPLE_LIST=($(<{night}/list_of_DL1_stereo_files.txt))\n")
        f.write("SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}\n")
        f.write(f'export LOG={output}'+'/DL1_to_DL2_${SLURM_ARRAY_TASK_ID}.log\n')
        f.write(f'conda run -n magic-lst python {scripts_dir}/lst1_magic_dl1_stereo_to_dl2.py --input-file-dl1 $SAMPLE --input-dir-rfs {RFs_dir} --output-dir {output} --config-file {scripts_dir}/{config} >$LOG 2>&1\n\n')
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
    source=config['directories']['target_name']
    listnsb = np.sort(glob.glob(f"{source}_LST_*_.txt"))
    nsb=[]
    for f in listnsb:
        nsb.append(f.split('_')[2])
    
    print('nsb', nsb)
    for nsblvl in nsb:
      
      print("***** Generating bashscripts for DL2...")
      DL1_to_2(scripts_dir,target_dir, nsblvl, args.config_file)
  
    
    
      print("***** Running lst1_magic_dl1_stereo_to_dl2.py in the DL1 data files...")
      print("Process name: DL2_"+target_dir.split("/")[-2:][1]+str(nsblvl))
      print("To check the jobs submitted to the cluster, type: squeue -n DL2_"+target_dir.split("/")[-2:][1]+str(nsblvl))
    
      #Below we run the bash scripts to perform the DL1 to DL2 cnoversion:
      list_of_DL1_to_2_scripts = np.sort(glob.glob(f"DL1_to_DL2_{nsblvl}*.sh"))
      print(list_of_DL1_to_2_scripts)   
      for n,run in enumerate(list_of_DL1_to_2_scripts):
        if n == 0:
            launch_jobs =  f"dl2{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = launch_jobs + f" && dl2{n}=$(sbatch --parsable --dependency=afterany:$dl2{n-1} {run})"
    
      #print(launch_jobs)
      os.system(launch_jobs)

if __name__ == "__main__":
    main()

            
            
            
            
            
            
            
            
    
    
