"""
Usage:
$ python IRF.py

"""

import os
import numpy as np
import glob
import yaml
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def configuration_IRF(target_dir):
    
    """
    This function creates the configuration file needed for the IRF step
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """
    
    f = open(target_dir+'/config_IRF.yaml','w')
    f.write('create_irf:\n    quality_cuts: "disp_diff_mean < 0.22"\n    event_type: "software"  # select "software", "software_only_3tel", "magic_only" or "hardware"\n    weight_type_dl2: "intensity"  # select "simple", "variance" or "intensity"\n    obs_time_irf: "50 h"  # used when creating a background HDU\n\n')
    
    f.write('    energy_bins:  # log space\n        start: "0.01 TeV"\n        stop: "1000 TeV"\n        n_edges: 26\n    migration_bins:  # log space\n        start: 0.2\n        stop: 5\n        n_edges: 31\n\n')

    f.write('    fov_offset_bins:  # linear space, used for diffuse MCs\n        start: "0 deg"\n        stop: "1 deg"\n        n_edges: 6\n\n')
    
    f.write('    source_offset_bins:  # linear space, used when creating PSF HDU\n        start: "0 deg"\n        stop: "1 deg"\n        n_edges: 101\n\n')
    
    f.write('    bkg_fov_offset_bins:  # linear space, used when creating background HDU\n        start: "0 deg"\n        stop: "10 deg"\n        n_edges: 21\n\n')
    
    f.write('    gammaness:\n        cut_type: "dynamic"  # select "global" or "dynamic"\n        global_cut_value: 0.8  # used for the global cut\n        efficiency: 0.9  # used for the dynamic cuts\n        min_cut: 0.05  # used for the dynamic cuts\n        max_cut: 0.85  # used for the dynamic cuts\n\n')
    
    f.write('    theta:\n        cut_type: "dynamic"  # select "global" or "dynamic"\n        global_cut_value: "0.2 deg"  # used for the global cut\n        efficiency: 0.75  # used for the dynamic cuts\n        min_cut: "0.1 deg"  # used for the dynamic cuts\n        max_cut: "0.3 deg"  # used for the dynamic cuts\n\n')
    
    
    f.close()

        

def IRF(target_dir):
    
    """
    This function creates the bash scripts to run lst1_magic_create_irf.py on the MC gammas.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """
    
    if not os.path.exists(target_dir+"/IRF"):
        os.mkdir(target_dir+"/IRF")
        
        
    process_name = "IRF_"+target_dir.split("/")[-2:][1]
    data_files_dir = target_dir+"/DL2/MC"
    listOfDL2MCgammas = np.sort(glob.glob(data_files_dir+"/*.h5"))
    
    output = target_dir+"/IRF"
    process_size = len(listOfDL2MCgammas)
    
    f = open('IRF.sh','w')
    f.write('#!/bin/sh\n\n')
    f.write('#SBATCH -p short\n')
    f.write('#SBATCH -J '+process_name+'\n')
    f.write(f"#SBATCH --array=0-{process_size}%50\n")
    f.write('#SBATCH -N 1\n\n')
    f.write('ulimit -l unlimited\n')
    f.write('ulimit -s unlimited\n')
    f.write('ulimit -a\n\n')

    for DL2_MC_file in listOfDL2MCgammas:
        f.write(f'export LOG={output}/{DL2_MC_file.split("/")[-1][:-3]}_IRF.log\n')
        f.write(f'conda run -n magic-lst1 python lst1_magic_create_irf.py --input-file-gamma {DL2_MC_file} --output-dir {output} --config-file {target_dir}/config_IRF.yaml >$LOG 2>&1\n\n')

    f.close()
    

def main():

    """
    Here we read the config_general.yaml file and call the functions defined above.
    """
    
    
    with open("config_general.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    
    print("***** Generating file config_IRF.yaml...")
    print("***** This file can be found in ",target_dir)
    configuration_IRF(target_dir)
    
    
    print("***** Generating bashscripts for IRF...")
    IRF(target_dir)
    
    print("***** Running lst1_magic_create_irf.py in the DL2 MC data files...")
    print("Process name: IRF_"+target_dir.split("/")[-2:][1])
    print("To check the jobs submitted to the cluster, type: squeue -n IRF_"+target_dir.split("/")[-2:][1])
    
    #Below we run the bash scripts to perform the DL1 to DL2 cnoversion:
    list_of_DL1_to_2_scripts = np.sort(glob.glob("IRF*.sh"))
    
    for n,run in enumerate(list_of_DL1_to_2_scripts):
        if n == 0:
            launch_jobs =  f"conversion{n}=$(sbatch --parsable {run})"
        else:
            launch_jobs = launch_jobs + f" && conversion{n}=$(sbatch --parsable --dependency=afterany:$conversion{n-1} {run})"
    
    #print(launch_jobs)
    os.system(launch_jobs)

if __name__ == "__main__":
    main()


    
    
    
