"""
Usage:
$ python RF.py

"""

import os
import numpy as np
import glob
import yaml
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def configuration_RFs(ids, target_dir):
    
    """
    This function creates the configuration file needed for the RF step
    
    Parameters
    ----------
    ids: list
        list of telescope IDs
    target_dir: str
        Path to the working directory
    """
    
    f = open(target_dir+'/config_RF.yaml','w')
    f.write("mc_tel_ids:\n    LST-1: "+str(ids[0])+"\n    LST-2: "+str(ids[1])+"\n    LST-3: "+str(ids[2])+"\n    LST-4: "+str(ids[3])+"\n    MAGIC-I: "+str(ids[4])+"\n    MAGIC-II: "+str(ids[5])+"\n\n")
    f.write('energy_regressor:\n    settings:\n        n_estimators: 150\n        criterion: "squared_error"\n        max_depth: 50\n        min_samples_split: 2\n        min_samples_leaf: 2\n        min_weight_fraction_leaf: 0.0\n        max_features: 1.0\n        max_leaf_nodes: null\n        min_impurity_decrease: 0.0\n        bootstrap: true\n        oob_score: false\n        n_jobs: 5\n        random_state: 42\n        verbose: 0\n        warm_start: false\n        ccp_alpha: 0.0\n        max_samples: null\n\n')
    f.write('    features: ["intensity", "length", "width", "skewness", "kurtosis", "slope", "intensity_width_2", "h_max", "impact", "pointing_alt", "pointing_az",\n ]\n\n')
    f.write('    gamma_offaxis:\n        min: 0.2 deg\n        max: 0.5 deg\n\n')

    f.write('disp_regressor:\n    settings:\n        n_estimators: 150\n        criterion: "squared_error"\n        max_depth: 50\n        min_samples_split: 2\n        min_samples_leaf: 2\n        min_weight_fraction_leaf: 0.0\n        max_features: 1.0\n        max_leaf_nodes: null\n        min_impurity_decrease: 0.0\n        bootstrap: true\n        oob_score: false\n        n_jobs: 5\n        random_state: 42\n        verbose: 0\n        warm_start: false\n        ccp_alpha: 0.0\n        max_samples: null\n\n')
    f.write('    features: ["intensity", "length", "width", "skewness", "kurtosis", "slope", "intensity_width_2", "h_max", "impact", "pointing_alt", "pointing_az",\n ]\n\n')
    f.write('    gamma_offaxis:\n        min: 0.2 deg\n        max: 0.5 deg\n\n')

    f.write('event_classifier:\n    settings:\n        n_estimators: 100\n        criterion: "gini"\n        max_depth: 100\n        min_samples_split: 2\n        min_samples_leaf: 2\n        min_weight_fraction_leaf: 0.0\n        max_features: "sqrt"\n        max_leaf_nodes: null\n        min_impurity_decrease: 0.0\n        bootstrap: true\n        oob_score: false\n        n_jobs: 5\n        random_state: 42\n        verbose: 0\n        warm_start: false\n        class_weight: null\n        ccp_alpha: 0.0\n        max_samples: null\n\n')
    f.write('    features: ["intensity", "length", "width", "skewness", "kurtosis", "slope", "intensity_width_2", "h_max", "impact", "pointing_alt", "pointing_az",\n ]\n\n')
    f.write('    gamma_offaxis:\n        min: 0.2 deg\n        max: 0.5 deg\n\n')

    f.close()

    

def RandomForest(target_dir):
    
    """
    This function creates the bash scripts to run lst1_magic_train_rfs.py.
    
    Parameters
    ----------
    target_dir: str
        Path to the working directory
    """
    
    process_name = "RF_"+target_dir.split("/")[-2:][1]
    MC_DL1_dir = target_dir+"/DL1/MC"
    
    if not os.path.exists(MC_DL1_dir+"/RFs"):
        os.mkdir(MC_DL1_dir+"/RFs")
    
    f = open("RF.sh","w")
    f.write('#!/bin/sh\n\n')
    f.write('#SBATCH -p long\n')
    f.write('#SBATCH -J '+process_name+'\n')
    f.write('#SBATCH --mem=50g\n')
    f.write('#SBATCH -N 1\n\n')
    f.write('ulimit -l unlimited\n')
    f.write('ulimit -s unlimited\n')
    f.write('ulimit -a\n\n')
    
    f.write(f'export LOG={MC_DL1_dir}/RFs/RF_Train.log\n\n')
    
    f.write(f'conda run -n magic-lst python lst1_magic_train_rfs.py --input-dir-gamma {MC_DL1_dir}/gammadiffuse/Merged/StereoMerged --input-dir-proton {MC_DL1_dir}/protons/Merged/StereoMerged --output-dir {MC_DL1_dir}/RFs --config-file {target_dir}/config_RF.yaml --train-energy --train-disp --train-classifier --use-unsigned >$LOG 2>&1\n')
    
    f.close()
    
    

def main():

    """
    Here we read the config_general.yaml file, split the pronton sample into "test" and "train", and merge the MAGIC files.
    """
    
    
    with open("config_general.yaml", "rb") as f:   # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    
    telescope_ids = list(config["mc_tel_ids"].values())
    target_dir = config["directories"]["workspace_dir"]+config["directories"]["target_name"]
    
    
    print("***** Generating file config_RF.yaml...")
    print("***** This file can be found in ",target_dir)
    configuration_RFs(telescope_ids,target_dir)
    
    
    print("***** Generating RF bashscript...")
    RandomForest(target_dir)
    
    print("***** Running lst1_magic_train_rfs.py in the DL1 data files...")
    print("Process name: RF_"+target_dir.split("/")[-2:][1])
    print("To check the jobs submitted to the cluster, type: squeue -n RF_"+target_dir.split("/")[-2:][1])
    
    #Below we run the bash scripts to perform the RF
    
    launch_jobs =  "sbatch RF.sh"
    
    
    #print(launch_jobs)
    os.system(launch_jobs)

if __name__ == "__main__":
    main()


    
    
    
