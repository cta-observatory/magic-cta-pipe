# Script for MAGIC and MAGIC+LST analysis

This folder contains scripts to perform MAGIC-only or MAGIC+LST analysis.

Each script can be called from the command line from anywhere in your system (some console scripts are created during installation). Please run them with `-h` option for the first time to check what are the options available.

## MAGIC-only analysis

MAGIC-only analysis starts from MAGIC-calibrated data (\_Y\_ files). The analysis flow is as follows:

- `magic_calib_to_dl1.py` on real and MC data (if you use MCs produced with MMCS), to convert them into DL1 format
- if you use SimTelArray MCs, run `lst1_magic_mc_dl0_to_dl1.py` over them to convert them into DL1 format
- optionally, but recommended, `merge_hdf_files.py` to merge subruns and/or runs together
- `lst1_magic_stereo_reco.py` to add stereo parameters to the DL1 data (use `--magic-only` argument if the MC DL1 data contains LST-1 events)
- `lst1_magic_train_rfs.py` to train the RFs (energy, direction, classification) on train gamma MCs and protons
- `lst1_magic_dl1_stereo_to_dl2.py` to apply the RFs to stereo DL1 data (real and test MCs) and produce DL2 data
- `lst1_magic_create_irf.py` to create the IRF (use `magic_stereo` as `irf_type` in the configuration file)
- `lst1_magic_dl2_to_dl3.py` to create DL3 files, and `create_dl3_index_files.py` to create DL3 HDU and index files

## MAGIC+LST analysis: overview

MAGIC+LST analysis starts from MAGIC calibrated data (\_Y\_ files), LST DL1 data and SimTelArray DL0 data. The analysis flow is as following:

- `magic_calib_to_dl1.py` on real MAGIC data, to convert them into DL1 format
- `lst1_magic_mc_dl0_to_dl1.py` over SimTelArray MCs to convert them into DL1 format
- optionally, but recommended, `merge_hdf_files.py` on MAGIC data to merge subruns and/or runs together
- `lst1_magic_event_coincidence.py` to find coincident events between MAGIC and LST-1, starting from DL1 data
- `lst1_magic_stereo_reco.py` to add stereo parameters to the DL1 data
- `lst1_magic_train_rfs.py` to train the RFs (energy, direction, classification) on train gamma MCs and protons
- `lst1_magic_dl1_stereo_to_dl2.py` to apply the RFs to stereo DL1 data (real and test MCs) and produce DL2 data
- `lst1_magic_create_irf.py` to create the IRF
- `lst1_magic_dl2_to_dl3.py` to create DL3 files, and `create_dl3_index_files.py` to create DL3 HDU and index files

## MAGIC+LST analysis: data reduction tutorial (PRELIMINARY)

1) The very first step to reduce MAGIC-LST data is to have remote access/credentials to the IT Container, so provide one. Once you have it, the connection steps are the following:  

Authorized institute server (Client) &rarr;  ssh connection to CTALaPalma &rarr; ssh connection to cp01/02  

2) Once connected to the IT Container, install MAGIC-CTA-PIPE (e.g. in your home directory in the IT Container) following the tutorial here: https://github.com/cta-observatory/magic-cta-pipe

3) Do not forget to open the magic-lst environment with the command `conda activate magic-lst` before starting the analysis

### DL0 to DL1

In this step, we will convert the MAGIC and Monte Carlo (MC) Data Level (DL) 0 to DL1 (our goal is to reach DL3).

Now copy all the python scripts available here to your preferred directory (e.g. /fefs/aswg/workspace/yourname/yourprojectname) in the IT Container, as well as the files `config_general.yaml`, `MAGIC_runs.txt` and `LST_runs.txt`.

The file `config_general.yaml` must contain the telescope IDs and the directories with the MC data, as shown below:  
```
mc_tel_ids:
    LST-1: 1
    LST-2: 0
    LST-3: 0
    LST-4: 0
    MAGIC-I: 2
    MAGIC-II: 3

directories:
    workspace_dir : "/fefs/aswg/workspace/yourname/yourprojectname/" 
    target_name   : "CrabTeste"
    MC_gammas     : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray"
    MC_electrons  : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/Electrons/sim_telarray/" 
    MC_helium     : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/Helium/sim_telarray/" 
    MC_protons    : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/Protons/dec_2276/sim_telarray"
    MC_gammadiff  : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/GammaDiffuse/dec_2276/sim_telarray/"
    
general:
    target_RA_deg          : 83.633083 #RA in degrees
    target_Dec_deg         : 22.0145   #Dec in degrees
    SimTel_version         : "v1.4"    
    LST_version            : "v0.9" 
    focal_length           : "effective" #effective #nominal
    MAGIC_runs             : "MAGIC_runs.txt"  #If there is no MAGIC data, please fill this file with "0, 0"
    LST_runs               : "LST_runs.txt"  
    proton_train_fraction  : 0.8 # 0.8 means that 80% of the DL1 protons will be used for training the Random Forest
    env_name               : magic-lst

```

The file `MAGIC_runs.txt` looks like that:  
```
2020_11_19,5093174
2020_11_19,5093175
2020_12_08,5093491
2020_12_08,5093492

```


The columns here represent the night and run in which you want to select data. Please do not add blank spaces in the rows, as these names will be used to i) find the MAGIC data in the IT Container and ii) create the subdirectories in your working directory. If there is no MAGIC data, please fill this file with "0,0". Similarly, the `LST_runs.txt` file looks like this:

```
2020_11_18,2923
2020_11_18,2924
2020_12_07,3093

```
Note that the LST nights appear as being one day before MAGIC's!!! This is because LST saves the date at the beginning of the night, while MAGIC saves it at the end. If there is no LST data, please fill this file with "0,0". These files are the only ones we need to modify in order to convert DL0 into DL1 data.

In this analysis, we use a wobble of 0.4°.

To convert the MAGIC and SimTelArray MCs data into DL1 format, you first do the following:
> $ python setting_up_config_and_dir.py

```
***** Linking MC paths - this may take a few minutes ******
*** Reducing DL0 to DL1 data - this can take many hours ***
Process name: yourprojectnameCrabTeste
To check the jobs submitted to the cluster, type: squeue -n yourprojectnameCrabTeste
```
Note that this script can be run as  
> $ python setting_up_config_and_dir.py --analysis-type onlyMAGIC  

or  

> $ python setting_up_config_and_dir.py --analysis-type onlyMC  

if you want to convert only MAGIC or only MC DL0 files to DL1, respectively.


The script `setting_up_config_and_dir.py` does a series of things:
- Creates a directory with your source name within the directory `yourprojectname` and several subdirectories inside it that are necessary for the rest of the data reduction.
- Generates a configuration file called config_step1.yaml with and telescope ID information and adopted imaging/cleaning cuts, and puts it in the directory created in the previous step.
- Links the MAGIC and MC data addresses to their respective subdirectories defined in the previous steps.
- Runs the scripts `lst1_magic_mc_dl0_to_dl1.py` and `magic_calib_to_dl1.py` for each one of the linked data files.

In the file `config_general.yaml`, the sequence of telescopes is always LST1, LST2, LST3, LST4, MAGIC-I, MAGIC-II. So in this tutorial, we have  
LST-1 ID = 1  
LST-2 ID = 0  
LST-3 ID = 0  
LST-4 ID = 0  
MAGIC-I ID = 2  
MAGIC-II ID = 3  
If the telescope ID is set to 0, this means that the telescope is not used in the analysis.

You can check if this process is done by typing  
> $ squeue -n yourprojectnameCrabTeste
or
> $ squeue -u your_user_name

in the terminal. Once it is done, all of the subdirectories in `/fefs/aswg/workspace/yourname/yourprojectname/CrabTeste/DL1/` will be filled with files of the type `dl1_[...]_LST1_MAGIC1_MAGIC2_runXXXXXX.h5` for the MCs and `dl1_MX.RunXXXXXX.0XX.h5` for the MAGIC runs. The next step of the conversion of DL0 to DL1 is to split the DL1 MC proton sample into "train" and "test" datasets (these will be used later in the Random Forest event classification and to do some diagnostic plots) and to merge all the MAGIC data files such that in the end, we have only one datafile per night. To do so, we run the following script:

> $ python merging_runs_and_splitting_training_samples.py  

```
***** Splitting protons into 'train' and 'test' datasets...  
***** Generating merge bashscripts...  
***** Running merge_hdf_files.py in the MAGIC data files...  
Process name: merging_CrabTeste  
To check the jobs submitted to the cluster, type: squeue -n merging_CrabTeste
```

This script will slice the proton MC sample according to the entry "proton_train_fraction" in the "config_general.yaml" file, and then it will merge the MAGIC data files in the following order:
- MAGIC subruns are merged into single runs.  
- MAGIC I and II runs are merged (only if both telescopes are used, of course).  
- All runs in specific nights are merged, such that in the end we have only one datafile per night.  
- Proton MC training data is merged.
- Proton MC testing data is merged.
- Diffuse MC gammas are merged.
- MC gammas are merged.

### Coincident events and stereo parameters on DL1

To find coincident events between MAGIC and LST, starting from DL1 data, we run the following script:

> $ python coincident_events.py

This script creates the file config_coincidence.yaml containing the telescope IDs and the following parameters:
```
mc_tel_ids:
    LST-1: 1
    LST-2: 0
    LST-3: 0
    LST-4: 0
    MAGIC-I: 2
    MAGIC-II: 3

event_coincidence:
    timestamp_type_lst: "dragon_time"  # select "dragon_time", "tib_time" or "ucts_time"
    window_half_width: "300 ns"
    pre_offset_search: true
    n_pre_offset_search_events: 100
    time_offset:
        start: "-10 us"
        stop: "0 us"
```

It then links the real LST data files to the output directory [...]DL1/Observations/Coincident, and runs the script lst1_magic_event_coincidence.py in all of them.

Once it is done, we add stereo parameters to the MAGIC+LST coincident DL1 data by running:

> $ python stereo_events.py

This script creates the file config_stereo.yaml with the follwoing parameters:
```
mc_tel_ids:
    LST-1: 1
    LST-2: 0
    LST-3: 0
    LST-4: 0
    MAGIC-I: 2
    MAGIC-II: 3

stereo_reco:
    quality_cuts: "(intensity > 50) & (width > 0)"
    theta_uplim: "6 arcmin"
```

It then creates the output directories for the DL1 with stereo parameters [...]DL1/Observations/Coincident_stereo/SEVERALNIGHTS and [...]/DL1/MC/GAMMAorPROTON/Merged/StereoMerged, and then runs the script lst1_magic_stereo_reco.py in all of the coincident DL1 files. The stereo DL1 files for MC and real data are then saved in these directories.

