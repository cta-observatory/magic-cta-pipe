# Script for MAGIC and MAGIC+LST-1 analysis

This folder contains scripts to perform MAGIC-only or MAGIC+LST-1 analysis.

Each script can be called from the command line from anywhere in your system (some console scripts are created during installation). Please run them with `-h` option for the first time to check what are the options available.

## MAGIC-only analysis

MAGIC-only analysis starts from MAGIC calibrated data (\_Y\_ files). The analysis flow is as following:

- `magic_calib_to_dl1.py` on real and MC data (if you use MCs produced with MMCS), to convert them into DL1 format
- if you use SimTelArray MCs, run `lst1_magic_mc_dl0_to_dl1.py` over them to convert them into DL1 format
- optionally, but recommended, `merge_hdf_files.py` to merge subruns and/or runs together
- `lst1_magic_stereo_reco.py` to add stereo parameters to the DL1 data (use `--magic-only` argument if the MC DL1 data contains LST-1 events)
- `lst1_magic_train_rfs.py` to train the RFs (energy, direction, classification) on train gamma MCs and protons
- `lst1_magic_dl1_stereo_to_dl2.py` to apply the RFs to stereo DL1 data (real and test MCs) and produce DL2 data
- `lst1_magic_create_irf.py` to create the IRF (use `magic_stereo` as `irf_type` in the configuration file)
- `lst1_magic_dl2_to_dl3.py` to create DL3 files, and `create_dl3_index_files.py` to create DL3 HDU and index files

## MAGIC+LST analysis: overview

MAGIC+LST-1 analysis starts from MAGIC calibrated data (\_Y\_ files), LST-1 DL1 data and SimTelArray DL0 data. The analysis flow is as following:

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

1) The very first step to reduce MAGIC-LST data is to have remote access/credentials to the IT Container, so provide one.
2) Once connected to the IT Container, install MAGIC-CTA-PIPE (e.g. in your home directory in the IT Container) following the tutorial here: https://github.com/cta-observatory/magic-cta-pipe
3) Now copy the scripts `lst1_magic_mc_dl0_to_dl1.py`, `magic_calib_to_dl1.py` and `setting_up_config_and_dir.py` to your workspace (e.g. /fefs/aswg/workspace/yourname) in the IT Container, as well as the files `config_general.yaml` and `MAGIC_runs.txt`.

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
    workspace_dir : "/fefs/aswg/workspace/yourname/MAYBE_ANOTHER_SUBDIRRECTORY"
    target_name   : "CrabTeste"
    MC_gammas     : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray/"
    MC_elec_helium: "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/"
    MC_prot_diff  : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/"
    target_dec_dir: "dec_2276"
    
general:
    SimTel_version: "v1.4"    
    focal_length  : "nominal"
    MAGIC_runs    : "MAGIC_runs.txt"  #If there is no MAGIC data, please fill this file with "0, 0"
```

The file `MAGIC_runs.txt` looks like that:  
```
2020_11_19,5093174
2020_11_19,5093175
2020_12_08,5093491
2020_12_08,5093492
2020_12_08,5093495
2020_12_08,5093496
2020_12_08,5093497
2020_12_16,5093711
2020_12_16,5093712
2020_12_16,5093713
2020_12_16,5093714
```
The columns here represent the night and run in which you want to select data. Please do not add blanck spaces in the rows, as these names will be used to i) find the MAGIC data in the IT Container and ii) create the subdirectories in your working directory. These two files are the only ones you need to modify in order to convert DL0 into DL1 data.


To convert the SimTelArray MCs data into DL1 format, you do the following:
> $ python setting_up_config_and_dir.py

```
The automatic list of telescope IDs is:
Name: LST1, LST2, LST3, LST4, MAGIC-I, MAGIC-II
ID  :   1     0     0     0      2       3
```

The script `setting_up_config_and_dir.py` does a series of things:
- Generates a configuration file called config_step1.yaml with MAGIC, LST, and telescope ID information.
- Creates a directory with your source name, in this case "CrabTeste", and several subdirectories inside it necessary for the rest of the data reduction.
- Links the MC data addresses to their respective subdirectories defined in the previous step.
- Runs the script `lst1_magic_mc_dl0_to_dl1.py` for each data file.

The default telescopes IDs are set as LST-1 ID = 1, MAGIC-I = 2, and MAGIC-II = 3. To change it, the user can do:
> $ python setting_up_config_and_dir.py --telescope_ids 1 2 0 0 3 4

where the sequence of telescopes is always LST1, LST2, LST3, LST4, MAGIC-I, MAGIC-II. So in this case, we have  
LST-1 ID = 1  
LST-2 ID = 2  
LST-3 ID = 0  
LST-4 ID = 0  
MAGIC-I ID = 3  
MAGIC-II ID = 4  
If the telescope ID is set to 0, this means that the telescope is not used in the analysis.
The full process can take several hours or a few days, depending on the size of your MC dataset.


## High level analysis

The folder [Notebooks](https://github.com/cta-observatory/magic-cta-pipe/tree/master/notebooks) contains Jupyter notebooks to perform checks on the IRF, to produce theta2 plots and SEDs. Note that the notebooks run with gammapy v0.20 or higher, therefore another conda environment is needed to run them, since the MAGIC+LST-1 pipeline at the moment depends on v0.19.
