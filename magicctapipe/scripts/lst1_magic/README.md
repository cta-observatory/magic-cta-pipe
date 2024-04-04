# Scripts for MAGIC+LST analysis

This folder contains the scripts to perform MAGIC+LST analysis in a semi-automatic way.

Each script can be called from the command line from anywhere in your system. Please run them with `-h` option for the first time to check what are the options available.


## Overview


MAGIC+LST analysis starts from MAGIC calibrated data (\_Y\_ files), LST  data level 1 (DL1) data and SimTelArray DL0 data, and our goal is to achieve data level 3 (DL3).

Behind the scenes, the semi-automatic scripts will run:
- `magic_calib_to_dl1` on real MAGIC data, to convert them into DL1 format.
- `merge_hdf_files.py` on MAGIC data to merge subruns and/or runs together.
- `lst1_magic_event_coincidence.py` to find coincident events between MAGIC and LST-1, starting from DL1 data.
- `lst1_magic_stereo_reco.py` to add stereo parameters to the DL1 data.
- `lst1_magic_train_rfs.py` to train the RFs (energy, direction, classification) on train gamma MCs and protons.
- `lst1_magic_dl1_stereo_to_dl2.py` to apply the RFs to stereo DL1 data (real and test MCs) and produce DL2 data.
- `lst1_magic_create_irf.py` to create the IRF.
- `lst1_magic_dl2_to_dl3.py` to create DL3 files, and `create_dl3_index_files.py` to create DL3 HDU and index files.

From DL3 on, the analysis is done with gammapy.

## Installation

1) The very first step to reduce MAGIC-LST data is to have remote access/credentials to the IT Container, so provide one. Once you have it, the connection steps are the following:

Authorized institute server (Client) &rarr;  ssh connection to CTALaPalma &rarr; ssh connection to cp01/02.

2) Once connected to the IT Container, install magic-cta-pipe (e.g. in your home directory in the IT Container) with the following commands (if you have mamba installed, we recommend yo uuse it instead of conda. The installation process will be much faster.):

```
git clone -b Torino_auto_MCP https://github.com/cta-observatory/magic-cta-pipe.git
cd magic-cta-pipe
conda env create -n magic-lst -f environment.yml
conda activate magic-lst
pip install .
```

## Analysis

### DL0 to DL1

In this step, we will convert the MAGIC and Monte Carlo (MC) Data Level (DL) 0 to DL1 (our goal is to reach DL3).

In your working IT Container directory (e.g. /fefs/aswg/workspace/yourname/yourprojectname), open the magic-lst environment with the command `conda activate magic-lst` and create the files `config_general.yaml`, `MAGIC_runs.txt` and `LST_runs.txt`.

The file `config_general.yaml` must contain the telescope IDs and the directories with the MC data, as shown below:
```
mc_tel_ids:
    LST-1: 1
    LST-2: 0  # If the telescope ID is set to 0, this means that this telescope is not used in the analysis.
    LST-3: 0
    LST-4: 0
    MAGIC-I: 2
    MAGIC-II: 3

directories:
    workspace_dir : "/fefs/aswg/workspace/yourname/yourprojectname/" 
    target_name   : "Crab"
    MC_gammas     : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray"
    MC_electrons  : "" 
    MC_helium     : "" 
    MC_protons    : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/Protons/dec_2276/sim_telarray"
    MC_gammadiff  : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/GammaDiffuse/dec_2276/sim_telarray/"

general:
    target_RA_deg : 83.629  # RA in degrees, the coordinates are useful only if the target name is not found in the catalogs.
    target_Dec_deg: 22.015  # Dec in degrees 
    SimTel_version: "v1.4"  
    LST_version   : "v0.9"
    LST_tailcut   : "tailcut84"
    focal_length  : "effective"
    MAGIC_runs    : "MAGIC_runs.txt"  #If there is no MAGIC data, please fill this file with "0, 0"
    LST_runs      : "LST_runs.txt"
    proton_train_fraction  : 0.8 # 0.8 means that 80% of the DL1 protons will be used for training the Random Forest  
    nsb           : [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # NSB = night sky background. This will be useful if NSB_matching = True
    env_name      : magic-lst
    cluster       : "SLURM"
    NSB_matching  : true
    
```

The file `MAGIC_runs.txt` looks like that:
```
2020_11_19,5093174
2020_11_19,5093175
2020_12_08,5093491
2020_12_08,5093492
2020_12_16,5093711
2020_12_16,5093712
2020_12_16,5093713
2020_12_16,5093714
2021_02_14,5094483
2021_02_14,5094484
2021_02_14,5094485
2021_02_14,5094486
2021_02_14,5094487
2021_02_14,5094488
2021_03_16,5095265
2021_03_16,5095266
2021_03_16,5095267
2021_03_16,5095268
2021_03_16,5095271
2021_03_16,5095272
2021_03_16,5095273
2021_03_16,5095277
2021_03_16,5095278
2021_03_16,5095281
2021_03_18,5095376
2021_03_18,5095377
2021_03_18,5095380
2021_03_18,5095381
2021_03_18,5095382
2021_03_18,5095383
```


The columns here represent the night and run in which you want to select data. Please **do not add blank spaces** in the rows, as these names will be used to i) find the MAGIC data in the IT Container and ii) create the subdirectories in your working directory. If there is no MAGIC data, please fill this file with "0,0". Similarly, the `LST_runs.txt` file looks like this:

```
2020_11_18,2923
2020_11_18,2924
2020_12_07,3093
2020_12_15,3265
2020_12_15,3266
2020_12_15,3267
2020_12_15,3268
2021_02_13,3631
2021_02_13,3633
2021_02_13,3634
2021_02_13,3635
2021_02_13,3636
2021_03_15,4069
2021_03_15,4070
2021_03_15,4071
2021_03_17,4125
```
Note that the LST nights appear as being one day before MAGIC's!!! This is because LST saves the date at the beginning of the night, while MAGIC saves it at the end. If there is no LST data, please fill this file with "0,0". These files are the only ones we need to modify in order to convert DL0 into DL1 data.

To convert the MAGIC data into DL1 format, you simply do:
> $ setting_up_config_and_dir -c config_general.yaml

The output in the terminal will be something like this:
```
*** Reducing DL0 to DL1 data - this can take many hours ***
Process name: yourprojectname_Crab
To check the jobs submitted to the cluster, type: squeue -n yourprojectname_Crab
```

The command `setting_up_config_and_dir` does a series of things:
- Creates a directory with your target name within the directory `yourprojectname` and several subdirectories inside it that are necessary for the rest of the data reduction. The main directories are:
```
/fefs/aswg/workspace/yourname/yourprojectname/Crab/
/fefs/aswg/workspace/yourname/yourprojectname/Crab/DL1
/fefs/aswg/workspace/yourname/yourprojectname/Crab/DL1/[subdirectories]
```
where [subdirectories] stands for several subdirectories containing the MAGIC subruns in the DL1 format.
- Generates a configuration file called `config_step1.yaml` with telescope ID information and adopted imaging/cleaning cuts, and puts it in the directory `[...]/yourprojectname/Crab/` created in the previous step.
- Links the MAGIC data addresses to their respective subdirectories defined in the previous steps.
- Runs the script `magic_calib_to_dl1.py` for each one of the linked data files.


You can check if this process is done with the following commands:

> $ squeue -n yourprojectname_Crab

or

> $ squeue -u your_user_name

Once it is done, all of the subdirectories in `/fefs/aswg/workspace/yourname/yourprojectname/Crab/DL1/` will be filled with files of the type `dl1_MX.RunXXXXXX.0XX.h5` for each MAGIC subrun. The next step of the conversion from DL0 to DL1 is to merge all the MAGIC data files such that in the end, we have only one datafile per night. To do so, we run the following command (always in the directory `yourprojectname`):

> $ merging_runs (-c config_general.yaml)

**The command inside parenthesis is not mandatory**. By the way, it is better if you don't use it unless you know what you are doing. 
The output in the terminal will be something like this:
```
***** Generating merge bashscripts...  
***** Running merge_hdf_files.py in the MAGIC data files...  
Process name: merging_Crab  
To check the jobs submitted to the cluster, type: squeue -n merging_Crab
```

This script will merge the MAGIC data files in the following order:
- MAGIC subruns are merged into single runs.
- MAGIC I and II runs are merged (only if both telescopes are used, of course).
- All runs in specific nights are merged, such that in the end we have only one datafile per night.

### Coincident events and stereo parameters on DL1

To find coincident events between MAGIC and LST, starting from DL1 data, we run the following command in the working directory:

> $ coincident_events (-c config_general.yaml)

This script creates the file config_coincidence.yaml containing the telescope IDs and the following parameters:
```
event_coincidence:
    timestamp_type_lst: "dragon_time"  # select "dragon_time", "tib_time" or "ucts_time"
    pre_offset_search: true
    n_pre_offset_search_events: 100
    window_half_width: "300 ns"
    time_offset:
        start: "-10 us"
        stop: "0 us
```

It then links the LST data files to the output directory [...]DL1/Observations/Coincident, and runs the script lst1_magic_event_coincidence.py in all of them.

Once it is done, we add stereo parameters to the MAGIC+LST coincident DL1 files by running:

> $ stereo_events (-c config_general.yaml)

This script creates the file config_stereo.yaml with the following parameters:
```
stereo_reco:
    quality_cuts: "(intensity > 50) & (width > 0)"
    theta_uplim: "6 arcmin"
```

It then creates the output directories for the DL1 with stereo parameters [...]DL1/Observations/Coincident__stereo/SEVERALNIGHTS, and then runs the script lst1_magic_stereo_reco.py in all of the coincident DL1 files. The stereo DL1 files are then saved in these directories.

### Random forest and DL1 to DL2

TBD.

### Instrument response function and DL3

TBD.

## High-level analysis

Since the DL3 may have only a few MBs, it is typically convenient to download it to your own computer at this point. It will be necessary to have astropy and gammapy (version > 0.20) installed before proceeding. 

We prepared a [Jupyter Notebook](https://github.com/ranieremenezes/magic-cta-pipe/blob/master/magicctapipe/scripts/lst1_magic/SED_and_LC_from_DL3.ipynb) that quickly creates a counts map, a significance curve, an SED, and a light curve. You can give it a try. 

The folder [Notebooks](https://github.com/cta-observatory/magic-cta-pipe/tree/master/notebooks) contains Jupyter notebooks to perform checks on the IRF, to produce theta2 plots and SEDs. Note that the notebooks run with gammapy v0.20 or higher, while the gammapy version adopted in the MAGIC+LST-1 pipeline is v0.19.
