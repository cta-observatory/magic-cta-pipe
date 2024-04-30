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

2) Once connected to the IT Container, install magic-cta-pipe (e.g. in your home directory in the IT Container) with the following commands (if you have mamba installed, we recommend you to use it instead of conda. The installation process will be much faster.):

```
git clone -b Torino_auto_MCP https://github.com/cta-observatory/magic-cta-pipe.git
cd magic-cta-pipe
mamba env create -n magic-lst -f environment.yml
mamba activate magic-lst
pip install .
```

## Analysis

During the analysis, some files are automatically produced by the scripts and are saved in your working directory. These files are necessary for the subsequent steps in the analysis chain. It is therefore mandatory to always launch the scripts from the same working directory so that the output files stored there can be correctly assigned as input files at the subsequent analysis steps.

### DL0 to DL1

In this step, we will convert the MAGIC and Monte Carlo (MC) Data Level (DL) 0 to DL1 (our goal is to reach DL3).

In your working IT Container directory (e.g. /fefs/aswg/workspace/yourname/yourprojectname), open the magic-lst environment with the command `conda activate magic-lst` and update the file `config_general.yaml` according to your analysis.

The file `config_general.yaml` must contain the telescope IDs, the directories with the MC data (mandatory only if NSB_matching = False), the data selection, and some information on the night sky background (NSB) level and software versions:

```
mc_tel_ids:
    LST-1: 1
    LST-2: 0
    LST-3: 0
    LST-4: 0
    MAGIC-I: 2
    MAGIC-II: 3

directories:
    workspace_dir : "/fefs/aswg/workspace/yourname/yourprojectname/"  # Output directory where all the data products will be saved.
    # MC paths below are ignored if you set NSB_matching = true.
    MC_gammas     : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray"  # set to "" if you don't want to process these Monte Carlo simulations.
    MC_electrons  : "" 
    MC_helium     : "" 
    MC_protons    : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/Protons/dec_2276/sim_telarray" 
    MC_gammadiff  : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/GammaDiffuse/dec_2276/sim_telarray/"
    
data_selection:
    source_name_database: "CrabNebula"  # MUST BE THE SAME AS IN THE DATABASE; Set to null to process all sources in the given time range.
    source_name_output: 'Crabtest'  # Name tag of your target. Used only if source_name_database != null.
    target_RA_deg : 83.629  # RA in degrees; Please set it to null if source_name_database=null.
    target_Dec_deg: 22.015  # Dec in degrees; Please set it to null if source_name_database=null.
    time_range : True  # Search for all runs in a LST time range (e.g., 2020_01_01 -> 2022_01_01).
    min : "2023_11_17"
    max : "2024_03_03"   
    date_list : ['2020_12_15','2021_03_11']  # LST list of days to be processed (only if time_range=False), format: YYYY_MM_DD.
    skip_LST_runs: [3216,3217]  # LST runs to ignore.
    skip_MAGIC_runs: [5094658]  # MAGIC runs to ignore.
    
general:
    SimTel_version: "v1.4"  
    LST_version   : "v0.10"
    LST_tailcut   : "tailcut84"
    focal_length  : "effective"
    proton_train_fraction  : 0.8  # 0.8 means that 80% of the DL1 protons will be used for training the Random Forest.
    nsb           : [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    env_name      : auto_MCP_PR  # name of the conda environment to be used to process data.
    cluster       : "SLURM"  # cluster management system on which data are processed. At the moment we have only SLURM available, in the future maybe also condor (PIC, CNAF).
    NSB_matching  : true  # Set to false to process also the MCs. Set to true if adequate MC productions (DLx) are already available on the IT Container.
    NSB_MC        : 0.5  # extra noise in dim pixels used to process MCs; e.g., you could put here the average NSB value of the processed LST runs. Ignored if NSB_matching=true.
    
    
```

Now that the configuration file is ready, let's create a list with all the MAGIC+LST1 runs for the time window (or list of nights) defined on the config_general.yaml file:

> $ list_from_h5 -c config_general.yaml

At this point, we can convert the MAGIC data into DL1 format with the following command:
> $ setting_up_config_and_dir -c config_general.yaml

The output in the terminal will be something like this:
```
*** Converting DL0 into DL1 data ***
Process name: yourprojectname
To check the jobs submitted to the cluster, type: squeue -n yourprojectname
This process will take about 10 min to run if the IT cluster is free.
```

The command `setting_up_config_and_dir` does a series of things:
- Collects the txt files produced above into a single txt file per NSB bin (e.g., `LST_1.5_.txt`), whose content is a list of all the `date,runs` couples associated to this background value.
- According to the date of the selected MAGIC runs, each run is associated (actually, each day, but here it is done run-wise) to the corresponding MAGIC observation period ST_XXXX.
- Creates a directory with the target name within the directory `yourprojectname` and several subdirectories inside it that are necessary for the rest of the data reduction. The main directories are:
```
/fefs/aswg/workspace/yourname/yourprojectname/Crab/
/fefs/aswg/workspace/yourname/yourprojectname/Crab/VERSION/DL1
/fefs/aswg/workspace/yourname/yourprojectname/Crab/VERSION/DL1/ST{}
/fefs/aswg/workspace/yourname/yourprojectname/Crab/VERSION/DL1/[subdirectories]
```
where [subdirectories] stands for several subdirectories containing the MAGIC subruns in the DL1 format.
- Generates a configuration file called `config_DL0_to_DL1.yaml` with telescope ID information and adopted imaging/cleaning cuts, and puts it in the directory `[...]/yourprojectname/Crab/` created in the previous step.
- Links the MAGIC data addresses to their respective subdirectories defined in the previous steps.
- Runs the script `magic_calib_to_dl1.py` for each one of the linked data files.


You can check if this process is done with the following commands:

> $ squeue -n yourprojectname_Crab

or

> $ squeue -u your_user_name

Once it is done, all of the subdirectories in `/fefs/aswg/workspace/yourname/yourprojectname/Crab/VERSION/DL1/` will be filled with files of the type `dl1_MX.RunXXXXXX.0XX.h5` for each MAGIC subrun. The next step of the conversion from DL0 to DL1 is to merge all the MAGIC data files such that in the end, we have only one datafile per night. To do so, we run the following command (always in the directory `yourprojectname`):

> $ merging_runs (-c config_general.yaml)

**The command inside parenthesis is not mandatory if you are running the command in the working directory**. By the way, it is better if you don't use it unless you know what you are doing. 
The output in the terminal will be something like this:
```
***** Generating merge_MAGIC bashscripts...  
***** Running merge_hdf_files.py in the MAGIC data files...  
Process name: merging_Crab  
To check the jobs submitted to the cluster, type: squeue -n merging_Crab
This process will take about 10 to 30 min to run.
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

Then, for each NSB level, it reads the corresponding txt file generated by the second script and, taking into account the day (and, as a consequence the MAGIC period of the joint MAGIC runs) links the LST data files for these runs to the output directory [...]DL1Coincident/ST{}/NSB{}, and runs the script lst1_magic_event_coincidence.py in all of them.

Once it is done, we add stereo parameters to the MAGIC+LST coincident DL1 files by running:

> $ stereo_events (-c config_general.yaml)

This script creates the file config_stereo.yaml with the following parameters:
```
stereo_reco:
    quality_cuts: "(intensity > 50) & (width > 0)"
    theta_uplim: "6 arcmin"
```

It then creates the output directories for the DL1 with stereo parameters [...]DL1CoincidentStereo/ST{}/NSB{}, and then runs the script lst1_magic_stereo_reco.py in all of the coincident DL1 files. The stereo DL1 files are then saved in these directories.

### Random forest and DL1 to DL2

TBD.

### Instrument response function and DL3

TBD.

## High-level analysis

Since the DL3 may have only a few MBs, it is typically convenient to download it to your own computer at this point. It will be necessary to have astropy and gammapy (version > 0.20) installed before proceeding. 

We prepared a [Jupyter Notebook](https://github.com/ranieremenezes/magic-cta-pipe/blob/master/magicctapipe/scripts/lst1_magic/SED_and_LC_from_DL3.ipynb) that quickly creates a counts map, a significance curve, an SED, and a light curve. You can give it a try. 

The folder [Notebooks](https://github.com/cta-observatory/magic-cta-pipe/tree/master/notebooks) contains Jupyter notebooks to perform checks on the IRF, to produce theta2 plots and SEDs. Note that the notebooks run with gammapy v0.20 or higher, while the gammapy version adopted in the MAGIC+LST-1 pipeline is v0.19.


## For mainteiners (creation of MAGIC adn LST databases)

To create and update the MAGIC and LST databases (from the one produced by AB and FDP) you should use the scripts in `database_production`

- `create_lst_table`: creates the LST database (1 row per LST run) by dropping some columns from the parent one (AB, FDP) and adding columns for NSB value (NaN by default), lstchain version (one column per version, False by default) and error codes (NaN by default). Launched as `python create_lst_table.py`

- `nsb_level`: evaluates, for every LST run, the respective NSB value (i.e., the median over the NSB estimated by lstchain over approx. 25 sub-runs per run). This scripts launch a set of jobs (one per run; each job calls the `LSTnsb` script) and each jobs produces an output txt file containing a string like `date,run,NSB`; in the title of these files, both the run number and the NSB range are indicated (0.5=(0,0.75), 1.0=(0.75, 1.25),...., 2.5=(2.25,2.75), 3.0=(2.75,3.25), `high`=(3.25,Infinity) ). To limit the number of simultaneous jobs running on SLURM, you should always provide a begin and a end date (format YYYY_MM_DD) in the options. Launched as `python nsb_level.py -c config_general.yaml -b begin_date -e end_date`

- `LSTnsb`: called by `nsb_level`, it gathers all the subruns for a run, evaluates the NSB for approx. 25 of them (using the lstchain `calculate_noise_parameters` function), evaluates the median over these values and the approximate NSB level (0.5, 1.0, 1.5, ...., 2.5, 3.0, `high`) and then creates one txt file per run. These files contain the value of the NSB (i.e., the median over subruns) and are needed to fill the database `nsb` column

- `nsb_to_h5`: this script reads the txt files created by `nsb_level` to know the NSB value for each run. This value is used to fill the `nsb` column of the database at the location of the respective run number. It also put '000' as an error code in case the NSB is NaN (i.e., not evaluated for the LST run), '001' in case NSB>3.0 (too high NSB to be processed with a standard analysis!). Launched as `python nsb_to_h5.py`

- `lstchain_version`: this scripts loop over all the rows of the database, estract date and run number from the table and look for the data saved in the IT (i.e., which version of lstchain has been used to process a run). For each run, it sets to True the lstchain_0.9(0.10) cell if this run has been processed up to DL1 with lstchain 0.9(0.10). It sets error code '002' in case none of the two versions has been used to process the run. Launched as `python lstchain_version.py`

Error codes:

- 000: no NSB

- 001: NSB>3.0

- 002: neither 0.9 nor 0.10 lstchain version
