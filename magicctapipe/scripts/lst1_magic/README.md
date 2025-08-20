# Scripts for MAGIC+LST analysis

This folder contains the scripts to perform MAGIC+LST analysis in a semi-automatic way.

Each script can be called from the command line from anywhere in your system. Please run them with `-h` option for the first time to check what are the options available.


## Overview


MAGIC+LST analysis starts from MAGIC calibrated data (\_Y\_ files), LST  data level 1 (DL1) data and SimTelArray DL0 data, and our goal is to achieve data level 3 (DL3).

Behind the scenes, the semi-automatic scripts will run:
- `magic_calib_to_dl1` on real MAGIC data, to convert them into DL1 format.
- `merge_hdf_files` on MAGIC data to merge subruns and/or runs together.
- `lst1_magic_event_coincidence` to find coincident events between MAGIC and LST-1, starting from DL1 data.
- `lst1_magic_stereo_reco` to add stereo parameters to the DL1 data.
- `lst1_magic_train_rfs` to train the RFs (energy, direction, classification) on train gamma MCs and protons.
- `lst1_magic_dl1_stereo_to_dl2` to apply the RFs to stereo DL1 data (real and test MCs) and produce DL2 data.
- `lst1_magic_create_irf` to create the IRF.
- `lst1_magic_dl2_to_dl3` to create DL3 files, and `create_dl3_index_files` to create DL3 HDU and index files.

From DL3 on, the analysis is done with gammapy.


## Analysis

During the analysis, some files (i.e., bash scripts, lists of sources and runs) are automatically produced by the scripts and are saved in your working directory. These files are necessary for the subsequent steps in the analysis chain. It is therefore mandatory to always launch the scripts from the same working directory so that the output files stored there can be correctly assigned as input files at the subsequent analysis steps.

### DL0 to DL1

In this step, we will convert the MAGIC Calibrated data to Data Level (DL) 1 (our goal is to reach DL3).

In your working IT Container directory (i.e., `workspace_dir`), open your environment with the command `conda activate {env_name}` and update the file `config_auto_MCP.yaml` according to your analysis. If you need non-standard parameters (e.g., for the cleaning), take care that the `resources/config.yaml` file gets installed when you install the pipeline, so you will have to copy it, e.g. in your workspace, modify it and put the path to this new file in the `config_auto_MCP.yaml` (this way you don't need to install again the pipeline).

The file `config_auto_MCP.yaml` must contain parameters for data selection and some information on the night sky background (NSB) level and software versions:

```

directories:
    workspace_dir : "/fefs/aswg/workspace/elisa.visentin/auto_MCP_PR/"  # Output directory where all the data products will be saved.
    
    
data_selection:
    source_name_database: "CrabNebula"  # MUST BE THE SAME AS IN THE DATABASE; Set to null to process all sources in the given time range.
    source_name_output: 'Crabtest'  # Name tag of your target. Used only if source_name_database != null.
    time_range : True  # Search for all runs in a LST time range (e.g., 2020_01_01 -> 2022_01_01).
    min : "2023_11_17"
    max : "2024_03_03"   
    date_list : ['2020_12_15','2021_03_11']  # LST list of days to be processed (only if time_range=False), format: YYYY_MM_DD.
    skip_LST_runs: [3216,3217]  # LST runs to ignore.
    skip_MAGIC_runs: [5094658]  # MAGIC runs to ignore.
    
general:
    base_config_file: ''    # path + name to a custom MCP config file. If not provided, the default config.yaml file will be used 
    LST_version   : "v0.10" # check the `processed_lstchain_file` version in the LST database!
    LST_tailcut   : "tailcut84"
    simtel_nsb    : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray/node_theta_14.984_az_355.158_/output_v1.4/simtel_corsika_theta_14.984_az_355.158_run10.simtel.gz" # simtel file (DL0) to evaluate NSB
    lstchain_modified_config : true # use_flatfield_heuristic = True to evaluate NSB    
    nsb           : [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    env_name      : magic-lst  # name of the conda environment to be used to process data.
    cluster       : "SLURM"  # cluster management system on which data are processed. At the moment we have only SLURM available, in the future maybe also condor (PIC, CNAF).
    

```

WARNING: Only the runs for which the `LST_version` parameter matches the `processed_lstchain_file` version in the LST database (i.e., the version used to evaluate the NSB level; generally the last available and processable version of a run) will be processed.

WARNING: `env_name` must be the same as the name of the environment in which you installed this version of the pipeline

Now that the configuration file is ready, let's create a list with all the MAGIC+LST1 runs for the time window (or list of nights) defined on the `config_auto_MCP.yaml` file:

> $ list_from_h5 -c config_auto_MCP.yaml

The output in the terminal should look like this:
```
Cleaning pre-existing *_LST_runs.txt and *_MAGIC_runs.txt files
Source: XXX
Finding LST runs...
Source: XXX
Finding MAGIC runs...
```
And it will save the files `{TARGET}_LST_runs.txt`, `{TARGET}_MAGIC_runs.txt`, and `list_sources.dat` (i.e., the list of all the sources found in the database according to both custom and default settings) in your current working directory. In case no runs are found for MAGIC and/or LST (for a source and a given time range/list of dates), a warning will be printed and no output text file will be produced for the given source and telescope(s).

At this point, we can convert the MAGIC data into DL1 format with the following command:
> $ dl1_production -c config_auto_MCP.yaml

The output in the terminal will be something like this:
```
*** Converting Calibrated into DL1 data ***
Process name: {source}
To check the jobs submitted to the cluster, type: squeue -n {source}
```

The command `dl1_production` does a series of things:

- Creates a directory with the target name within the directory `yourprojectname/{MCP_version}` and several subdirectories inside it that are necessary for the rest of the data reduction. The main directories are:
```
workspace_dir/VERSION/
workspace_dir/VERSION/{source}/DL1
workspace_dir/VERSION/{source}/DL1/[subdirectories]
```
where [subdirectories] stands for several subdirectories containing the MAGIC subruns in the DL1 format.
- Generates a configuration file called `config_DL0_to_DL1.yaml` with telescope ID information and adopted imaging/cleaning cuts, and puts it in the directory `[...]/yourprojectname/VERSION/{source}/` created in the previous step.
- Links the MAGIC data addresses to their respective subdirectories defined in the previous steps.
- Runs the script `magic_calib_to_dl1.py` for each one of the linked data files.


You can check if this process is done with the following commands:

> $ squeue -n {source}

or

> $ squeue -u your_user_name

Once it is done, all of the subdirectories in `workspace_dir/VERSION/{source}/DL1` will be filled with files of the type `dl1_MX.RunXXXXXX.0XX.h5` for each MAGIC subrun. 

WARNING: some of these jobs could fail due to 'broken' input files: before moving to the next step, check for failed jobs (through `job_accounting` and/or log files) and remove the output files produced by these failed jobs (these output files will generally have a very small size, lower than few kB, and cannot be read in the following steps)

The next step of the conversion from calibrated to DL1 is to merge all the MAGIC data files such that in the end, we have only one datafile per night. To do so, we run the following command (always in the directory `yourprojectname`):

> $ merging_runs (-c config_auto_MCP.yaml)

The output in the terminal will be something like this:
```
***** Generating merge_MAGIC bashscripts...  
***** Running merge_hdf_files.py in the MAGIC data files...  
Process name: merging_{source} 
To check the jobs submitted to the cluster, type: squeue -n merging_{source} 
```

This script will merge MAGIC-I (and MAGIC-II) subruns into runs.

### Coincident events and stereo parameters on DL1

To find coincident events between MAGIC and LST, starting from DL1 data, we run the following command in the working directory:

> $ coincident_events (-c config_auto_MCP.yaml)

This script creates the file `config_coincidence.yaml` containing both the telescope IDs and the coincidence parameters listed in the general `config.yaml` file (the one in `magicctapipe/resources`).

Then, matches LST and MAGIC dates and links the LST data files to the output directory `[...]/DL1Coincident`; eventually, it runs the script `lst1_magic_event_coincidence.py` in all of them.

Once it is done, we add stereo parameters to the MAGIC+LST coincident DL1 files by running:

> $ stereo_events (-c config_auto_MCP.yaml)

This script creates the file `config_stereo.yaml` containing both the telescope IDs and the stereo parameters listed in the general `config.yaml` file (the one in `magicctapipe/resources`).

It then creates the output directories for the DL1 with stereo parameters `[...]/DL1Stereo`, and then runs the script `lst1_magic_stereo_reco.py` in all of the coincident DL1 files. The stereo DL1 files are then saved in these directories.

Eventually, to merge DL1 stereo (LST) subruns into runs, we run the `merge_stereo.py` script, whose output will be saved in `[...]/DL1Stereo/Merged`:

> $ merge_stereo (-c config_auto_MCP.yaml)

### DL1 to DL2

TBD.

### DL2 to DL3

TBD.

## High-level analysis

Since the DL3 may have only a few MBs, it is typically convenient to download it to your own computer at this point. It will be necessary to have astropy and gammapy (version >= 0.20) installed before proceeding. 

The folder [Notebooks](https://github.com/cta-observatory/magic-cta-pipe/tree/master/notebooks) contains Jupyter notebooks to perform checks on the IRF, to produce theta2 plots and SEDs.


## For mainteiners (MAGIC and LST databases)

To create and update the MAGIC and LST databases (from the one produced by AB and FDP) you should use the scripts in `database_production`

- `create_lst_table`: creates the LST database (1 row per LST run) by dropping some columns from the parent one (AB, FDP) and adding columns for NSB value (default: NaN), lstchain available versions, most recent lstchain version, processed file and NSB error codes (default: -1). It could also be used to update the given database, possibly selecting a given time range from the parent databases (by the -b and -e parameters, which stand for begin and end date of the range). Launched as `create_lst_table (-b YYYYMMDD -e YYYYMMDD)`

- `lstchain_version`: this scripts loop over all the rows of the database, estract date and run number from the table and look for the data stored on the IT (i.e., which version of lstchain has been used to process a run). It evaluates all the versions used to process a run and the most recent MCP-compatible one according to a hard-coded, ordered list. Launched as `lstchain_version`

- `nsb_level`: evaluates, for the last (MCP compatible) version of every LST run, the respective NSB value (i.e., the median over the NSB estimated by lstchain over a sub-set of sub-runs per run). This scripts launch a set of jobs (one per run; each job calls the `LSTnsb.py` script) and each jobs produces an output txt file containing a string like `date,run,NSB`; in the title of these files, both the run number and the NSB range are indicated (0.5=(0,0.75), 1.0=(0.75, 1.25),...., 2.5=(2.25,2.75), 3.0=(2.75,3.25), `high`=(3.25,Infinity) ). To limit the number of simultaneous jobs running on SLURM, the script requires that you provide a begin and a end date (-b and -e parameters) in the options. Launched as `nsb_level -c config_auto_MCP.yaml -b YYYY_MM_DD -e YYYY_MM_DD`

- `LSTnsb`: called by `nsb_level`, it gathers all the subruns for a run, evaluates the NSB for a subset of them (using the lstchain `calculate_noise_parameters` function), evaluates the median over these values and the approximate NSB level according to the list provided in `config_auto_MCP.yaml` (e.g., 0.5, 1.0, 1.5, ...., 2.5, 3.0, `high`) and then creates one txt file per run. These files contain the value of the NSB (i.e., the median over subruns) and are needed to fill the `nsb` column in the LST database. Launched as `LSTnsb (-c MCP_config) -i run -d date -l lstchain_config (-s N_subruns)`

- `nsb_to_h5`: this script reads the txt files created by `nsb_level` to know the NSB value for each run. This value is used to fill the `nsb` column of the LST database at the location of the respective run number. It also updates the error codes (0: NSB lower than 3.0, 1: NSB could not be evaluated, 2: NSB higher than 3.0). Launched as `nsb_to_h5`

- `update_magic_db`: this script updates (or creates, if it does not exist) the MAGIC database from a time range provided by the user (-m and -M parameters, which stand for minimum and maximum date). Not to accidentally destroy the current database, the updated database is saved as a new file instead of overwriting the current one. Launched as `update_magic_db -m YYYYMMDD -M YYYYMMDD`

- `job_accounting`: this script (in `semi_automatic_scripts` directory) allows to track progress of the submitted jobs, in particular listing errors. If you don-t use the `--no-accounting` option, it also provides basic resource statistics (CPU and memory) of the completed jobs. Finally, it can be also used to update the database files with the progress of data processing. Launched as `job_accounting (-c config) (-d data_level) (-v MCP_version) (--no-accounting) (-r h5_database)`

- `check_MAGIC_runs`: this script checks the MAGIC data stored on the IT (i.e., missing and existing data) in a given time range (-m and -M parameters, which stand for minimum and maximum date). Launched as `check_MAGIC_runs -m YYYYMMDD -M YYYYMMDD`


