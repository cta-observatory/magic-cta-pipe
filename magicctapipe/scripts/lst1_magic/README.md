# Script for MAGIC and MAGIC+LST analysis

This folder contains scripts to perform MAGIC-only or MAGIC+LST analysis.

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

2) Once connected to the IT Container, install MAGIC-CTA-PIPE (e.g. in your home directory in the IT Container) following the tutorial here: https://github.com/ranieremenezes/magic-cta-pipe

### DL0 to DL1 step

In this step we will convert the MAGIC and Monte Carlo (MC) Data Level (DL) 0 to DL1 (our goal is to reach DL3).

3) Now copy all the python scripts available here to your preferred directory (e.g. /fefs/aswg/workspace/yourname/yourprojectname) in the IT Container, as well as the files `config_general.yaml`, `MAGIC_runs.txt` and `LST_runs.txt`.

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
    workspace_dir : "/fefs/aswg/workspace/yourname/yourprojectname/" #Always put the last "/"!!!
    target_name   : "CrabTeste"
    MC_gammas     : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/sim_telarray/" #Always put the last "/"!!!
    MC_electrons  : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/Electrons/sim_telarray/" #Always put the last "/"!!!
    MC_helium     : "/fefs/aswg/data/mc/DL0/LSTProd2/TestDataset/Helium/sim_telarray/" #Always put the last "/"!!!
    MC_protons    : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/Protons/dec_2276/sim_telarray/" #Always put the last "/"!!!
    MC_gammadiff  : "/fefs/aswg/data/mc/DL0/LSTProd2/TrainingDataset/GammaDiffuse/dec_2276/sim_telarray/" #Always put the last "/"!!!
    
general:
    SimTel_version: "v1.4"    #This is the version of the SimTel used in the MC simulations
    focal_length  : "nominal" 
    MAGIC_runs    : "MAGIC_runs.txt"  #If there is no MAGIC data, please fill the MAGIC_runs.txt file with "0, 0"
    LST_runs      : "LST_runs.txt"  
    proton_train  : 0.8 # 0.8 means that 80% of the DL1 proton files will be used for training the Random Forest
```

The file `MAGIC_runs.txt` looks like that:  
```
2020_11_19,5093174
2020_11_19,5093175
2020_12_08,5093491
2020_12_16,5093711
2020_12_16,5093712
2020_12_16,5093713
2020_12_16,5093714
2021_02_14,5094483
2021_02_14,5094484
2021_02_14,5094485
2021_02_14,5094486
2021_02_14,5094487
2021_03_16,5095265
2021_03_16,5095266
2021_03_16,5095267
2021_03_18,5095376
```


The columns here represent the night and run in which you want to select data. Please do not add blanck spaces in the rows, as these names will be used to i) find the MAGIC data in the IT Container and ii) create the subdirectories in your working directory. If there is no MAGIC data, please fill this file with "0,0". Similarly, the `LST_runs.txt` file looks like:

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
Note that the LST nights are appear as being one day before MAGIC's!!! This is because LST saves the date at the beggining of the night, while MAGIC saves it at the end. If there is no LST data, please fill this file with "0,0". These files are the only ones we need to modify in order to convert DL0 into DL1 data.

In this analysis, we use a wobble of 0.4$^{\circ}$!

To convert the MAGIC and SimTelArray MCs data into DL1 format, you first do the following:
> $ python setting_up_config_and_dir.py

```
***** Linking MC paths - this may take a few minutes ******
*** Reducing DL0 to DL1 data - this can take many hours ***
Process name: yourprojectnameCrabTeste
To check the jobs submitted to the cluster, type: squeue -n yourprojectnameCrabTeste
```
Note that this script can be run as  
> $ python setting_up_config_and_dir.py --partial-analysis onlyMAGIC  

or  

> $ python setting_up_config_and_dir.py --partial-analysis onlyMC  

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

in the terminal. Once it is done, all of the subdirectories in `/fefs/aswg/workspace/yourname/yourprojectname/CrabTeste/DL1/` will be filled with files of the type `dl1_[...]_LST1_MAGIC1_MAGIC2_runXXXXXX.h5` for the MCs and `dl1_MX.RunXXXXXX.0XX.h5` for the MAGIC runs. The next step of the conversion of DL0 to DL1 is to split the DL1 MC proton sample into "train" and "test" datasets (these will be used later in the Random Forest event classification and to do some diagnostic plots), and to merge all the MAGIC data files such that in the end we have only one datafile per night. To do so, we run the following script:

> $ python merging_runs_and_spliting_training_samples.py  

```
***** Spliting protons into 'train' and 'test' datasets...  
***** Generating merge bashscripts...  
***** Running merge_hdf_files.py in the MAGIC data files...  
Process name: merging_CrabTeste  
To check the jobs submitted to the cluster, type: squeue -n merging_CrabTeste
```

This script will slice the proton MC sample according to the entry "proton_train" in the "config_general.yaml" file, and then it will merge the MAGIC data files in the following order:
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
event_coincidence:
    timestamp_type_lst: "dragon_time"  # select "dragon_time", "tib_time" or "ucts_time"
    window_half_width: "300 ns"
    time_offset:
        start: "-10 us"
        stop: "0 us"
```

It then links the real LST data files to the output directory [...]DL1/Observations/Coincident, and runs the script lst1_magic_event_coincidence.py in all of them.

Once it is done, we add stereo parameters to the MAGIC+LST coincident DL1 data by running:

> $ python stereo_events.py

This script creates the file config_stereo.yaml with the follwoing parameters:
```
stereo_reco:
    quality_cuts: "(intensity > 50) & (width > 0)"
    theta_uplim: "6 arcmin"
```

It then creates the output directories for the DL1 with stereo parameters [...]DL1/Observations/Coincident_stereo/SEVERALNIGHTS and [...]/DL1/MC/GAMMAorPROTON/Merged/StereoMerged, and then runs the script lst1_magic_stereo_reco.py in all of the coincident DL1 files. The stereo DL1 files for MC and real data are then saved in these directories.

### Random forest

Once we have the DL1 stereo parameters for all real and MC data, we can train the Random Forest:

> $ python RF.py

This script creates the file config_RF.yaml with several parameters related to the energy regressor, disp regressor and event classifier, and then computes the RF (energy, disp, and classifier) based on the merged-stereo MC diffuse gammas and training proton samples by calling the script lst1_magic_train_rfs.py. The results are saved in [...]/DL1/MC/RFs.

Once it is done, we can finally convert our DL1 stereo data files into DL2 by running:

> $ python DL1_to_DL2.py

This script runs `lst1_magic_dl1_stereo_to_dl2.py` on all DL1 stereo files, which applies the RFs saved in [...]/DL1/MC/RFs to stereo DL1 data (real and test MCs) and produce DL2 real and MC data. The results are saved in [...]/DL2/Observations and [...]/DL2/MC.

### Instrument response function and DL3

Once the previous step is done, we compute the IRF with

> $ python IRF.py

which creates the configuration file config_IRF.yaml with several parameters. The main of which are shown below:

```
[...]
quality_cuts: "disp_diff_mean < 0.22"
event_type: "software"  # select "software", "software_only_3tel", "magic_only" or "hardware"
weight_type_dl2: "simple"  # select "simple", "variance" or "intensity"
[...]
gammaness:
    cut_type: "dynamic"  # select "global" or "dynamic"
    [...]

theta:
    cut_type: "global"  # select "global" or "dynamic"
    global_cut_value: "0.2 deg"  # used for the global cut
    [...]
```

It then runs the script lst1_magic_create_irf.py over the DL2 MC gammas, generating the IRF and saving it at [...]/IRF.

Optionally, but recommended, we can run the "diagnostic.py" script with:

> $ python diagnostic.py

This will create several diagnostic plots (gammaness, effective area, angular resolution, energy resolution, migration matrix, energy bias and gamma-hadron classification comparisons. All of these plots will be saved on the directory defined on "target_name" in the config_general.yaml file.

After the IRF, we run the DL2-to-DL3 conversion by doing:

> $ python DL2_to_DL3.py

which will save the DL3 files in the directory [...]/DL3. Finally, the last script to run is `create_dl3_index_files.py`. Since it is very fast, we can simply run it directly in the interactive mode by doing:

> $ conda run -n magic-lst python create_dl3_index_files.py --input-dir ./CrabTeste/DL3

That's it. Now you can play with the DL3 data using the high-level notebooks.

## High level analysis

The folder [Notebooks](https://github.com/cta-observatory/magic-cta-pipe/tree/master/notebooks) contains Jupyter notebooks to perform checks on the IRF, to produce theta2 plots and SEDs. Note that the notebooks run with gammapy v0.20 or higher, therefore another conda environment is needed to run them, since the MAGIC+LST-1 pipeline at the moment depends on v0.19.
