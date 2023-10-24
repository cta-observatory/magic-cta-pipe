Scripts to be used to analyse real data by matching them to the available MCs (according to NSB level)
The lstchain 'lstchain_standard_config.json' is needed (in the same directory) to run them
Based on an upgraded version of the pipeline (scripts, modules, environment), which is needed to run these scripts: see "Semi-automatic MCP and expansion towards 4 LSTs - Torino team update" PR. So, to be merged after this PR.

This is a temporary branch (to store these scripts): A new branch and a PR will be created from the master after the '4 LSTs' PR merge.

nsb_level.py to be launched (bash script) at the beginning of the analysis, to classify LST runs according to NSB level. Then, collect_nsb.py to create the 'NSB-wise' LST lists. Then, standard semi-automatic analysis (see README in the 4-LSTs PR) on real data 


# TODO: 

1. Database (joint observations)

# Quick start tutorial

Update the config_h5.yaml file with the time range, target name and bad runs.

```python list_from_h5.py``` to create the lists with the runs

```python nsb_level.py``` If there is no file called "config_general" or if you want to set another name to the config file, you can use the option "-c PG1553_config_general.yaml". This is actually valid for all scripts in the NSB branch. This script computes the NSB level for each LST run found by the first script. For a single LST run, it takes around 50 min to compute the NSB level, so we launch the jobs in parallel. For each job, it creates one txt file with information about the NSB. 

```python collect_nsb.py``` This script stacks the information from the txt files created above separated by NSB level.

```python nsb_setting_up_config_and_dir.py``` Creates the directories for DL1 MAGIC data separated by observation period and processes MAGIC data up to DL1.

To merge the subruns into runs, the M1 and M2 runs, and then runs into nights, we do:

```python nsb_merge_subruns.py```

```python nsb_merge_M1_M2_runs.py```

```python nsb_merge_M1_M2_night.py```


```python nsb_coincident_events.py``` Find the MAGIC-LST coincident events and organize them by NSB level.

```python nsb_stereo_events.py``` Computes the stereo parameters for the coincident runs.





