.. _IT_data_NSB:

IT Cluster analysis: NSB_matching
=================================

TODO:
----- 

1. Database (joint observations)

Quick start tutorial
--------------------

Update the config_h5.yaml file with the time range, target name and bad runs.

``python list_from_h5.py`` to create the lists with the runs

``python nsb_level.py`` If there is no file called "config_general" or if you want to set another name to the config file, you can use the option "-c PG1553_config_general.yaml". This is actually valid for all scripts in the NSB branch. This script computes the NSB level for each LST run found by the first script. For a single LST run, it takes around 50 min to compute the NSB level, so we launch the jobs in parallel. For each job, it creates one txt file with information about the NSB. 

``python collect_nsb.py`` This script stacks the information from the txt files created above separated by NSB level.

``python nsb_setting_up_config_and_dir.py`` Creates the directories for DL1 MAGIC data separated by observation period and processes MAGIC data up to DL1.

To merge the subruns into runs, the M1 and M2 runs, and then runs into nights, we do:

``python nsb_merge_subruns.py``

``python nsb_merge_M1_M2_runs.py``

``python nsb_merge_M1_M2_night.py``

``python nsb_coincident_events.py`` Find the MAGIC-LST coincident events and organize them by NSB level.

``python nsb_stereo_events.py`` Computes the stereo parameters for the coincident runs.





