Scripts to be used to analyse real data by matching them to the available MCs (according to NSB level)
The lstchain 'lstchain_standard_config.json' is needed (in the same directory) to run them
Based on an upgraded version of the pipeline (scripts, modules, environment), which is needed to run these scripts: see "Semi-automatic MCP and expansion towards 4 LSTs - Torino team update" PR. So, to be merged after this PR.

This is a temporary branch (to store these scripts): A new branchand a PR will be created from the master after the '4 LSTs' PR had been merged.

nsb.sh to be launched (bash script) at the beginning of the analysis, to classify LST runs according to NSB level. Then, standard semi-automatic analysis (see README in the 4 LSTs PR) on real data 


# TODO: 
1. Fix an issue with the lists of LST-MAGIC runs (in case of only one run in the .txt files, lists must be properly read): now 'rough' solution, a better solution is needed
2. Directory structure (more consistent with the MC one)
3. Matching of the observation period
4. RF/IRF paths (config?)
5. LSTnsb.py: check if string (run/night) already in the file before writing it (no duplicates)
6. Adding new data (runs) to a source folder/analysis 
7. Multi-source
8. Database (joint observations)
9. Fix pyflakes errors and run black
10. Final checks (CI...)
