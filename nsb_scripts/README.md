Scripts to be used to analyse real data by matching them to the available MCs (according to NSB level)
The lstchain 'lstchain_standard_config.json' is needed (in the same directory) to run them
Based on an upgraded version of the pipeline (scripts, modules, environment), which is needed to run these scripts: see "Semi-automatic MCP and expansion towards 4 LSTs - Torino team update" PR. So, to be merged after this PR.

This is a temporary branch (to store these scripts): A new branch and a PR will be created from the master after the '4 LSTs' PR merge.

nsb_level.py to be launched (bash script) at the beginning of the analysis, to classify LST runs according to NSB level. Then, standard semi-automatic analysis (see README in the 4-LSTs PR) on real data 


# TODO: 

1. Directory structure (more consistent with the MC one)
2. Matching of the observation period
3. RF/IRF paths (config?)
4. Database (joint observations)
5. Fix pyflakes errors and run black
6. Fix comments...
7. remove print and dependencies
8. Final checks (CI...)
