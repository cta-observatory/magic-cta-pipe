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
- `lst1_magig_create_irf.py` to create the IRF (use `magic_stereo` as `irf_type` in the configuration file)
- `lst1_magic_dl2_to_dl3.py` to create DL3 files, and `create_dl3_index_files.py` to create DL3 HDU and index files

## MAGIC+LST-1 analysis

MAGIC+LST-1 analysis starts from MAGIC calibrated data (\_Y\_ files), LST-1 DL1 data and SimTelArray DL0 data. The analysis flow is as following:

- `magic_calib_to_dl1.py` on real MAGIC data, to convert them into DL1 format
- `lst1_magic_mc_dl0_to_dl1.py` over SimTelArray MCs to convert them into DL1 format
- optionally, but recommended, `merge_hdf_files.py` to merge subruns and/or runs together
- `lst1_magic_event_coincidence.py` to find coincident events between MAGIC and LST-1, starting from DL1 data
- `lst1_magic_stereo_reco.py` to add stereo parameters to the DL1 data
- `lst1_magic_train_rfs.py` to train the RFs (energy, direction, classification) on train gamma MCs and protons
- `lst1_magic_dl1_stereo_to_dl2.py` to apply the RFs to stereo DL1 data (real and test MCs) and produce DL2 data
- `lst1_magig_create_irf.py` to create the IRF
- `lst1_magic_dl2_to_dl3.py` to create DL3 files, and `create_dl3_index_files.py` to create DL3 HDU and index files

## High level analysis

The folder [Notebooks](https://github.com/cta-observatory/magic-cta-pipe/tree/master/notebooks) contains Jupyter notebooks to perform checks on the IRF, to produce theta2 plots and SEDs.
