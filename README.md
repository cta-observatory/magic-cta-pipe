# ICRR-MPP analysis pipeline for MAGIC and LST data

This repository contains the scripts needed to perform MAGIC+LST analysis with ctapipe.

### Note: This package is under heavy development. Usage at your own risk, it is recommended to contact one of the latest committers if you plan to use this package.


A brief description:
1. `CrabNebula.yaml`: an example of the configuration file, used by all the scripts.
2. `hillas_preprocessing.py`: compute the hillas parameters. Loops over MCs and real data.
3. `train_energy_rf.py`: trains the energy RF.
4. `train_direction_rf.py`: trains the direction "disp" RF.
5. `train_classifier_rf.py`: trains the event classification RF.
6. `apply_rfs.py`: applies the trained RFs to the "test" event sample.
7. `add_orig_mc_tree.py`: adds the "original MC" tree info to the MC events tree processed earlier.
8. `make_irf.py`: generates IRFs based on the event lists with reconstructed parameters.
9. `make_event_lists.py`: produces the FITS event lists with application of the cuts.
