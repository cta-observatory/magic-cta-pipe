# ICRR-MPP analysis pipeline for MAGIC and LST data

This repository contains the scripts needed to perform MAGIC+LST analysis with ctapipe.
*It's still under development.*

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

Here below you can find a more detailed description of the pipeline work flow.

### Configuration file CrabNebula.yaml ###

This is an example of the configuration file which is used by all the scripts of the pipeline.
It is in [YAML](https://yaml.org/) standard, which can be easily parsed and also easily readable by humans.
Through this file, the user can configure the details of the analysis like input files, output files, details
of the cleaning and Random Forest generation and analysis cuts to be applied to the events.

More in detail, the configuration file is a series of main keys, each having other nested (key, value) pairs.
The main keys are:

* `data_files`
* `image_cleaning`
* `energy_rf`
* `direction_rf`
* `classifier_rf`
* `event_list`

`data_files` specifies the input and output files, both for simulated (MonteCarlo) and real data, denoted by the `mc`
and `data` keys. Each set of data is has a `train_sample` and `test_sample` keys. For simulated data, the `train_sample`
key refers to the simulated data sample used for the training of the Random Forest classifier, whereas the `test_sample`
is the sample used to compute the Instrument Response Functions (IRFs). For real data, the `train_sample` is what usually
is called OFF data, which are used together with simulated data in the Random Forest algorithm, while the `test_sample` refers
to the so called ON data, that is the data the user wants to analyze.
Each `train_sample` and `test_sample` keys have two sub-keys, called `magic1` and `magic2`. As their name implies, the input and
output files are specified for each telescope independently, since the pipeline starts its processing from MAGIC calibrated data.
If the analysis uses data from a third telescope, as LST1, an additional key called, for example, `lst1` can be added to specify the
input and output files. For the moment though, the pipeline works with MAGIC data only.
Each telescope key is used to specify the input and output files at different stages of the pipeline:

* `input_mask`: it specifies the input files to the pipeline; absolute and relative paths can be used; wildcards are allowed;
* `hillas_output`: it specifies the name of the output file of the script `hillas_preprocessing.py`;
* `reco_output`: it specifies the name of the output file after applying the Random Forests to the data. **NB:** this key must be set
only for the `test_sample` data, either simulated or real.

The `image_cleaning` key is used to specify the cleaning parameters. In particular, since for both MAGIC telescopes the cleaning settings
are the same, only one key called `magic` is used. As for `data_files`, when in the future LST1 will be added in the analysis, an additional
key should be added to specify the cleaning settings for that telescope.

The `energy_rf`, `direction_rf` and `classifier_rf` keys specify the settings used for the each type of Random Forest used in the analysis.
Each of these keys have other sub-keys:

* `save_name` is the name of the output file for the specific Random Forest
* `cuts` is a string to be applied on the input data to the Random Forests
* `settings` is a set of keys specifying the settings for each Random Forest e.g. the number of estimators, the minimum number of events in each
leaf and the number of jobs
* `features` is a list of strings specifying the parameters to be used in the Random Forests training. **NB:** for the `direction_rf` key, `features`
is actually a dictionary with two keys, `disp` and `pos_angle_shift`. For each of those keys, a list is used to specify the parameters to be used for
each of those Random Forests.

Finally, the `event_list` key is used to specify some cuts, `quality` or user `selection` cuts.


