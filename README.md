# magic-cta-pipe

Repository for the analysis of MAGIC and MAGIC+LST1 data, based on [*ctapipe*](https://github.com/cta-observatory/ctapipe).

* Code: https://github.com/cta-observatory/magic-cta-pipe

v0.3.1 of *magic-cta-pipe* provides all the functionalities to perform a MAGIC+LST-1 or a MAGIC-only analysis. Both types of analyses can be performed using the scripts within the *lst1_magic* folder.
See the [README](https://github.com/cta-observatory/magic-cta-pipe/blob/master/magicctapipe/scripts/lst1_magic/README.md) for more details on how to run the analysis.

**NOTE**

v0.3.1 of *magic-cta-pipe* will be the last one before cleanup of old files. Also the last one supporting ctapipe v0.12 and most probably having backward incompatible changes!

# Installation for users

*magic-cta-pipe* and its dependencies may be installed using the *Anaconda* or *Miniconda* package system. We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your master environment (this is optional).

The following command will set up a conda virtual environment, add the necessary package channels, and install *magic-cta-pipe* and its dependencies::

    git clone https://github.com/cta-observatory/magic-cta-pipe.git
    cd magic-cta-pipe
    conda env create -n magic-lst1 -f environment.yml
    conda activate magic-lst1
    pip install .

# Instructions for developers

People who would like to join the development of *magic-cta-pipe*, please contact Alessio Berti (<alessioberti90@gmail.com>) to get write access to the repository.

Developers should follow the coding style guidelines of the *ctapipe* project, see https://ctapipe.readthedocs.io/en/stable/developer-guide/style-guide.html and https://ctapipe.readthedocs.io/en/stable/developer-guide/code-guidelines.html.

In short, to check for code/style errors and for reformatting the code:

```
pip install hacking     # installs all checker tools
pip install black       # installs black formatter
pyflakes magicctapipe   # checks for code errors
flake8 magicctapipe     # checks style and code errors
black filename.py       # reformats filename.py with black
```

In general, if you want to add a new feature or fix a bug, please open a new issue, and then create a new branch to develop the new feature or code the bug fix. You can create an early pull request even if it is not complete yet, you can tag it as "Draft" so that it will not be merged, and other developers can already check it and provide comments. When the code is ready, remove the tag "Draft" and select two people to review the pull request (at the moment the merge is not blocked if no review is performed, but that may change in the future). When the review is complete, the branch will be merged into the main branch.

<!--
A brief description:
1. `config/CrabNebula.yaml`: an example of the configuration file, used by all the scripts.
2. `config/magic-cta-pipe_config_stereo.yaml`: an example of the configuration file for stereo analysis.
3. `hillas_preprocessing.py`: compute the hillas parameters. Loops over MCs and real data. This script uses the tailcuts cleaning.
4. `hillas_preprocessing_stereo.py`: compute the hillas and stereo parameters. Loops over MCs and real data. This script uses the tailcuts cleaning.
5. `hillas_preprocessing_MAGICCleaning.py`: compute the hillas parameters. Loops over MCs and real data. This script used the MAGIC cleaning implemented in MARS.
6. `hillas_preprocessing_MAGICCleaning_stereo.py`: compute the hillas and stereo parameters. Loops over MCs and real data. This script used the MAGIC cleaning implemented in MARS.
7. `train_energy_rf.py`: trains the energy RF.
8. `train_direction_rf.py`: trains the direction "disp" RF.
9. `train_classifier_rf.py`: trains the event classification RF.
10. `apply_rfs.py`: applies the trained RFs to the "test" event sample.
11. `add_orig_mc_tree.py`: adds the "original MC" tree info to the MC events tree processed earlier.
12. `make_irf.py`: generates IRFs based on the event lists with reconstructed parameters.
13. `make_event_lists.py`: produces the FITS event lists with application of the cuts.

Moreover, the `utils` directory contains two modules:
* `MAGIC_Badpixels.py`: finds the so called bad/hot pixels i.e. pixels affected by stars, or pixels turned off or dead.
* `MAGIC_Cleaning.py`: implements the MAGIC cleaning as defined in MARS.

There is also an IPython notebook, `magic_lst_event_coincidence.ipynb`, which shows how to perform the coincidence of events between MAGIC and LST1 data, when data are taken by both systems.

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
* `irf`
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
* `reco_output`: it specifies the name of the output file after applying the Random Forests to the data. **NB:** this key must be set only for the `test_sample` data, either simulated or real.

The `image_cleaning` key is used to specify the cleaning parameters. In particular, since for both MAGIC telescopes the cleaning settings
are the same, only one key called `magic` is used. As for `data_files`, when in the future LST1 will be added in the analysis, an additional
key should be added to specify the cleaning settings for that telescope.

The `energy_rf`, `direction_rf` and `classifier_rf` keys specify the settings used for the each type of Random Forest used in the analysis.
Each of these keys have other sub-keys:

* `save_name` is the name of the output file for the specific Random Forest
* `cuts` is a string to be applied on the input data to the Random Forests
* `settings` is a set of keys specifying the settings for each Random Forest e.g. the number of estimators, the minimum number of events in each leaf and the number of jobs
* `features` is a list of strings specifying the parameters to be used in the Random Forests training. **NB:** for the `direction_rf` key, `features` is actually a dictionary with two keys, `disp` and `pos_angle_shift`. For each of those keys, a list is used to specify the parameters to be used for each of those Random Forests.

The `irf` key has only one sub-key, called `output_name`, which is the name (plus path) of the file where IRF will be stored in FITS format.

Finally, the `event_list` key is used to specify some cuts, `quality` or user `selection` cuts.

### Configuration file magic-cta-pipe\_config\_stereo.yaml ###

This configuration file is very similar to the previous one, but it should be used when stereo analysis has to be performed. In particular, what changes wrt
`CrabNebula.yaml` is that there is only one telescope name key, namely `magic`. This is because the input mask in this case will specify data from both
M1 and M2 to allow for stereo reconstruction.

### hillas\_preprocessing.py ###

The first script to run the pipeline is `hillas_preprocessing.py`. It takes calibrated files (both simulated and real data) as input and processes them:

* it performs the image cleaning
* it calculates the Hillas parameters (using the `ctapipe.image.hillas_parameters` and `ctapipe.image.leakage` functions)
* it computes the timing parameters (using the `ctapipe.image.timing_parameters.timing_parameters` function)

The settings of the cleaning, as well as the input and output files of the script, are specified in the configuration file. The format of the output files
is HDF5.

For MAGIC data, its reading is performed through the [`ctapipe_io_magic`](https://gitlab.mpcdf.mpg.de/ievo/ctapipe_io_magic) module. It defines the class
`MAGICEventSource`, which inherits from the [`EventSource`](https://cta-observatory.github.io/ctapipe/api/ctapipe.io.EventSource.html) class defined in `ctapipe`,
used to setup classes to read different sources of data.

Running the script is straightforward:

```bash
$ python hillas_preprocessing.py --config=config.yaml
```

where `config.yaml` is the name of the configuration file.

Other available options are:
* `--usereal`: run the script only over real data
* `--usemc`: run the script only over MC data
* `--usetest`: run the script only over test sample data
* `--usetrain`: run the script only over train sample data
* `--usem1`: run the script only over M1 data
* `--usem2`: run the script only over M2 data

These options can be concatenated, e.g.:

```bash
$ python hillas_preprocessing.py --config=config.yaml --usereal --usetest --usem1
```

will run the script over real data from the test sample and from the M1 telescope only.

The next step in the pipeline is training the Random Forests for event classification, energy and direction reconstruction.

### hillas\_preprocessing\_MAGICCleaning.py ###

It is similar to `hillas_preprocessing.py`, the only difference is that it uses the MAGIC cleaning implemented in MARS. Its usage is the same as `hillas_preprocessing.py`, see above.

### hillas\_preprocessing\_stereo.py and hillas\_preprocessing\_MAGICCleaning\_stereo.py ###

These script are very similar to `hillas_preprocessing.py` and `hillas_preprocessing_MAGICCleaning.py`, but they include also the reconstruction of stereo parameters.

Running the scripts is straightforward, e.g.:

```bash
$ python hillas_preprocessing_stereo.py --config=config_stereo.yaml
```

where `config_stereo.yaml` is the name of the configuration file, the proper one for stereo analysis.

Other available options are:
* `--usereal`: run the script only over real data
* `--usemc`: run the script only over MC data
* `--usetest`: run the script only over test sample data
* `--usetrain`: run the script only over train sample data

### train\_energy\_rf.py, train\_direction\_rf.py, train\_classifier\_rf.py ###

These scripts take care of training different Random Forests with different purposes:

* `train_energy_rf.py` trains the Random Forest for the energy reconstruction
* `train_direction_rf.py` trains the Random Forest for the event direction reconstruction
* `train_classifier_rf.py` trains the Random Forest for the event classification

`train_energy_rf.py` and `train_direction_rf.py` run on simulated data from both the train and test sample. `train_classifier_rf.py`
instead runs on the test sample of simulated data and on OFF data.

Each scripts saves some performance summary plots as PNG images:

* `train_energy_rf.py` saves the energy migration matrix and the energy bias and RMS
* `train_direction_rf.py` saves the histogram of theta2 and the PSF as a function of the energy and offset distance
* `train_classifier_rf.py` saves the event classification histograms

To run these scripts, taking as example `train_energy_rf.py`, just do:

```bash
$ python train_energy_rf.py --config=config.yaml
```

If you want to run these three scripts over DL1 data containing the stereo information, i.e. generated by `hillas_preprocessing_stereo.py` or `hillas_preprocessing_MAGICCleaning_stereo.py`, you need to add the `--stereo` option when calling them from the command line.

Once the Random Forests are trained, they can be applied to the data. Before this step, another one must be performed using the script
`add_orig_mc_tree.py`, described in the following paragraph.

### add\_orig\_mc\_tree.py ###

The script `add_orig_mc_tree.py` opens the calibrated simulated files (for both train and test samples) to read the `OriginalMC` tree,
containing the information about the simulated values for each event (e.g. energy, arrival direction of the events).
The information is then copied to the output files created by `hillas_preprocessing.py`.

Run this script with the command:

```bash
$ python add_orig_mc_tree.py --config=config.yaml
```

Other available options are:
* `--usetest`: run the script only over test sample data
* `--usetrain`: run the script only over train sample data
* `--usem1`: run the script only over M1 data
* `--usem2`: run the script only over M2 data
* `--stereo`: run over DL1 data containing stereo information (i.e. generated by `hillas_preprocessing_stereo.py` or `hillas_preprocessing_MAGICCleaning_stereo.py`)

After this step, the Random Forests can be applied to the ON data and simulated data (test sample).

### apply\_rfs.py ###

The script `apply_rfs.py` is responsible for applying the trained Random Forests (energy, event direction and classification) to the ON
and the test sample of simulated data, reconstructing the properties of the events. The result of the reconstruction is saved in a HDF5
output file, one for the ON and one for the simulated data, as specified by the `reco_output` keys of the configuration file.

To run the script, just do:

```bash
$ python apply_rfs.py --config=config.yaml
```

If you want to run the script over DL1 data containing the stereo information, i.e. generated by `hillas_preprocessing_stereo.py` or `hillas_preprocessing_MAGICCleaning_stereo.py`, you need to add the `--stereo` option when calling them from the command line.

### make\_irf.py ###

The script `make_irf.py` generates the instrument response functions (IRFs) starting from the test sample of simulated data, after the Random
Forests have been applied to them. The result is a FITS file containing the following tables (the names are self-explanatory):

* `POINT SPREAD FUNCTION`
* `ENERGY DISPERSION`
* `EFFECTIVE AREA`

For the time being, the name of the reconstructed test sample simulated data file and of the output FITS file is hardcoded in the script, but
it will be changed in the future so that they can be set with the YAML configuration file. In any case, the script needs the configuration file
to be passed as command line argument:

```bash
$ python make_irf.py --config=config.yaml
```

If you run the script `apply_rfs.py` with the `--stereo` option, then also `make_irf.py` should be called with the `--stereo` option.


### make\_event\_lists.py ###

`make_event_lists.py` is the last script of the pipeline and is responsible of creating an event list. First, a list of good time intervals (GTI)
is created (applying the cuts specified in the configuration file), then event information (ID, time, sky coordinates and reconstructed energy) are
extracted. The GTI and the event information are used to create two tables in the resulting FITS files: for each MAGIC run, a FITS file is generated.

To run this script:

```bash
$ python make_event_lists.py --config=config.yaml
```

If you used the `--stereo` option for the previous scripts, then also `make_event_lists.py` should be called with the `--stereo` option.
-->
