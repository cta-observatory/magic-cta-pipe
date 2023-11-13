#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from setuptools import find_packages, setup

entry_points = {}
entry_points["console_scripts"] = [
    "coincident_events = magicctapipe.scripts.lst1_magic.semi_automatic_scripts.coincident_events:main",
    "create_dl3_index_files = magicctapipe.scripts.lst1_magic.create_dl3_index_files:main",
    "lst1_magic_create_irf = magicctapipe.scripts.lst1_magic.lst1_magic_create_irf:main",
    "lst1_magic_dl1_stereo_to_dl2 = magicctapipe.scripts.lst1_magic.lst1_magic_dl1_stereo_to_dl2:main",
    "lst1_magic_dl2_to_dl3 = magicctapipe.scripts.lst1_magic.lst1_magic_dl2_to_dl3:main",
    "lst1_magic_event_coincidence = magicctapipe.scripts.lst1_magic.lst1_magic_event_coincidence:main",
    "lst1_magic_mc_dl0_to_dl1 = magicctapipe.scripts.lst1_magic.lst1_magic_mc_dl0_to_dl1:main",
    "lst1_magic_stereo_reco = magicctapipe.scripts.lst1_magic.lst1_magic_stereo_reco:main",
    "lst1_magic_train_rfs = magicctapipe.scripts.lst1_magic.lst1_magic_train_rfs:main",
    "LSTnsb = magicctapipe.scripts.lst1_magic.semi_automatic_scripts.LSTnsb:main",
    "magic_calib_to_dl1 = magicctapipe.scripts.lst1_magic.magic_calib_to_dl1:main",
    "merge_hdf_files = magicctapipe.scripts.lst1_magic.merge_hdf_files:main",
    "merging_runs_and_splitting_training_samples = magicctapipe.scripts.lst1_magic.semi_automatic_scripts.merging_runs_and_splitting_training_samples:main",
    "nsb_level = magicctapipe.scripts.lst1_magic.semi_automatic_scripts.nsb_level:main",
    "setting_up_config_and_dir = magicctapipe.scripts.lst1_magic.semi_automatic_scripts.setting_up_config_and_dir:main",
    "stereo_events = magicctapipe.scripts.lst1_magic.semi_automatic_scripts.stereo_events:main",
]

tests_require = ["pytest", "pandas>=0.24.0", "importlib_resources;python_version<'3.9'"]

docs_require = [
    "sphinx",
    "sphinx-automodapi",
    "sphinx_argparse",
    "sphinx_rtd_theme",
    "pydata_sphinx_theme",
    "numpydoc",
    "nbsphinx",
    "sphinx-design",
]

setup(
    use_scm_version={"write_to": os.path.join("magicctapipe", "_version.py")},
    packages=find_packages(),
    install_requires=[
        "lstchain~=0.9.6",
        "ctapipe~=0.12.0",
        "ctapipe_io_magic~=0.4.7",
        "ctaplot~=0.5.5",
        "gammapy~=0.19.0",
        "uproot~=4.1",
        "numba<=0.56",
        "numpy<1.22.0a0",
        "joblib",
        "pandas",
        "pyirf~=0.6.0",
        "seaborn",
        "scikit-learn",
        "setuptools_scm",
    ],
    entry_points=entry_points,
    extras_require={
        "all": tests_require + docs_require,
        "tests": tests_require,
        "docs": docs_require,
    },
)
