#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from setuptools import setup, find_packages

entry_points = {}
entry_points["console_scripts"] = [
    "create_dl3_index_files = magicctapipe.scripts.lst1_magic.create_dl3_index_files:main",
    "lst1_magic_create_irf = magicctapipe.scripts.lst1_magic.lst1_magic_create_irf:main",
    "lst1_magic_dl1_stereo_to_dl2 = magicctapipe.scripts.lst1_magic.lst1_magic_dl1_stereo_to_dl2:main",
    "lst1_magic_dl2_to_dl3 = magicctapipe.scripts.lst1_magic.lst1_magic_dl2_to_dl3:main",
    "lst1_magic_event_coincidence = magicctapipe.scripts.lst1_magic.lst1_magic_event_coincidence:main",
    "lst1_magic_mc_dl0_to_dl1 = magicctapipe.scripts.lst1_magic.lst1_magic_mc_dl0_to_dl1:main",
    "lst1_magic_stereo_reco = magicctapipe.scripts.lst1_magic.lst1_magic_stereo_reco:main",
    "lst1_magic_train_rfs = magicctapipe.scripts.lst1_magic.lst1_magic_train_rfs:main",
    "magic_calib_to_dl1 = magicctapipe.scripts.lst1_magic.magic_calib_to_dl1:main",
    "merge_hdf_files = magicctapipe.scripts.lst1_magic.merge_hdf_files:main",
]

tests_require = ["pytest", "pandas>=0.24.0", "importlib_resources;python_version<'3.9'"]

docs_require = [
    "sphinx",
    "sphinx-automodapi",
    "sphinx_argparse",
    "sphinx_rtd_theme",
    "numpydoc",
    "nbsphinx",
]

setup(
    use_scm_version={"write_to": os.path.join("magicctapipe", "_version.py")},
    packages=find_packages(),
    install_requires=[
        "ctapipe~=0.19.0",
        "ctaplot~=0.5.5",
        "gammapy~=0.19.0",
        "uproot",
        "pandas",
        "pyirf",
        "seaborn",
        "setuptools_scm",
        "astropy",
        "bokeh",
        "eventio>=1.9.1,<2.0.0a0",  # at least 1.1.1, but not 2
        "h5py",
        "iminuit>=2",
        "joblib~=1.2.0",
        "matplotlib~=3.5",
        "numba",
        "numpy<1.22.0a0",
        "protobuf~=3.20.0",
        "scipy",
        "scikit-learn~=1.0",
        "tables",
        "toml",
        "pymongo",
        "pyparsing",
        "jinja2",
    ],
    entry_points=entry_points,
    extras_require={
        "all": tests_require + docs_require,
        "tests": tests_require,
        "docs": docs_require,
    },
)
