#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# import ah_bootstrap
from setuptools import setup, find_packages
import os

tests_require = ["pytest", "pandas>=0.24.0", "importlib_resources;python_version<'3.9'"]

docs_require = [
    "sphinx~=4.2",
    "sphinx-automodapi",
    "sphinx_argparse",
    "sphinx_rtd_theme",
    "numpydoc",
    "nbsphinx"
]

setup(
    use_scm_version={"write_to": os.path.join("magicctapipe", "_version.py")},
    packages=find_packages(),
    install_requires=[
        'astropy~=4.2',
        'ctapipe~=0.12.0',
        'ctapipe_io_magic~=0.3.0',
        'ctaplot~=0.5.3',
        'eventio>=1.5.1,<2.0.0a0',  # at least 1.1.1, but not 2
        'gammapy>=0.18',
        'h5py',
        'joblib',
        'matplotlib>=3.5',
        'numba',
        'numpy',
        'pandas',
        'pyirf~=0.5.0',
        'scipy',
        'seaborn',
        'scikit-learn',
        'tables',
        'toml',
        'traitlets~=5.0.5',
        'setuptools_scm',
    ],
    extras_require={
        "all": tests_require + docs_require,
        "tests": tests_require,
        "docs": docs_require,
    },
)
