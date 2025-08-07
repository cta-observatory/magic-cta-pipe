==============
magic-cta-pipe
==============

.. container::

    |Actions Status| |PyPI Status| |Conda Status| |Documentation Status| |Pre-Commit| |isort Status| |black|

Repository for the analysis of MAGIC and MAGIC+LST1 data, based on `ctapipe <https://github.com/cta-observatory/ctapipe>`_.

* Code: https://github.com/cta-observatory/magic-cta-pipe
* Docs (preliminary): https://magic-cta-pipe.readthedocs.io/

v0.5.5 of *magic-cta-pipe* provides all the functionalities to perform a MAGIC+LST-1 or a MAGIC-only analysis. Both types of analyses can be performed using the scripts within the *lst1_magic* folder.
See `here <https://magic-cta-pipe.readthedocs.io/en/latest/user-guide/magic-lst-scripts.html>`_ for more details on how to run the analysis.

v0.5.5 is based on *ctapipe* v0.19.x and *cta-lstchain* v0.10.x (with 5<=x<12).

**NOTE ON OLD RELEASES**

v0.3.1 of *magic-cta-pipe* was the last release before the cleanup of old files. Also, it was the last one supporting ctapipe v0.12.
In order to exploit fully the new functionalities provided by *ctapipe*, use always the latest stable release of *magic-cta-pipe* (currently v0.5.5).
v0.4.0 contained backward incompatible changes with respect to v0.3.1. Therefore, you cannot mix analyses performed with the two releases (and more recent ones).

**COMPATIBILITY OF MAGIC-CTA-PIPE WITH LSTCHAIN DATA**

At the moment of the release v0.4.0 of *magic-cta-pipe*, some LST-1 data were processed with *cta-lstchain* v0.9.x,
while the most recent ones are processed with v0.10.x. v0.4.2 of *magic-cta-pipe* and more recent releases allow to read in LST data files
created with both v0.9.x and v0.10.x, so that you do not need to use different versions of *magic-cta-pipe* to
process LST data.

Note that there are quite a lot of differences between v0.3.1 and v0.4.x, like for the telescope combinations definition,
the way IRF are created (due to different *pyirf* versions) and so on. Therefore it may not be straightforward to stack the
data at high level. We recommend to use only releases more recent than v0.4.2 for the processing, so that there will be no mismatches during the analysis.

Installation for users
----------------------

The very first step to reduce MAGIC-LST data is to have remote access/credentials to the IT Container. If you do not have it, please write an email to request it to <admin-ctan@cta-observatory.org>, and the admin will send you the instructions to connect to the IT container.

*magic-cta-pipe* and its dependencies may be installed using the *Anaconda* or *Miniconda* package system (if you have mamba installed, we recommend you to use it instead of conda, so that the installation process will be much faster; if you don't have anaconda/miniconda/miniforge, please install one of them into your workspace directory). We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your master environment (this is optional).

Since version 0.5.1, *magic-cta-pipe* is on conda-forge (https://anaconda.org/conda-forge/magic-cta-pipe), which is the easiest way to install it.

To install into an exisiting environment, just do::

    # or conda
    $ mamba install -c conda-forge magic-cta-pipe

or, to create a new environment::

    # or conda
    mamba create -c conda-forge -n mcp python=3.11 magic-cta-pipe

Alternatively, the following command will set up a conda virtual environment, add the necessary package channels, and install *magic-cta-pipe* and its dependencies::

    git clone https://github.com/cta-observatory/magic-cta-pipe.git
    cd magic-cta-pipe
    conda env create -n magic-lst -f environment.yml
    conda activate magic-lst
    pip install .

In general, *magic-cta-pipe* is still in heavy development phase, so expect large changes between different releases.

Instructions for developers
---------------------------

Developers should follow the development install instructions found in the
`documentation <https://magic-cta-pipe.readthedocs.io/en/latest/developer-guide/getting-started.html>`_.

.. |Actions Status| image:: https://github.com/cta-observatory/magic-cta-pipe/actions/workflows/ci.yml/badge.svg?branch=master
    :target: https://github.com/cta-observatory/magic-cta-pipe/actions
    :alt: magic-cta-pipe GitHub Actions CI Status

.. |PyPI Status| image:: https://badge.fury.io/py/magic-cta-pipe.svg
    :target: https://pypi.org/project/magic-cta-pipe
    :alt: magic-cta-pipe PyPI Status

.. |Conda Status| image:: https://anaconda.org/conda-forge/magic-cta-pipe/badges/version.svg
    :target: https://anaconda.org/conda-forge/magic-cta-pipe
    :alt: magic-cta-pipe Conda Status

.. |Documentation Status| image:: https://readthedocs.org/projects/magic-cta-pipe/badge/?version=latest&style=flat
    :target: https://magic-cta-pipe.readthedocs.io/en/latest/
    :alt: magic-cta-pipe documentation Status

.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. |isort Status| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/
    :alt: isort Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
