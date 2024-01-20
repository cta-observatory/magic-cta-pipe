==============
magic-cta-pipe
==============

.. container::

    |Actions Status| |PyPI Status| |Documentation Status| |Pre-Commit| |isort Status| |black|

Repository for the analysis of MAGIC and MAGIC+LST1 data, based on `ctapipe <https://github.com/cta-observatory/ctapipe>`_.

* Code: https://github.com/cta-observatory/magic-cta-pipe
* Docs (preliminary): https://magic-cta-pipe.readthedocs.io/

v0.4.0 of *magic-cta-pipe* provides all the functionalities to perform a MAGIC+LST-1 or a MAGIC-only analysis. Both types of analyses can be performed using the scripts within the *lst1_magic* folder.
See `here <https://magic-cta-pipe.readthedocs.io/en/latest/user-guide/magic-lst-scripts.html>`_ for more details on how to run the analysis.

v0.4.0 is based on *ctapipe* v0.19.x and *cta-lstchain* v0.10.x.

**NOTE ON OLD RELEASES**

v0.3.1 of *magic-cta-pipe* was the last release before the cleanup of old files. Also, it was the last one supporting ctapipe v0.12.
In order to exploit fully the new functionalities provided by *ctapipe*, use always the latest stable release of *magic-cta-pipe* (currently v0.4.0).
v0.4.0 contains backward incompatible changes with respect to v0.3.1. Therefore, you cannot mix analyses performed with the two releases.

Installation for users
----------------------

*magic-cta-pipe* and its dependencies may be installed using the *Anaconda* or *Miniconda* package system. We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your master environment (this is optional).

The following command will set up a conda virtual environment, add the necessary package channels, and install *magic-cta-pipe* and its dependencies::

    git clone https://github.com/cta-observatory/magic-cta-pipe.git
    cd magic-cta-pipe
    conda env create -n magic-lst1 -f environment.yml
    conda activate magic-lst1
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

.. |Documentation Status| image:: https://img.shields.io/readthedocs/magic-cta-pipe/latest.svg?logo=read%20the%20docs&logoColor=white&label=Docs&version=stable
    :target: https://magic-cta-pipe.readthedocs.io/en/latest/?badge=stable
    :alt: magic-cta-pipe documentation Status

.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. |isort Status| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/
    :alt: isort Status

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
