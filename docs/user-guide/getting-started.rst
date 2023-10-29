.. _getting_started_users:

Getting started for users
=========================

.. _installation:

Installation
------------

*magic-cta-pipe* and its dependencies may be installed using the *Anaconda* or *Miniconda* package system. We recommend creating a conda virtual environment
first, to isolate the installed version and dependencies from your master environment (this is optional).

The following command will set up a conda virtual environment, add the necessary package channels, and install *magic-cta-pipe* and its dependencies:

.. code-block:: console

    git clone https://github.com/cta-observatory/magic-cta-pipe.git
    cd magic-cta-pipe
    conda env create -n magic-lst1 -f environment.yml
    conda activate magic-lst1
    pip install .
