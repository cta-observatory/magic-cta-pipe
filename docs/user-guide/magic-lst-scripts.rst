.. _magic_lst_scripts:

MAGIC+LST scripts
=================

The scripts in the ``magicctapipe/scripts`` directory can be used to perform a MAGIC+LST analysis or a MAGIC-only analysis, as explained in the following.

.. _magic_only_analysis:

MAGIC-only analysis
-------------------

MAGIC-only analysis starts from MAGIC-calibrated data (``_Y_`` files). The analysis flow is as follows:

- ``magic_calib_to_dl1.py`` on real data, to convert them into DL1 format. If you use MC data produced with MMCS, you need to run this script over MC data as well. If you use SimTelArray MCs, run ``lst1_magic_mc_dl0_to_dl1.py`` over them instead.
- optionally, but recommended, ``merge_hdf_files.py`` to merge subruns and/or runs together
- ``lst1_magic_stereo_reco.py`` to add stereo parameters to the DL1 data (use ``--magic-only`` argument if the MC DL1 data contains LST-1 events)
- ``lst1_magic_train_rfs.py`` to train the RFs (energy, direction, classification) on train gamma MCs and protons
- ``lst1_magic_dl1_stereo_to_dl2.py`` to apply the RFs to stereo DL1 data (real and test MCs) and produce DL2 data
- ``lst1_magic_create_irf.py`` to create the IRF (use ``magic_stereo`` as ``irf_type`` in the configuration file)
- ``lst1_magic_dl2_to_dl3.py`` to create DL3 files, and ``create_dl3_index_files.py`` to create DL3 HDU and index files

.. _magic_lst_analysis:

MAGIC-LST analysis
-------------------

MAGIC+LST analysis starts from MAGIC calibrated data (``_Y_`` files), LST DL1 data and SimTelArray DL0 data. The analysis flow is as following:

- ``magic_calib_to_dl1.py`` on real MAGIC data, to convert them into DL1 format
- ``lst1_magic_mc_dl0_to_dl1.py`` over SimTelArray MCs to convert them into DL1 format
- optionally, but recommended, ``merge_hdf_files.py`` on MAGIC data to merge subruns and/or runs together
- ``lst1_magic_event_coincidence.py`` to find coincident events between MAGIC and LST-1, starting from DL1 data
- ``lst1_magic_stereo_reco.py`` to add stereo parameters to the DL1 data
- ``lst1_magic_train_rfs.py`` to train the RFs (energy, direction, classification) on train gamma MCs and protons
- ``lst1_magic_dl1_stereo_to_dl2.py`` to apply the RFs to stereo DL1 data (real and test MCs) and produce DL2 data
- ``lst1_magic_create_irf.py`` to create the IRF
- ``lst1_magic_dl2_to_dl3.py`` to create DL3 files, and ``create_dl3_index_files.py`` to create DL3 HDU and index files

.. _high_level:

High-level analysis
-------------------

The folder ``notebooks`` contains Jupyter notebooks to perform checks on the IRF, to produce theta2 plots and SEDs. Note that the notebooks run with gammapy v0.20 or higher, therefore another conda environment is needed to run them, since the MAGIC+LST-1 pipeline at the moment depends on gammapy v0.19.
