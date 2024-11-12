.. _magicctapipe:

============================================
Welcome to ``magic-cta-pipe`` documentation!
============================================

.. currentmodule:: magicctapipe

**Version**: |version| **Date**: |today|

**Useful links**:
`Source Repository <https://github.com/cta-observatory/magic-cta-pipe>`__ |
`Issue Tracker <https://github.com/cta-observatory/magic-cta-pipe/issues>`__ |
`Discussions <https://github.com/cta-observatory/magic-cta-pipe/discussions>`__

**License**: BSD-3

**Python**: |python_requires|

**magic-cta-pipe** is a pipeline for the analysis of joint data taken with MAGIC and LST-1.

Check out the :doc:`user-guide/index` section for the analysis steps, including how to :ref:`installation` the project.

.. note::

   This project is under active development.

.. note::

   At the moment of the release v0.5.0 of *magic-cta-pipe*, some LST-1 data are processed with *cta-lstchain* v0.9.x,
   while the most recent ones are processed with v0.10.x. v0.5.0 of *magic-cta-pipe* allows to read in LST data files
   created with both v0.9.x and v0.10.x, so that you do not need to use different versions of *magic-cta-pipe* to
   process LST data.

   Note that there are quite a lot of differences between v0.3.1 and v0.4.x, like for the telescope combinations definition,
   the way IRF are created (due to different *pyirf* versions) and so on. Therefore it may not be straightforward to stack the
   data at high level. We recommend to use only v0.5.0 for the processing, so that there will be no mismatches during the analysis.

Contents
--------

.. _magicctapipe_docs:

.. toctree::
  :maxdepth: 1
  :hidden:

  user-guide/index
  developer-guide/index
  api-reference/index

.. grid:: 1 2 2 3

   .. grid-item-card::

      :octicon:`book;40px`

      User Guide
      ^^^^^^^^^^

      Learn how to get started as a user. This guide
      will help you install and using magic-cta-pipe.

      +++

      .. button-ref:: user-guide/index
         :expand:
         :color: primary
         :click-parent:

         To the user guide

   .. grid-item-card::

      :octicon:`git-branch;40px`

      Developer Guide
      ^^^^^^^^^^^^^^^

      Learn how to get started as a developer.
      This guide will help you install magic-cta-pipe for development
      and explains how to contribute.

      +++

      .. button-ref:: developer-guide/index
         :expand:
         :color: primary
         :click-parent:

         To the developer guide

   .. grid-item-card::

        :octicon:`code;40px`

        API Docs
        ^^^^^^^^

        The API docs contain detailed descriptions of
        of the various modules, classes and functions
        included in ctapipe.

        +++

        .. button-ref:: api-reference/index
            :expand:
            :color: primary
            :click-parent:

            To API docs
