Maintainer info
===============

This is a collection of some notes for maintainers.

How to update the online docs?
------------------------------

The docs are automatically built and deployed using ``readthedocs``.


How to make a release?
----------------------
1. Open a new pull request to prepare the release.
   This should be the last pull request to be merged before making the actual release.

   Add the planned new version to the ``docs/_static/switcher.json`` file, so it will be
   available from the version dropdown once the documentation is built.

2. After merging the pull request above, you should add a tag called ``v*`` (e.g. ``v0.3.4``)
   in the ``master`` branch. In that way, the PyPI upload of the new release will be done
   automatically by Github Actions.

3. Create a new github release in the repository (https://github.com/cta-observatory/magic-cta-pipe/releases),
   a good starting point should already be made by the release drafter plugin.

.. 4. conda packages are built by conda-forge, the recipe is maintained here: https://github.com/conda-forge/ctapipe-feedstock/
   A pull request to update the recipe should be opened automatically by a conda-forge bot when a new version is published to PyPi. This can take a couple of hours.