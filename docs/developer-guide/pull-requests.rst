.. _pull-requests:

Making and Accepting Pull Requests
==================================

Making a Pull Request
---------------------

In the pull request description (editable on GitHub), you should
include the following information (some may be omitted if it is e.g. a
small bug fix and not a new feature or design change):

* **What** does the change do?  (description of the new features, changes,
  or refactorings)
* **Where** should the reviewer start the review? (what is the central
  module that changed, etc.)
* What is the **use case**, if it is a new feature?
* give an **example** of use or screenshot/plot if applicable
* Are any **new dependencies** required? (dependencies should be kept to a
  minimum, so all new dependences need to be accepted by management)
* is there a relevant **issue** open that this addresses? (use the
  #ISSUENUMBER syntax to link it)


Note that you can include syntax-highlighted code examples by using 3 back-tics:

.. code-block:: none

   ```python

   code here

   ```
.. warning::
    Note that the continous integration system will run only if you open a pull request
    from a branch in the main repository. Pull requests opened from a branch from a
    forked repository cannot run the continous integration system since they cannot
    access the repository secrets used in some of the workflow (e.g. to download proprietary data).

Continuous integration system
-----------------------------

The *Travis* continuos integration (CI) system runs tests for pushes on a pull request
or the ``master`` branch. At the moment the CI system runs the following workflows:

* ``lint``, which runs the ``pre-commit`` hook over the code (runs for all pull requests and push commits to the ``master`` branch)
* ``pyflakes``, which runs ``pyflakes`` over the code (runs for all pull requests and push commits to the ``master`` branch)
* ``docs``, which tests if the documentation can be built correctly (runs for all pull requests and push commits to the ``master`` branch)
* ``tests``, which runs unit tests over a set of test files (runs for all pull requests and push commit to the ``master`` branch)
* ``release-drafter``, which drafts release notes (runs only for pushes on the ``master`` branch). Note that the
  release drafter will produce different sections based on the label applied to the pull requests merged into the
  ``master`` branch. See `categories` in the `release drafter configuration file <https://github.com/cta-observatory/magic-cta-pipe/blob/master/.github/release-drafter.yml>`_
* ``deploy``, which deploys automatically a new release to PyPI whenever a tag called ``v*``
  (e.g. ``v0.3.4``) is pushed to the ``master`` branch

.. note::

    If you apply the ``documentation-only`` label on a pull request, only the ``docs`` workflow will run!
    This allows to have a faster CI run when changes are made only to the documentation part.

Keep in mind
------------

* make sure you remember to update the **documentation** as well as the code!
  (see the ``docs/`` directory), and make sure it builds with no errors
  (``make doc``)

* pull requests that cause tests to fail on *Travis* will not be accepted until those tests pass

.. * make sure to add a news fragment for the changelog.  In order to do this add a file to the directory ``docs/changes`` and use the following naming scheme
  ``<PULL REQUEST>.<TYPE>.rst`` (take a look at the ``README`` inside of the directory for more details). The file should contain a brief summary of the purpose of this pull request.


Accepting a Pull Request
------------------------

``magic-cta-pipe`` maintainers must do a *code review* before accepting any
pull request. During the review the reviewer can ask for changes to be
made, and the requester can simply push them to the branch associated
with the request and they will automatically appear (no new pull
request needed).  The following guidelines should be used to
facilitate the review procedure:

* Perform a scientific or conceptual Review if the request introduces
  new features, algorithms, or design changes

* Look at the use case for the proposed change.

  - if the use case is missing, ask for one
  - does it make sense? Is it connected to a goal, requirement, or specification?

* Perform a Code Review

  - Check that all automatic checks succeeded, if not notify the author and give
    guidance on how to fix the identified issues.
  - Check that all functions and classes have API documentation in the
    correct format.
  - Check that there are at least basic unit tests for the added functionality / fixed bug.
  - Check that the API (function and class definitions) is clear and
    easy to understand.
  - Check for common coding mistakes.
  - Check for obvious performance issues.
  - Check that the code uses the existing features of ctapipe.
  - Check that the code doesn't introduce new features that are
    already present in another form.
