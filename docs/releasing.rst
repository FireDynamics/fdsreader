Releasing a New Version
=======================

This page explains how to publish a new version of fdsreader to PyPI.
The entire process is automated — no manual upload or token handling is required.

Overview
--------

Versioning is driven entirely by **Git tags**.
When you push a tag that starts with ``v``, GitHub Actions automatically:

1. Builds the Python package (wheel + sdist)
2. Publishes it to PyPI (stable release) or TestPyPI (beta release)
3. Creates a GitHub Release with auto-generated release notes

Prerequisites (one-time setup)
-------------------------------

Before the first release, a maintainer must configure **Trusted Publishing** on PyPI:

* Log in to `pypi.org <https://pypi.org>`_ with the project account
* Go to **Account settings → Publishing → Add a new publisher**
* Fill in:

  * PyPI project name: ``fdsreader``
  * GitHub owner: ``FireDynamics``
  * Repository: ``fdsreader``
  * Workflow name: ``release.yml``
  * Environment name: ``pypi``

* Repeat for `test.pypi.org <https://test.pypi.org>`_ with environment name ``testpypi``

This setup only needs to be done once.
See :ref:`Trusted Publishing docs <https://docs.pypi.org/trusted-publishers/>` for details.

Stable release (publishes to PyPI)
-----------------------------------

.. code-block:: bash

   # 1. Make sure master is up to date and all tests are green
   git checkout master
   git pull

   # 2. Tag the release (use semantic versioning: MAJOR.MINOR.PATCH)
   git tag -a v1.12.0 -m "Version 1.12.0"

   # 3. Push the tag — this triggers the release workflow
   git push origin v1.12.0

That's it. GitHub Actions takes over from here.
The new version appears on `PyPI <https://pypi.org/project/fdsreader/>`_
within a few minutes.

Beta / pre-release (publishes to TestPyPI)
-------------------------------------------

If you want to test the package on TestPyPI before a stable release:

.. code-block:: bash

   git tag -a v1.12.0b1 -m "Version 1.12.0 beta 1"
   git push origin v1.12.0b1

Tags containing ``a``, ``b``, or ``rc`` (e.g. ``v1.12.0b1``, ``v1.12.0rc1``)
are automatically routed to TestPyPI instead of PyPI.

Checking the release
--------------------

After pushing the tag:

1. Go to **GitHub → Actions → Release** to watch the build progress
2. Check the **GitHub Releases** page for the auto-generated release notes
3. Verify the new version on `pypi.org/project/fdsreader <https://pypi.org/project/fdsreader/>`_

Updating the FDS compatibility table
--------------------------------------

After testing with a new FDS version, update the compatibility table in ``README.md``
and commit the change to master before tagging the release.

Version number guidelines
--------------------------

We follow `Semantic Versioning <https://semver.org/>`_:

* **PATCH** (1.x.y → 1.x.(y+1)): Bug fixes, no breaking changes
* **MINOR** (1.x.y → 1.(x+1).0): New features, backwards compatible
* **MAJOR** (1.x.y → 2.0.0): Breaking changes to the public API
