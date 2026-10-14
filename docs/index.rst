fdsreader
=========

**Fast and easy-to-use Python reader for** `Fire Dynamics Simulator (FDS)
<https://pages.nist.gov/fds-smv/>`_ **output.**

fdsreader reads the binary and CSV files an FDS simulation writes and hands you the data
as NumPy arrays. Point it at a simulation directory and every output is available as a
Python object, loaded only when you first access it. Data from multi-mesh simulations can
be merged into one global array. For a quick look without writing any plotting code,
there is an :doc:`explorer <guide/explorer>` for Jupyter and the terminal.

Source code, issue tracker and releases are on
`GitHub <https://github.com/FireDynamics/fdsreader>`_.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   guide/installation
   guide/quickstart
   guide/configuration
   guide/explorer
   guide/examples
   guide/compatibility

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/simulation
   api/slcf
   api/bndf
   api/geom
   api/isof
   api/pl3d
   api/smoke3d
   api/part
   api/devc
   api/evac
   api/quantity
   api/fds_classes
   api/utils
   api/export
   api/explorer

.. toctree::
   :maxdepth: 1
   :caption: Development

   dev/contributing
   dev/releasing
