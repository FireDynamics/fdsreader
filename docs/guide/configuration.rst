Configuration
=============

A few options can be set through the ``settings`` module, before the simulation is loaded:

.. code-block:: python

   import fdsreader as fds

   fds.settings.DEBUG = True
   sim = fds.Simulation("./my_simulation")

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Setting
     - Default
     - Description
   * - ``LAZY_LOAD``
     - ``True``
     - Load data only when it is first accessed (``True``) or everything when the
       simulation is loaded (``False``).
   * - ``ENABLE_CACHING``
     - ``True``
     - Cache the parsed simulation in a ``.pickle`` file next to the ``.smv`` file, so
       loading the same simulation again is faster. Needs write access to the
       simulation directory.
   * - ``DEBUG``
     - ``False``
     - Raise an exception on non-critical errors (``True``) instead of only warning
       about them (``False``).
   * - ``IGNORE_ERRORS``
     - ``False``
     - Ignore non-critical errors completely, without a warning.

.. note::

   The cache is rebuilt automatically when the ``.smv`` file or the fdsreader version
   changes, but not when settings change. If you load the same simulation again with a
   different ``LAZY_LOAD`` or ``IGNORE_ERRORS`` value, delete the ``.pickle`` file first
   or set ``ENABLE_CACHING = False``.
