FDS compatibility
=================

fdsreader reads output of FDS 6.7.5 and later. Versions below 6.7.5 may work, but are not
tested. The latest FDS version tested is 6.11.1.

.. list-table::
   :header-rows: 1

   * - fdsreader
     - FDS 6.7
     - FDS 6.8
     - FDS 6.9
     - FDS 6.10
     - FDS 6.11
   * - ≤ 1.12.0
     - ✅
     - ✅
     - ✅
     - ⚠️ complex geometry
     - ⚠️ complex geometry
   * - ≥ 1.12.1
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

⚠️ complex geometry: FDS 6.10 changed the file format for boundary data on complex
geometry (``&GEOM``). fdsreader reads it from version 1.12.1 on.

Changes in FDS itself
---------------------

Some quantity names written by FDS change between versions; fdsreader returns them as
written to the ``.smv`` file. For example, FDS 6.11 calls the Smoke3D temperature
``EFFECTIVE FLAME TEMPERATURE`` instead of ``TEMPERATURE``, so
``sim.smoke_3d.get_by_quantity("TEMPERATURE")`` raises a ``KeyError`` for FDS 6.11 output. Check
the available names with ``[s.quantity for s in sim.smoke_3d]``.

Reporting problems
------------------

The table is based on tests against FDS output. If something does not read correctly,
please `open an issue <https://github.com/FireDynamics/fdsreader/issues>`_ and mention
your FDS and fdsreader versions.
