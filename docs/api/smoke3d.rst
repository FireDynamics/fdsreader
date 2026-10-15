Smoke3D
=======

.. warning::

   **Smoke3D data are display codes, not physical values.** FDS writes Smoke3D output
   for rendering in Smokeview: every value is stored as a single byte between 0 and 254,
   and fdsreader returns exactly these codes.

   - HRRPUV is scaled to ``HRRPUV_MAX_SMV`` (1200 kW/m³ by default). Higher values are
     cut off, and values below one 8-bit step (about 4.7 kW/m³, half of it from FDS 6.11
     on, which rounds instead of truncating) become 0.
   - Soot is stored as obscuration before FDS 6.11, and from FDS 6.11 on normalised by
     the maximum of each frame, so codes are not comparable between time steps. FDS 6.10
     additionally writes the normalised density into a ``.s3dd`` file.
   - FDS averages the values onto cell corners before writing them.

   Summing or integrating Smoke3D values therefore does not give physical results, for
   example the total heat release rate (see `#115
   <https://github.com/FireDynamics/fdsreader/issues/115>`_). For quantitative analysis
   use slices (``&SLCF``), Plot3D output, or the CSV files (e.g. the HRR column of
   ``CHID_hrr.csv``).

   :meth:`Smoke3D.to_physical() <fdsreader.smoke3d.smoke3d.Smoke3D.to_physical>` and
   ``to_global(physical=True)`` convert the codes back to kg/m³, kW/m³ or °C with the
   encoding of the FDS version that wrote them, and the limits from the ``.smv`` file (FDS
   6.10 and later; the defaults otherwise). The result keeps the 8-bit resolution and the
   averaging onto cell corners; values FDS cut off at the upper limit are returned at the
   limit and marked with ``return_saturated=True``. Checked against slices of the same
   quantities with FDS 6.10.1 and 6.11.1: the median difference is about a third of an 8-bit
   step, for soot below one percent. Soot from FDS 6.10 is read from the ``.s3dd`` file
   when it exists; otherwise, and for FDS 6.9 and older, it is recovered from the
   obscuration, which resolves thin smoke coarsely (a few percent).

   The default Smoke3D temperature is the effective flame temperature, which FDS before 6.11
   labels ``TEMPERATURE``; compare it with an ``EFFECTIVE FLAME TEMPERATURE`` slice, not the
   gas temperature.

.. automodule:: fdsreader.smoke3d.smoke3D_collection
   :autosummary:
   :members:
   :undoc-members:

.. automodule:: fdsreader.smoke3d.smoke3d
   :autosummary:
   :members:
   :undoc-members:
