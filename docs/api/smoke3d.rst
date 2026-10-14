Smoke3D
=======

.. warning::

   **Smoke3D data are display codes, not physical values.** FDS writes Smoke3D output
   for rendering in Smokeview: every value is stored as a single byte between 0 and 254,
   and fdsreader returns exactly these codes.

   - HRRPUV is scaled to ``HRRPUV_MAX_SMV`` (1200 kW/m³ by default). Higher values are
     cut off, and values below about 2.4 kW/m³ become 0.
   - Soot is stored as obscuration before FDS 6.11, and from FDS 6.11 on normalised by
     the maximum of each frame, so codes are not comparable between time steps.
   - FDS averages the values onto cell corners before writing them.

   Summing or integrating Smoke3D values therefore does not give physical results, for
   example the total heat release rate (see `#115
   <https://github.com/FireDynamics/fdsreader/issues/115>`_). For quantitative analysis
   use slices (``&SLCF``), Plot3D output, or the CSV files (e.g. the HRR column of
   ``CHID_hrr.csv``).

.. automodule:: fdsreader.smoke3d.smoke3D_collection
   :autosummary:
   :members:
   :undoc-members:

.. automodule:: fdsreader.smoke3d.smoke3d
   :autosummary:
   :members:
   :undoc-members:
