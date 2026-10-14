Examples
========

The `examples directory <https://github.com/FireDynamics/fdsreader/tree/master/examples>`_
on GitHub has a script for every data type. Each folder contains the FDS input files
(``.fds``) the script was written for; run them with FDS first, then run the script in
the same folder. The scripts are tested with FDS 6.10.1 and 6.11.1.

Most scripts plot with matplotlib or `PyVista <https://pyvista.org>`_, and the
animations are written as GIF with imageio. None of these are installed with fdsreader:

.. code-block:: sh

   pip install matplotlib pyvista imageio pandas "fdsreader[export]"

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Folder
     - What it shows
   * - `slcf <https://github.com/FireDynamics/fdsreader/tree/master/examples/slcf>`_
     - Merging a slice of a multi-mesh simulation into one array with obstructions
       masked out (``slcf_example_global.py``), and masking the wall of the Steckler
       room in a single-mesh slice (``slcf_example_mask.py``).
   * - `bndf <https://github.com/FireDynamics/fdsreader/tree/master/examples/bndf>`_
     - Wall temperature on one face of an obstruction, animated over time.
   * - `geom <https://github.com/FireDynamics/fdsreader/tree/master/examples/geom>`_
     - Radiative heat flux on a sphere defined with ``&GEOM`` (``geom_example.py``), and
       the cells cut by a cone, written with ``SLICETYPE='CUT_CELLS'``
       (``geom_slices.py``).
   * - `pl3d <https://github.com/FireDynamics/fdsreader/tree/master/examples/pl3d>`_
     - Plot3D temperature as a 3D grid, rendered with PyVista.
   * - `smoke3d <https://github.com/FireDynamics/fdsreader/tree/master/examples/smoke3d>`_
     - Smoke3D data as a 3D grid, animated with PyVista. The values are display codes,
       see :doc:`../api/smoke3d`.
   * - `isof <https://github.com/FireDynamics/fdsreader/tree/master/examples/isof>`_
     - Isosurfaces as triangle meshes, rendered with PyVista.
   * - `part <https://github.com/FireDynamics/fdsreader/tree/master/examples/part>`_
     - Water droplets falling over a stack of boxes, coloured by droplet diameter and
       written as an animation.
   * - `devc <https://github.com/FireDynamics/fdsreader/tree/master/examples/devc>`_
     - A device time series, a line of thermocouples as a temperature profile over the
       height, and all devices as a pandas DataFrame.
   * - `export <https://github.com/FireDynamics/fdsreader/tree/master/examples/export>`_
     - Writing slices, obstructions and Smoke3D data of the Steckler room to raw binary
       files with YAML metadata.
