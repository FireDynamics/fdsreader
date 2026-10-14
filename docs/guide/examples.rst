Examples
========

The examples are Jupyter notebooks that walk through each data type step by step. They are
shown here with their output. To run them yourself, get the
`examples directory <https://github.com/FireDynamics/fdsreader/tree/master/examples>`_,
run FDS on the input files (``.fds``) next to each notebook, then open the notebook in
Jupyter. They need a few packages that are not installed with fdsreader:

.. code-block:: sh

   pip install "fdsreader[notebook,export]" jupyterlab pyvista pandas

.. toctree::
   :maxdepth: 1

   ../examples/slices
   ../examples/boundary
   ../examples/devices
   ../examples/complex_geometry
   ../examples/isosurfaces
   ../examples/particles
   ../examples/plot3d
   ../examples/smoke3d
   ../examples/export

``examples/evac`` still holds a script
for the evacuation output of FDS+Evac, which current FDS versions no longer include.
