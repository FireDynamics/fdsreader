Boundary data
=============

Boundary files (``&BNDF``) hold a quantity on solid surfaces. Data on the faces of an obstruction belong to the
:class:`~fdsreader.bndf.obstruction.Obstruction`, see ``Obstruction.get_global_boundary_data_arrays``. Data on the
walls of the simulation domain that are not part of an obstruction, e.g. the ground of an outdoor case, belong to
the meshes: ``Mesh.get_boundary_data`` for one mesh, and ``sim.meshes.get_global_boundary_data_arrays`` for all
meshes combined.

.. automodule:: fdsreader.bndf.obstruction_collection
   :autosummary:
   :members:
   :undoc-members:

.. automodule:: fdsreader.bndf.obstruction
   :autosummary:
   :members:
   :undoc-members:

.. automodule:: fdsreader.bndf.utils
   :autosummary:
   :members:
   :undoc-members:
