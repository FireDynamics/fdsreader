Quickstart
==========

Load a simulation
-----------------

Point :class:`~fdsreader.simulation.Simulation` at the directory that contains the
``.smv`` file of your simulation. Printing it shows what the simulation wrote:

.. code-block:: python

   import fdsreader as fds

   sim = fds.Simulation("./my_simulation")
   print(sim)

.. code-block:: text

   Simulation(chid=Steckler_010,
              meshes=1,
              obstructions=4,
              slices=1,
              data_3d=5,
              smoke_3d=3,
              isosurfaces=2,
              devices=6)

Data is only read from disk when you first access it (see :doc:`configuration`).

Slices
------

Collections such as ``sim.slices`` can be filtered by quantity and searched by position.
:meth:`~fdsreader.slcf.slice.Slice.to_global` merges the data of all meshes into one
NumPy array with time as the first axis:

.. code-block:: python

   slc = sim.slices.filter_by_quantity("TEMPERATURE").get_nearest(x=0)
   temperature = slc.to_global()

   step = slc.get_nearest_timestep(30.0)
   print(slc.times[step], temperature[step].max())

Time series
-----------

Devices are looked up by the ``ID`` from the input file; the heat release rate and
other values from ``CHID_hrr.csv`` are in ``sim.hrr``:

.. code-block:: python

   time = sim.devices["Time"].data
   hgl_temp = sim.devices["HGL Temp"].data
   hrr = sim.hrr["HRR"]

A device with a line of points (``POINTS`` on ``&DEVC``) returns a list with one device
per point. With pandas installed,
:meth:`~fdsreader.devc.device_collection.DeviceCollection.to_pandas_dataframe` returns
all devices as one table.

What is where
-------------

.. list-table::
   :header-rows: 1

   * - Output
     - FDS input
     - Access
     - API
   * - Slices (2D and 3D)
     - ``&SLCF``
     - ``sim.slices``
     - :doc:`../api/slcf`
   * - Boundary data on obstructions
     - ``&BNDF``
     - ``sim.obstructions``
     - :doc:`../api/bndf`
   * - Complex geometry
     - ``&GEOM``
     - ``sim.geom_data``, ``sim.geomslices``
     - :doc:`../api/geom`
   * - Plot3D
     - ``DUMP PLOT3D_QUANTITY``
     - ``sim.data_3d``
     - :doc:`../api/pl3d`
   * - Smoke3D
     - ``&SMOKE3D``
     - ``sim.smoke_3d``
     - :doc:`../api/smoke3d`
   * - Isosurfaces
     - ``&ISOF``
     - ``sim.isosurfaces``
     - :doc:`../api/isof`
   * - Particles
     - ``&PART``
     - ``sim.particles``
     - :doc:`../api/part`
   * - Devices
     - ``&DEVC``
     - ``sim.devices``
     - :doc:`../api/devc`
   * - Profiles
     - ``&PROF``
     - ``sim.profiles``
     - :doc:`../api/quantity`
   * - Heat release rate, CPU usage
     - ``CHID_hrr.csv``, ``CHID_cpu.csv``
     - ``sim.hrr``, ``sim.cpu``
     - :doc:`../api/simulation`
   * - Meshes, surfaces, vents
     - ``&MESH``, ``&SURF``, ``&VENT``
     - ``sim.meshes``, ``sim.surfaces``, ``sim.ventilations``
     - :doc:`../api/fds_classes`

Smoke3D values are Smokeview display codes, not physical quantities; see the warning on
the :doc:`../api/smoke3d` page.

Data structure
--------------

Every output type follows the same pattern: a collection on the simulation holds one
object per output (for example one :class:`~fdsreader.slcf.slice.Slice` per ``&SLCF``
line), and that object holds one part per mesh. ``to_global()`` merges the parts of all
meshes into one array.

The simulation domain, meaning meshes, obstructions and the other objects defined in the
input file:

.. mermaid::

   flowchart LR
       sim([Simulation])
       sim -- meshes --> MeshCollection --> Mesh
       sim -- obstructions --> ObstructionCollection --> Obstruction --> SubObstruction
       sim -- surfaces --> Surface
       sim -- ventilations --> Ventilation --> SubVentilation
       sim -- geoms --> Geometry

The output data. Boundary data is attached to the obstructions above: each
``SubObstruction`` holds a ``Boundary`` per quantity, which holds one ``Patch`` per face.

.. mermaid::

   flowchart LR
       sim([Simulation])
       sim -- slices --> SliceCollection --> Slice --> SubSlice
       sim -- geomslices --> GeomSliceCollection --> GeomSlice --> SubGeomSlice
       sim -- geom_data --> GeometryCollection --> GeomBoundary
       sim -- data_3d --> Plot3DCollection --> Plot3D --> SubPlot3D
       sim -- smoke_3d --> Smoke3DCollection --> Smoke3D --> SubSmoke3D
       sim -- isosurfaces --> IsosurfaceCollection --> Isosurface --> SubSurface
       sim -- particles --> ParticleCollection --> Particle
       sim -- devices --> DeviceCollection --> Device
       sim -- profiles --> Profile
       sim -- evacs --> EvacCollection --> Evacuation

Objects that hold data also carry a :class:`~fdsreader.utils.data.Quantity` (name, short
name and unit), and the per-mesh parts refer to their :class:`~fdsreader.fds_classes.mesh.Mesh`.
``sim.hrr`` and ``sim.cpu`` are plain dictionaries of NumPy arrays.

More complete scripts for every data type are collected under :doc:`examples`.
