import glob
import logging
import os
import warnings
from typing import List, TextIO, Dict, AnyStr, Sequence, Tuple, Union

import numpy as np
import pickle

from fdsreader.fds_classes import Mesh, MeshCollection, Surface, Ventilation
from fdsreader.bndf import Obstruction, Patch, ObstructionCollection, SubObstruction
from fdsreader.geom import Geometry, GeomBoundary, GeometryCollection
from fdsreader.isof import Isosurface, IsosurfaceCollection
from fdsreader.part import Particle, ParticleCollection
from fdsreader.evac import Evacuation, EvacCollection
from fdsreader.pl3d import Plot3D, Plot3DCollection
from fdsreader.smoke3d import Smoke3D, Smoke3DCollection
from fdsreader.slcf import Slice, SliceCollection, GeomSliceCollection, GeomSlice
from fdsreader.devc import Device, DeviceCollection
from fdsreader.utils import Dimension, Quantity, Extent, log_error
from fdsreader.utils.data import create_hash, get_smv_file, Profile
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings
from fdsreader._version import __version__


class Simulation:
    """Master class managing all data for a given simulation.

    :ivar reader_version: The version of the fdsreader used to load the Simulation.
    :ivar smv_file_path: Path to the .smv file of the simulation.
    :ivar root_path: Path to the root directory of the simulation.
    :ivar fds_version: Version of FDS the simulation was performed with.
    :ivar chid: Name (ID) of the simulation.
    :ivar hrrpuv_cutoff: The hrrpuv_cutoff value.
    :ivar default_texture_origin: The default origin used for textures with no explicit origin.
    :ivar out_file_path: Path to the .out file of the simulation.
    :ivar surfaces: List containing all surfaces defined in this simulation.
    :ivar meshes: List containing all meshes (grids) defined in this simulation.
    :ivar ventilations: List containing all ventilations defined in this simulation.
    :ivar obstructions: All defined obstructions combined into a :class:`ObstructionCollection`.
    :ivar slices: All defined slices combined into a :class:`SliceCollection`.
    :ivar data_3d: All defined 3D plotting data combined into a :class:`Plot3DCollection`.
    :ivar smoke_3d: All defined 3D smoke data combine into a :class:`Smoke3DCollecction`
    :ivar isosurfaces: All defined isosurfaces combined into a :class:`IsosurfaceCollection`.
    :ivar particles: All defined particles combined into a :class:`ParticleCollection`.
    :ivar evacs: All defined evacuations combined into a :class:`EvacCollection`.
    :ivar devices: List containing all :class:`Device` s defined in this simulation.
    :ivar profiles: Dictionary mapping profile ids to the corresponding :class:`Profile` s defined in this simulation.
    :ivar geoms: List containing all geometries (:class:`Geometry`) defined in this simulation.
    :ivar geom_data: All geometry data by quantity combined into a :class:`GeometryCollection`.
    :ivar cpu: Dictionary mapping .csv header keys to numpy arrays containing cpu data.
    :ivar hrr: Dictionary mapping .csv header keys to numpy arrays containing hrr data.
    :ivar steps: Dictionary mapping .csv header keys to numpy arrays containing steps data.
    """

    _loading = False

    def __new__(cls, path: str):
        smv_file_path = get_smv_file(path)
        root_path = os.path.dirname(smv_file_path)

        with open(smv_file_path, 'r') as infile:
            for line in infile:
                if line.strip() == "CHID":
                    chid = infile.readline().strip()
                    break

        pickle_file_path = Simulation._get_pickle_filename(root_path, chid)
        if settings.ENABLE_CACHING:
            if not Simulation._loading and os.path.isfile(pickle_file_path):
                Simulation._loading = True
                try:
                    with open(pickle_file_path, 'rb') as f:
                        sim = pickle.load(f)
                except Exception as e:
                    if settings.DEBUG:
                        logging.exception(e)
                else:
                    valid = True
                    # Check if pickle file stores a Simulation
                    valid &= isinstance(sim, cls)
                    # Check if the fdsreader version still matches
                    valid &= sim.reader_version == __version__
                    # Check if the smv_file did not change
                    valid &= sim._hash == create_hash(smv_file_path)

                    if valid:
                        # Return cached sim if it turned out to be valid
                        return sim

                os.remove(pickle_file_path)
        else:
            if os.path.isfile(pickle_file_path):
                os.remove(pickle_file_path)
        return super(Simulation, cls).__new__(cls)

    def __getnewargs__(self):
        return self.smv_file_path,

    def __repr__(self):
        r = f"Simulation(chid={self.chid},\n" + \
            f"           meshes={len(self.meshes)},\n" + \
            (f"           obstructions={len(self.obstructions)},\n" if len(self.obstructions) > 0 else "") + \
            (f"           geoms={len(self.geoms)},\n" if len(self.geoms) > 0 else "") + \
            (f"           slices={len(self.slices)},\n" if len(self.slices) > 0 else "") + \
            (f"           geomslices={len(self.geomslices)},\n" if len(self.geomslices) > 0 else "") + \
            (f"           data_3d={len(self.data_3d)},\n" if len(self.data_3d) > 0 else "") + \
            (f"           smoke_3d={len(self.smoke_3d)},\n" if len(self.smoke_3d) > 0 else "") + \
            (f"           isosurfaces={len(self.isosurfaces)},\n" if len(self.isosurfaces) > 0 else "") + \
            (f"           particles={len(self.particles)},\n" if len(self.particles) > 0 else "") + \
            (f"           evacs={len(self.evacs)},\n" if len(self.evacs) > 0 else "") + \
            (f"           devices={len(self.devices)},\n" if len(self.devices) > 0 else "")
        return r[:-2] + ')'

    def __init__(self, path: str):
        """
        :param path: Either the path to the directory containing the simulation data or direct path
            to the .smv file for the simulation in case that multiple simulation output was written to
            the same directory.
        """
        if settings.IGNORE_ERRORS:
            warnings.filterwarnings("ignore")

        # Check if the file has already been instantiated via a cached pickle file
        if not hasattr(self, "_hash"):
            self.reader_version = __version__

            self.smv_file_path = get_smv_file(path)

            self.root_path = os.path.dirname(self.smv_file_path)

            self.geoms: List[Geometry] = list()
            self.surfaces: List[Surface] = list()
            self.ventilations = dict()

            # Will only be used during the loading process to map boundary data to the correct
            # obstruction
            self._subobstructions: Dict[str, List[SubObstruction]] = dict()

            # First collect all meta-information for any FDS data to later combine the gathered
            # information into data collections. While collecting the meta-data simple python
            # containers are used.
            self._obstructions = list()
            self._slices = dict()
            self._geomslices = dict()
            self._data_3d = dict()
            self._smoke_3d = dict()
            self._isosurfaces = dict()
            self._particles = list()
            self._evacs = list()
            self._geom_data = list()
            self._meshes: List[Mesh] = list()
            self._devices = dict()

            self.profiles: Dict[str, Profile] = dict()

            self.parse_smv_file()

            self.cpu = self._load_CPU_data()
            self._load_profiles()

            # POST INIT (post read)
            self.out_file_path = os.path.join(self.root_path, self.chid + ".out")
            self.ventilations: List[Ventilation] = list(self.ventilations.values())

            for device_id, device in self._devices.items():
                if type(device) == list:
                    for devc in device:
                        devc._data_callback = self._load_DEVC_data
                else:
                    device._data_callback = self._load_DEVC_data

            # Combine the gathered temporary information into data collections
            self.geom_data = GeometryCollection(self._geom_data)
            self.slices = SliceCollection(
                Slice(self.root_path, slice_data[0]["id"], slice_data[0]["cell_centered"],
                      slice_data[0]["times"], slice_data[1:]) for slice_data in self._slices.values())
            self.geomslices = GeomSliceCollection(
                GeomSlice(self.root_path, slice_data[0]["id"],
                          slice_data[0]["times"], slice_data[1:]) for slice_data in self._geomslices.values())
            self.data_3d = Plot3DCollection(self._data_3d.keys(), self._data_3d.values())
            self.smoke_3d = Smoke3DCollection(self._smoke_3d.values())
            self.isosurfaces = IsosurfaceCollection(self._isosurfaces.values())
            self.devices = DeviceCollection(self._devices.values())
            self.obstructions = ObstructionCollection(self._obstructions)
            # If no particles are simulated, initialize empty data container for consistency
            if type(self._particles) == list:
                self.particles = ParticleCollection((), ())
            else:
                self.particles = self._particles
                self.particles._post_init()
            # If no evacs are simulates, initialize empty data container for consistency
            if len(self._evacs) == 0:
                self.evacs = EvacCollection((), "", ())
            self.meshes = MeshCollection(self._meshes)
            del self._geom_data, self._geomslices, self._slices, self._obstructions, self._data_3d, self._smoke_3d, self._isosurfaces, self._devices, self._particles, self._evacs, self._meshes, self._subobstructions

            if settings.ENABLE_CACHING:
                # Hash will be saved to simulation pickle file and compared to new hash when loading
                # the pickled simulation again in the next run of the program.
                self._hash = create_hash(self.smv_file_path)
                pickle.dump(self, open(Simulation._get_pickle_filename(self.root_path, self.chid), 'wb'), protocol=4)

    def parse_smv_file(self):
        with open(self.smv_file_path, 'r') as smv_file:
            for line in smv_file:
                keyword = line.strip()
                if keyword == "VERSION" or keyword == "FDSVERSION":
                    self.fds_version = smv_file.readline().strip()
                elif keyword == "CHID":
                    self.chid = smv_file.readline().strip()
                elif keyword == "CSVF":
                    csv_type = smv_file.readline().strip()
                    filename = smv_file.readline().strip()
                    file_path = os.path.join(self.root_path, filename)
                    if csv_type == "hrr":
                        self.hrr = self._load_HRR_data(file_path)
                    elif csv_type == "steps":
                        self.steps = self._load_step_data(file_path)
                    elif csv_type == "devc":
                        self.devc_path = file_path
                        self._devices["Time"] = Device("Time", Quantity("TIME", "TIME", "s"), (.0, .0, .0),
                                                       (.0, .0, .0))
                elif keyword == "HRRPUVCUT":
                    self.hrrpuv_cutoff = float(smv_file.readline().strip())
                elif keyword == "TOFFSET":
                    offsets = smv_file.readline().strip().split()
                    self.default_texture_origin = tuple(float(offsets[i]) for i in range(3))
                elif keyword == "CLASS_OF_PARTICLES":
                    self._particles.append(self._register_particle(smv_file))
                elif keyword == "CLASS_OF_HUMANS":
                    self._evacs.append(self._register_evac(smv_file))
                elif keyword.startswith("GEOM"):
                    self._load_geoms(smv_file, keyword)
                elif keyword.startswith("GRID"):
                    self._meshes.append(self._load_mesh(smv_file, keyword))
                elif keyword == "SURFACE":
                    self.surfaces.append(self._load_surface(smv_file))
                elif keyword == "DEVICE":
                    device_id, device = self._register_device(smv_file)
                    if device_id in self._devices:
                        if type(self._devices[device_id]) == list:
                            self._devices[device_id].append(device)
                        else:
                            self._devices[device_id] = [self._devices[device_id], device]
                    else:
                        self._devices[device_id] = device
                elif keyword.startswith("SLC"):
                    self._load_slice(smv_file, keyword)
                elif keyword.startswith("BNDS"):
                    self._load_geomslice(smv_file, keyword)
                elif "ISOG" in keyword:
                    self._load_isosurface(smv_file, keyword)
                elif keyword.startswith("PL3D"):
                    self._load_plot_3d(smv_file, keyword)
                elif keyword.startswith("SMOKG3D") or keyword.startswith("SMOKF3D"):
                    self._load_smoke_3d(smv_file, keyword)
                elif keyword.startswith("BNDF"):
                    self._load_boundary_data(smv_file, keyword, cell_centered=False)
                elif keyword.startswith("BNDC"):
                    self._load_boundary_data(smv_file, keyword, cell_centered=True)
                elif keyword.startswith("BNDE"):
                    self._load_boundary_data_geom(smv_file, keyword)
                elif keyword.startswith("PRT5"):
                    self._load_particle_data(smv_file, keyword)
                elif keyword.startswith("EVA5"):
                    self._load_evac_data(smv_file, keyword)
                elif keyword.startswith("SHOW_OBST"):
                    self._toggle_obst(smv_file, keyword)
                elif keyword.startswith("HIDE_OBST"):
                    self._toggle_obst(smv_file, keyword)

    @classmethod
    def _get_pickle_filename(cls, root_path: str, chid: str) -> AnyStr:
        """Get the filename used to save the pickled simulation.
        """
        return os.path.join(root_path, chid + ".pickle")

    def _load_mesh(self, smv_file: TextIO, line: str) -> Mesh:
        """Load information for a single mesh from the smv file at current pointer position.
        """
        mesh_id = "".join(line.split()[1:])

        grid_numbers = smv_file.readline().strip().split()
        grid_dimensions = {'x': int(grid_numbers[0]) + 1, 'y': int(grid_numbers[1]) + 1,
                           'z': int(grid_numbers[2]) + 1}

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "PDIM"
        coordinates = dict()
        extents = smv_file.readline().split()
        extents = {'x': (float(extents[0]), float(extents[1])),
                   'y': (float(extents[2]), float(extents[3])),
                   'z': (float(extents[4]), float(extents[5]))}

        for dim in ('x', 'y', 'z'):
            smv_file.readline()  # Blank line
            assert smv_file.readline().strip()[:3] == "TRN"
            noc = int(smv_file.readline().strip())
            for _ in range(noc):
                smv_file.readline()
            coordinates[dim] = np.empty(grid_dimensions[dim], dtype=np.float32)
            for i in range(grid_dimensions[dim]):
                coordinates[dim][i] = float(smv_file.readline().split()[1])

        mesh = Mesh(coordinates, extents, mesh_id)

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "OBST"
        self._load_obstructions(smv_file, mesh)

        a = smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "VENT"
        self._load_vents(smv_file, mesh)

        return mesh

    @log_error("obst")
    def _load_obstructions(self, smv_file: TextIO, mesh: Mesh):
        temp_data = dict()
        n = int(smv_file.readline().strip())

        if n > 0:
            self._subobstructions[mesh.id] = list()

        for _ in range(n):
            line = smv_file.readline().strip().split('!')
            line_floats = line[0].strip().split()
            ext = [float(line_floats[i]) for i in range(6)]
            # The ordinal is negative if the obstruction was created due to a hole, the negative
            # sign is ignored here and obstructions created by FDS due to holes are handled later
            # when there are multiple obstructions with the same ID
            obst_id = line[1].strip() if len(line) > 1 else str(abs(int(line_floats[6])))

            side_surfaces = tuple(self.surfaces[int(line_floats[i]) - 1] for i in range(7, 13))
            if len(line_floats) > 13:
                texture_origin = (float(line_floats[13]), float(line_floats[14]), float(line_floats[15]))
            else:
                texture_origin = self.default_texture_origin

            # Check if there is already an obst in this mesh with the same ID (due to holes).
            # This will only be the case the first time we notice that there are multiple
            # obstructions with the same name, as their IDs will now change...
            if obst_id in temp_data:
                # The obstruction that was already added to the temp_data has to receive an
                # updated ID as well, so the user can recognize it as special obstruction
                temp_data[obst_id + '_from-hole-1'] = temp_data[obst_id]
                del temp_data[obst_id]

                # Now set the next obst_id to '_from-hole-2'
                obst_id = obst_id + '_from-hole-2'

            # ...For subsequent cases (i.e., when the third or fourth obstruction with the same id
            # is found), we need to catch the case differently
            if obst_id + '_from-hole-1' in temp_data:
                # Find the counter of the last added obstruction with the same id and set the id of
                # the current obstruction to the next following number.
                # Starting at 3, as 1 and 2 are guaranteed to be in the dictionary already, due to
                # the if-branch above.
                counter = 3
                obst_id = obst_id + '_from-hole-' + str(counter)
                while obst_id in temp_data:
                    counter += 1
                    obst_id = obst_id[:-1] + str(counter)

            temp_data[obst_id] = (Extent(*ext), side_surfaces, texture_origin)

        for obst_id, tmp in temp_data.items():
            extent, side_surfaces, texture_origin = tmp

            line = smv_file.readline().strip().split()
            bound_indices = (
                int(float(line[0])), int(float(line[1])), int(float(line[2])),
                int(float(line[3])), int(float(line[4])), int(float(line[5])))
            color_index = int(line[6])
            block_type = int(line[7])
            rgba = tuple(float(line[i]) for i in range(8, 12)) if color_index == -3 else ()

            obst = next((o for o in self._obstructions if obst_id == o.id), None)
            if obst is None:
                obst = Obstruction(obst_id, color_index, block_type, texture_origin, rgba)
                self._obstructions.append(obst)

            if not any(obst_id == o.id for o in mesh.obstructions):
                mesh.obstructions.append(obst)

            subobst = SubObstruction(side_surfaces, bound_indices, extent, mesh)

            self._subobstructions[mesh.id].append(subobst)
            obst._subobstructions[mesh.id] = subobst

    def _toggle_obst(self, smv_file: TextIO, line: str):
        line = line.split()
        mesh = self._meshes[int(line[-1]) - 1]

        obst_index, time = smv_file.readline().split()
        time = float(time)
        subobst = self._subobstructions[mesh.id][int(obst_index) - 1]

        if "HIDE_OBST" in line[0]:
            subobst._hide(time)
        else:
            subobst._show(time)

    def _load_geoms(self, smv_file: TextIO, line: str):
        ngeoms = int(line.split()[1])

        filename = smv_file.readline()
        file_path = os.path.join(self.root_path, filename)

        for g in range(ngeoms):
            line = smv_file.readline().split("!")
            texture_line = line[0].split()
            rgb_line = line[1].split()

            texture_mapping = texture_line[0]
            texture_origin = (
                float(texture_line[1]), float(texture_line[2]), float(texture_line[3]))
            is_terrain = bool(texture_line[4])
            rgb = (int(rgb_line[0]), int(rgb_line[1]), int(rgb_line[2]))

            if '%' in line[0]:
                surface_id = line[0].split("%")[-1]
                surface = next((s for s in self.surfaces if s.id() == surface_id), None)
                geom = Geometry(file_path, texture_mapping, texture_origin, is_terrain, rgb,
                                surface=surface)
            else:
                geom = Geometry(file_path, texture_mapping, texture_origin, is_terrain, rgb)
            self.geoms.append(geom)

    @log_error("vents")
    def _load_vents(self, smv_file: TextIO, mesh: Mesh):
        line = smv_file.readline().split()
        n, n_dummies = int(line[0]), int(line[1])

        temp_data = list()

        def read_common_info():
            line = smv_file.readline().strip().split()
            return line, [float(line[i]) for i in range(6)], int(line[6]) - 1, self.surfaces[
                int(line[7])]

        def read_common_info2():
            line = smv_file.readline().strip().split()
            bound_indices = tuple(int(line[i]) for i in range(6))
            color_index = int(line[6])
            draw_type = int(line[7])
            if len(line) > 8:
                rgba = tuple(float(line[i]) for i in range(8, 12))
            else:
                rgba = ()
            return bound_indices, color_index, draw_type, rgba

        texture_origin = ()
        for _ in range(n - n_dummies):
            line, ext, vent_index, surface = read_common_info()
            texture_origin = (float(line[8]), float(line[9]), float(line[10]))
            temp_data.append((Extent(*ext), vent_index, surface, texture_origin))

        for _ in range(n_dummies):
            _, ext, vent_index, surface = read_common_info()
            temp_data.append((Extent(*ext), vent_index, surface))

        for v in range(n):
            if v < n - n_dummies:
                extent, vent_index, surface, texture_origin = temp_data[v]
            else:
                extent, vent_index, surface = temp_data[v]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            if vent_index not in self.ventilations:
                self.ventilations[vent_index] = Ventilation(surface, bound_indices, color_index,
                                                            draw_type, rgba=rgba,
                                                            texture_origin=texture_origin)
            self.ventilations[vent_index]._add_subventilation(mesh, extent)

        smv_file.readline()
        assert "CVENT" in smv_file.readline()

        n = int(smv_file.readline().strip())
        temp_data.clear()
        for _ in range(n):
            line, extent, vent_index, surface = read_common_info()
            circular_vent_origin = (float(line[12]), float(line[13]), float(line[14]))
            radius = float(line[15])
            temp_data.append(
                (extent, vent_index, surface, texture_origin, circular_vent_origin, radius))

        for v in range(n):
            extent, vent_index, surface, texture_origin, circular_vent_origin, radius = temp_data[v]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            if vent_index not in self.ventilations:
                self.ventilations[vent_index] = Ventilation(surface, bound_indices,
                                                            color_index, draw_type, rgba=rgba,
                                                            texture_origin=texture_origin,
                                                            circular_vent_origin=circular_vent_origin,
                                                            radius=radius)
            self.ventilations[vent_index]._add_subventilation(mesh, extent)

    @log_error("surface")
    def _load_surface(self, smv_file: TextIO) -> Surface:
        """Load the information for a single surface from the smv file at current pointer position.
        """

        surface_id = smv_file.readline().strip()

        line = smv_file.readline().strip().split()
        tmpm, material_emissivity = float(line[0]), float(line[1])

        line = smv_file.readline().strip().split()

        surface_type = int(line[0])
        texture_width, texture_height = float(line[1]), float(line[2])
        rgb = (float(line[3]), float(line[4]), float(line[5]))
        transparency = float(line[6])

        texture_map = smv_file.readline().strip()
        texture_map = None if texture_map == "null" else os.path.join(self.root_path, texture_map)

        return Surface(surface_id, tmpm, material_emissivity, surface_type, texture_width,
                       texture_height, texture_map, rgb, transparency)

    @log_error("slcf")
    def _load_slice(self, smv_file: TextIO, line: str):
        """Loads the slice at current pointer position.
        """
        if "SLCC" in line:
            cell_centered = True
        else:
            cell_centered = False

        slice_index = int(line.split('!')[1].strip().split()[0])

        slice_id = line.split('%')[1].split('&')[0].strip() if '%' in line else ""

        mesh_index = int(line.split('&')[0].strip().split()[1]) - 1
        mesh = self._meshes[mesh_index]

        # Read in index ranges for x, y and z
        bound_indices = [int(i.strip()) for i in line.split('&')[1].split('!')[0].strip().split()]
        extent, dimension = self._indices_to_extent(bound_indices, mesh)

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        file_path = os.path.join(self.root_path, filename)

        if os.path.exists(file_path + ".bnd"):
            times = list()
            with open(file_path + ".bnd", 'r') as bnd_file:
                for line in bnd_file:
                    times.append(float(line.split()[0]))
            times = np.array(times)
        else:
            times = None

        if slice_index not in self._slices:
            self._slices[slice_index] = [
                {"cell_centered": cell_centered, "times": times, "id": slice_id}]
        self._slices[slice_index].append(
            {"dimension": dimension, "extent": extent, "mesh": mesh, "filename": filename,
             "quantity": quantity, "short_name": short_name, "unit": unit})

    @log_error("slcf")
    def _load_geomslice(self, smv_file: TextIO, line: str):
        """Loads the geomslice at current pointer position.
        """
        slice_index = int(line.split('!')[1].strip().split()[0])

        slice_id = "".join(line.split('%')[1].split('&')).strip() if '%' in line else ""

        mesh_index = int(line.split('&')[0].strip().split()[1]) - 1
        mesh = self._meshes[mesh_index]

        # Read in index ranges for x, y and z
        bound_indices = [int(i.strip()) for i in line.split('&')[1].split('!')[0].strip().split()]
        extent, _ = self._indices_to_extent(bound_indices, mesh)

        filename = smv_file.readline().strip()
        geom_filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        file_path = os.path.join(self.root_path, filename)

        if os.path.exists(file_path + ".bnd"):
            times = list()
            with open(file_path + ".bnd", 'r') as bnd_file:
                for line in bnd_file:
                    times.append(float(line.split()[0]))
            times = np.array(times)
        else:
            times = None

        if slice_index not in self._slices:
            self._geomslices[slice_index] = [
                {"times": times, "id": slice_id}]
        self._geomslices[slice_index].append(
            {"extent": extent, "mesh": mesh, "filename": filename, "geomfilename": geom_filename,
             "quantity": quantity, "short_name": short_name, "unit": unit})

    @log_error("bndf")
    def _load_boundary_data(self, smv_file: TextIO, line: str, cell_centered: bool):
        """Loads the boundary data at current pointer position.
        """
        line = line.split()
        mesh_index = int(line[1]) - 1
        mesh = self._meshes[mesh_index]

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        bid = int(filename.split('_')[-1][:-3]) - 1

        file_path = os.path.join(self.root_path, filename)

        patches = dict()
        mesh_patches = dict()

        if os.path.exists(file_path + ".bnd"):
            times = list()
            lower_bounds = list()
            upper_bounds = list()
            with open(file_path + ".bnd", 'r') as bnd_file:
                for line in bnd_file:
                    splits = line.split()
                    times.append(float(splits[0]))
                    lower_bounds.append(float(splits[1]))
                    upper_bounds.append(float(splits[2]))
            times = np.array(times)
            lower_bounds = np.array(lower_bounds, dtype=np.float32)
            upper_bounds = np.array(upper_bounds, dtype=np.float32)
            n_t = times.shape[0]
        else:
            times = None
            n_t = -1
            lower_bounds = np.array([0.0], dtype=np.float32)
            upper_bounds = np.array([np.float32(-1e33)], dtype=np.float32)

        with open(file_path, 'rb') as infile:
            # Offset of the binary file to the end of the file header.
            offset = 3 * fdtype.new((('c', 30),)).itemsize
            infile.seek(offset)

            n_patches = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            dtype_patches = fdtype.new((('i', 9),))
            patch_infos = fdtype.read(infile, dtype_patches, n_patches)
            offset += fdtype.INT.itemsize + dtype_patches.itemsize * n_patches
            patch_offset = fdtype.FLOAT.itemsize

            for patch_info in patch_infos:
                patch_info = patch_info[0]

                extent, dimension = self._indices_to_extent(patch_info[:6], mesh)
                orientation = patch_info[6]
                obst_index = patch_info[7]

                p = Patch(file_path, dimension, extent, orientation, cell_centered,
                          patch_offset, offset, n_t, mesh)

                # "Obstacles" with index 0 give the extent of the (whole) mesh faces and refer to
                # "closed" mesh faces, therefore that data will be added to the corresponding mesh instead
                if obst_index != 0:
                    obst_index -= 1  # Account for fortran indexing
                    if obst_index not in patches:
                        patches[obst_index] = list()
                    patches[obst_index].append(p)
                else:
                    if mesh.id not in mesh_patches:
                        mesh_patches[mesh.id] = list()
                    mesh_patches[mesh.id].append(p)

                patch_offset += fdtype.new((('f', str(p.dimension.shape(cell_centered=False))),)).itemsize

        for obst_index, p in patches.items():
            for patch in p:
                patch._post_init(patch_offset)

            self._subobstructions[mesh.id][obst_index]._add_patches(bid, cell_centered, quantity, short_name, unit, p,
                                                                 times, lower_bounds, upper_bounds)

        for p in mesh_patches.values():
            for patch in p:
                patch._post_init(patch_offset)
            patch.mesh._add_patches(bid, cell_centered, quantity, short_name, unit, p, times, lower_bounds, upper_bounds)

    @log_error("geom")
    def _load_boundary_data_geom(self, smv_file: TextIO, line: str):
        line = line.split()
        mesh_index = int(line[1]) - 1
        # Meshes are not loaded yet
        # mesh = self.meshes[mesh_index]

        filename_be = smv_file.readline().strip()
        filename_gbf = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        bid = int(filename_be.split('_')[-1][:-3]) - 1

        file_path_be = os.path.join(self.root_path, filename_be)
        file_path_gbf = os.path.join(self.root_path, filename_gbf)

        times = list()
        lower_bounds = list()
        upper_bounds = list()
        with open(file_path_be + ".bnd", 'r') as bnd_file:
            for line in bnd_file:
                splits = line.split()
                times.append(float(splits[0]))
                lower_bounds.append(float(splits[1]))
                upper_bounds.append(float(splits[2]))
        times = np.array(times)
        lower_bounds = np.array(lower_bounds, dtype=np.float32)
        upper_bounds = np.array(upper_bounds, dtype=np.float32)
        n_t = times.shape[0]

        if bid >= len(self._geom_data):
            self._geom_data.append(GeomBoundary(Quantity(quantity, short_name, unit), times, n_t))
        self._geom_data[bid]._add_data(mesh_index, file_path_be, file_path_gbf, lower_bounds,
                                       upper_bounds)

    @log_error("pl3d")
    def _load_plot_3d(self, smv_file: TextIO, line: str):
        """Loads the pl3d at current pointer position.
        """
        line = line.strip().split()

        time = float(line[1])

        mesh_index = int(line[2]) - 1

        filename = smv_file.readline().strip()
        quantities = list()
        for _ in range(5):
            quantity = smv_file.readline().strip()
            short_name = smv_file.readline().strip()
            unit = smv_file.readline().strip()
            quantities.append(Quantity(quantity, short_name, unit))

        if time not in self._data_3d:
            self._data_3d[time] = Plot3D(self.root_path, time, quantities)
        self._data_3d[time]._add_subplot(filename, self._meshes[mesh_index])

    @log_error("smoke3d")
    def _load_smoke_3d(self, smv_file: TextIO, line: str):
        """Loads the smoke3d at current pointer position.
        """
        line = line.strip().split()

        mesh_index = int(line[1]) - 1

        filename = smv_file.readline().strip()

        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        times = list()
        upper_bounds = list()
        with open(os.path.join(self.root_path, filename + ".sz"), 'r') as sizefile:
            # skip version line
            sizefile.readline()
            for line in sizefile:
                line = line.split()
                times.append(float(line[0]))
                upper_bounds.append(float(line[-1]))
        times = np.array(times)
        upper_bounds = np.array(upper_bounds)

        quantity = Quantity(quantity, short_name, unit)

        if quantity not in self._smoke_3d:
            self._smoke_3d[quantity] = Smoke3D(self.root_path, times, quantity)
        self._smoke_3d[quantity]._add_subsmoke(filename, self._meshes[mesh_index], upper_bounds)

    @log_error("isof")
    def _load_isosurface(self, smv_file: TextIO, line: str):
        """Loads the isosurface at current pointer position.
        """
        double_quantity = line[0] == 'T'
        mesh_index = int(line.strip().split()[1]) - 1

        iso_filename = smv_file.readline().strip()
        iso_id = int(iso_filename.split('_')[-1][:-4])
        iso_file_path = os.path.join(self.root_path, iso_filename)

        if double_quantity:
            viso_file_path = os.path.join(self.root_path, smv_file.readline().strip())
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()
        if double_quantity:
            v_quantity = smv_file.readline().strip()
            v_short_name = smv_file.readline().strip()
            v_unit = smv_file.readline().strip()

        if iso_id not in self._isosurfaces:
            with open(iso_file_path, 'rb') as infile:
                nlevels = fdtype.read(infile, fdtype.INT, 3)[2][0][0]

                dtype_header_levels = fdtype.new((('f', nlevels),))
                levels = fdtype.read(infile, dtype_header_levels, 1)[0]
        if double_quantity:
            if iso_id not in self._isosurfaces:
                self._isosurfaces[iso_id] = Isosurface(iso_id, double_quantity, quantity, short_name,
                                                       unit, levels, v_quantity=v_quantity,
                                                       v_short_name=v_short_name, v_unit=v_unit)
            self._isosurfaces[iso_id]._add_subsurface(self._meshes[mesh_index], iso_file_path,
                                                      viso_file_path=viso_file_path)
        else:
            if iso_id not in self._isosurfaces:
                self._isosurfaces[iso_id] = Isosurface(iso_id, double_quantity, quantity, short_name,
                                                       unit, levels)
            self._isosurfaces[iso_id]._add_subsurface(self._meshes[mesh_index], iso_file_path)

    @log_error("part")
    def _register_particle(self, smv_file: TextIO) -> Particle:
        particle_class = smv_file.readline().strip()
        color = tuple(float(c) for c in smv_file.readline().strip().split())

        n_quantities = int(smv_file.readline().strip())
        quantities = list()
        for _ in range(n_quantities):
            quantity = smv_file.readline().strip()
            short_name = smv_file.readline().strip()
            unit = smv_file.readline().strip()
            quantities.append(Quantity(quantity, short_name, unit))
        return Particle(particle_class, quantities, color)

    def _load_prt5_meta(self, prts: Union[List[Particle], ParticleCollection, List[Evacuation], EvacCollection],
                        file_path: str, mesh: Mesh) -> List[float]:
        is_evac = type(prts[0]) == Evacuation
        with open(file_path, 'r') as bnd_file:
            line = bnd_file.readline().strip().split()
            n_classes = int(line[1])
            times = list()
            n_quantities = list()
            for i in range(n_classes):
                line = bnd_file.readline().strip().split()
                n_quantities.append(int(line[0]))
                for _ in range(n_quantities[-1]):
                    bnd_file.readline()
                if is_evac:
                    prts[i].n_humans[mesh.id] = list()
                else:
                    prts[i].n_particles[mesh.id] = list()
            bnd_file.seek(0)

            for line in bnd_file:
                times.append(float(line.strip().split()[0]))
                for i in range(n_classes):
                    prt = prts[i]
                    n = int(bnd_file.readline().strip().split()[1].strip())
                    if is_evac:
                        prt.n_humans[mesh.id].append(n)
                    else:
                        prt.n_particles[mesh.id].append(n)
                    for q in range(n_quantities[i]):
                        line = bnd_file.readline().strip().split()
                        quantity = prt.quantities[q].name
                        prt.lower_bounds[quantity].append(float(line[0]))
                        prt.upper_bounds[quantity].append(float(line[1]))
        return times

    @log_error("part")
    def _load_particle_data(self, smv_file: TextIO, line: str):
        file_path = os.path.join(self.root_path, smv_file.readline().strip())

        mesh_index = int(line.split()[1].strip()) - 1
        mesh = self._meshes[mesh_index]

        times = self._load_prt5_meta(self._particles, file_path + '.bnd', mesh)
        if type(self._particles) == list:
            self._particles = ParticleCollection(times, self._particles)

        self._particles._file_paths[mesh.id] = file_path

        n_classes = int(smv_file.readline().strip())
        for i in range(n_classes):
            smv_file.readline()  # Skip "N" values

    @log_error("evac")
    def _register_evac(self, smv_file: TextIO) -> Evacuation:
        class_name = smv_file.readline().split(" % % ")[0].strip()
        color = tuple(float(c) for c in smv_file.readline().strip().split())

        n_quantities = int(smv_file.readline().strip())
        quantities = list()
        for _ in range(n_quantities):
            quantity = smv_file.readline().strip()
            short_name = smv_file.readline().strip()
            unit = smv_file.readline().strip()
            quantities.append(Quantity(quantity, short_name, unit))
        return Evacuation(class_name, quantities, color)

    @log_error("evac")
    def _load_evac_data(self, smv_file: TextIO, line: str):
        file_path = os.path.join(self.root_path, smv_file.readline().strip())

        mesh_index, z_offset = line.split()[1:]
        mesh = self._meshes[int(mesh_index) - 1]

        times = self._load_prt5_meta(self._evacs, file_path + '.bnd', mesh)[1:]  # First timestep is weird somehow
        if type(self._evacs) == list:
            self.evacs = EvacCollection(self._evacs, os.path.join(self.root_path, self.chid + "_evac"), times)

        self.evacs.z_offsets[mesh.id] = float(z_offset)
        self.evacs._file_paths[mesh.id] = file_path

        n_evacs = int(smv_file.readline().strip())
        for i in range(n_evacs):
            smv_file.readline()  # Skip "N" values

    @log_error("prof")
    def _load_profiles(self):
        for f in glob.glob(str(os.path.join(self.root_path, self.chid)) + "_prof*"):
            with open(f, 'r') as infile:
                profile_id = infile.readline()
                infile.readline()  # Skip header
                data: np.ndarray = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True).T
                times = data[0]
                npoints = data[1].astype(int)
                depths = np.empty((data.shape[1],), dtype=object)
                values = np.empty((data.shape[1],), dtype=object)

                for i, n in enumerate(npoints):
                    depths[i] = data[2: 2 + n, i]
                    values[i] = data[2 + n:, i]

                self.profiles[profile_id] = Profile(profile_id, times, npoints, depths, values)

    @log_error("devc")
    def _register_device(self, smv_file: TextIO) -> Tuple[str, Device]:
        line = smv_file.readline().strip().split('%')
        device_id = line[0].strip()
        quantity = None
        if len(line) > 1:
            quantity_name = line[1].strip()
            quantity = Quantity(quantity_name, quantity_name, "")
        line = smv_file.readline().strip().split('#')[0].split()
        position = (float(line[0]), float(line[1]), float(line[2]))
        orientation = (float(line[3]), float(line[4]), float(line[5]))
        return device_id, Device(device_id, quantity, position, orientation)

    def _load_DEVC_data(self):
        with open(self.devc_path, 'r') as infile:
            units = infile.readline().split(',')
            names = [name.replace('"', '').replace('\n', '').strip() for name in infile.readline().split(',"')]
            values = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
            for k in range(len(names)):
                if type(self.devices[names[k]]) == list:
                    for devc in self.devices[names[k]]:
                        if not hasattr(devc, "_data"):
                            # Find the first device in the list that does not yet have any data associated with it
                            break
                else:
                    devc = self.devices[names[k]]

                devc.quantity.unit = units[k]
                size = values.shape[0]
                devc._data = np.empty((size,), dtype=np.float32)
                for i in range(size):
                    devc._data[i] = values[i][k]

        line_path = self.devc_path.replace("devc", "line")
        if os.path.exists(line_path):
            with open(line_path, 'r') as infile:
                units = infile.readline()
                names = [name.replace('"', '').replace('\n', '').strip() for name in infile.readline().split(',')]
                data = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
                for k, key in enumerate(names):
                    if key in self.devices:
                        devc = self.devices[key]
                        for i in range(len(devc)):
                            devc[i].quantity.unit = units[k]
                            devc[i]._data = data[i, k]
                    else:
                        pass  # Probably only x,y,z coordinates

    @log_error("csv")
    def _load_HRR_data(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path, 'r') as infile:
            infile.readline()
            keys = [name.replace('"', '').replace('\n', '').strip() for name in infile.readline().split(',')]

        values = np.loadtxt(file_path, delimiter=',', ndmin=2, skiprows=2)
        return self._transform_csv_data(keys, values)

    @log_error("csv")
    def _load_step_data(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path, 'r') as infile:
            infile.readline()
            keys = [name.replace('"', '').replace('\n', '').strip() for name in infile.readline().split(',')][2:]
        timesteps = np.loadtxt(file_path, dtype=np.dtype("datetime64[ms]"), delimiter=',', ndmin=1, usecols=1,
                               skiprows=2)
        float_values = np.loadtxt(file_path, delimiter=',', usecols=range(2, len(keys) + 2), ndmin=2, skiprows=2)
        data = self._transform_csv_data(keys, float_values)
        data["Time Step"] = timesteps
        return data

    @log_error("csv")
    def _load_CPU_data(self) -> Dict[str, np.ndarray]:
        file_path = os.path.join(self.root_path, self.chid + "_cpu.csv")
        if os.path.exists(file_path):
            with open(file_path, 'r') as infile:
                keys = [name.replace('"', '').replace('\n', '').strip() for name in infile.readline().split(',')]
            values = np.loadtxt(file_path, delimiter=',', ndmin=2, skiprows=1)
        else:
            return dict()
        data = self._transform_csv_data(keys, values)
        data["Rank"] = data["Rank"].astype(int)
        return data

    def _transform_csv_data(self, keys, values):
        size = values.shape[0]
        data = {keys[i]: np.empty((size,), dtype=float) for i in range(len(keys))}
        for k, arr in enumerate(data.values()):
            for i in range(size):
                arr[i] = values[i][k]
        return data

    def _indices_to_extent(self, indices: Sequence[Union[int, str]], mesh: Mesh) -> Tuple[
        Extent, Dimension]:
        co = mesh.coordinates

        indices = tuple(int(index) for index in indices)

        x_min, x_max, y_min, y_max, z_min, z_max = indices
        co_x_min, co_x_max, co_y_min, co_y_max, co_z_min, co_z_max = (
            co['x'][x_min], co['x'][x_max], co['y'][y_min],
            co['y'][y_max], co['z'][z_min], co['z'][z_max])
        dimension = Dimension(indices[1] - indices[0] + 1, indices[3] - indices[2] + 1, indices[5] - indices[4] + 1)

        extent = Extent(co_x_min, co_x_max, co_y_min, co_y_max, co_z_min, co_z_max)
        return extent, dimension

    def clear_cache(self, clear_persistent_cache=False):
        """Remove all data from the internal cache that has been loaded so far to free memory.

        :param clear_persistent_cache: Whether to clear the persistent simulation cache as well.
        """
        self.slices.clear_cache()
        self.data_3d.clear_cache()
        self.smoke_3d.clear_cache()
        self.isosurfaces.clear_cache()
        self.obstructions.clear_cache()
        self.devices.clear_cache()
        self.evacs.clear_cache()
        self.particles.clear_cache()

        if clear_persistent_cache:
            os.remove(Simulation._get_pickle_filename(self.root_path, self.chid))
