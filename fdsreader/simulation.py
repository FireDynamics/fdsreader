import logging
import os
from typing import List, TextIO, Dict, AnyStr, Sequence, Tuple, Union

import numpy as np
import pickle

from fdsreader.bndf import Obstruction, Patch, ObstructionCollection, SubObstruction
from fdsreader.geom import Geometry, GeomBoundary, GeometryCollection
from fdsreader.isof import Isosurface, IsosurfaceCollection
from fdsreader.part import Particle, ParticleCollection
from fdsreader.pl3d import Plot3D, Plot3DCollection
from fdsreader.slcf import Slice, SliceCollection
from fdsreader.utils import Mesh, MeshCollection, Dimension, Surface, Quantity, Ventilation, Extent, log_error
from fdsreader.utils.data import create_hash, get_smv_file, Device
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
    :ivar isosurfaces: All defined isosurfaces combined into a :class:`IsosurfaceCollection`.
    :ivar particles: All defined particles combined into a :class:`ParticleCollection`.
    :ivar devices: List containing all devices defined in this simulation.
    :ivar geoms: List containing all geometries defined in this simulation.
    :ivar geom_data: All geometry data by quantity combined into a :class:`GeometryCollection`.
    :ivar cpu: Dictionary mapping .csv header keys to numpy arrays containing cpu data.
    :ivar hrr: Dictionary mapping .csv header keys to numpy arrays containing hrr data.
    :ivar steps: Dictionary mapping .csv header keys to numpy arrays containing steps data.
    """

    _loading = False

    def __new__(cls, path: str):
        if settings.ENABLE_CACHING:
            smv_file_path = get_smv_file(path)
            root_path = os.path.dirname(smv_file_path)

            with open(smv_file_path, 'r') as infile:
                for line in infile:
                    if line.strip() == "CHID":
                        chid = infile.readline().strip()
                        break

            pickle_file_path = Simulation._get_pickle_filename(root_path, chid)

            if not Simulation._loading and os.path.isfile(pickle_file_path):
                Simulation._loading = True
                try:
                    with open(pickle_file_path, 'rb') as f:
                        sim = pickle.load(f)
                    # Check if pickle file stores a Simulation
                    assert isinstance(sim, cls)
                    # Check if the fdsreader version still matches
                    assert sim.reader_version == __version__
                    # Check if the smv_file did not change
                    assert sim._hash == create_hash(smv_file_path)

                    # Return cached sim if it turned out to be valid
                    return sim
                except Exception as e:
                    if settings.DEBUG:
                        logging.exception(e)
                    os.remove(pickle_file_path)

        return super(Simulation, cls).__new__(cls)

    def __getnewargs__(self):
        return self.smv_file_path,

    def __repr__(self):
        r = f"Simulation(chid={self.chid},\n" + \
            f"           meshes={len(self.meshes)},\n" + \
            (f"           obstructions={len(self.obstructions)},\n" if len(self.obstructions) > 0 else "") + \
            (f"           geometries={len(self.geoms)},\n" if len(self.geoms) > 0 else "") + \
            (f"           slices={len(self.slices)},\n" if len(self.slices) > 0 else "") + \
            (f"           plot3d={len(self.data_3d)},\n" if len(self.data_3d) > 0 else "") + \
            (f"           isosurfaces={len(self.isosurfaces)},\n" if len(self.isosurfaces) > 0 else "") + \
            (f"           particles={len(self.particles)},\n" if len(self.particles) > 0 else "") + \
            (f"           devices={len(self.devices)},\n" if len(self.devices) > 0 else "")
        return r[:-2] + ')'

    def __init__(self, path: str):
        """
        :param path: Either the path to the directory containing the simulation data or direct path
            to the .smv file for the simulation in case that multiple simulation output was written to
            the same directory.
        """
        # Check if the file has already been instantiated via a cached pickle file
        if not hasattr(self, "_hash"):
            self.reader_version = __version__

            self.smv_file_path = get_smv_file(path)

            self.root_path = os.path.dirname(self.smv_file_path)

            self.geoms: List[Geometry] = list()
            self.surfaces: List[Surface] = list()
            self.obstructions = list()
            self.ventilations = dict()

            # Will only be used during the loading process to map boundary data to the correct obstruction
            self._subobstructions: Dict[Mesh, List[SubObstruction]] = dict()

            # First collect all meta-information for any FDS data to later combine the gathered
            # information into data collections
            self.slices = dict()
            self.data_3d = dict()
            self.isosurfaces = dict()
            self.particles = list()
            self.geom_data = list()
            self.meshes = list()
            self.devices: Dict[str, Union[Device, List[Device]]] = dict()
            device_tmp = str()

            with open(self.smv_file_path, 'r') as smv_file:
                for line in smv_file:
                    keyword = line.strip()
                    if keyword == "VERSION":
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
                            device_tmp = file_path
                            self.devices["Time"] = Device(Quantity("Time", "Time", ""),
                                                          (.0, .0, .0), (.0, .0, .0))
                    elif keyword == "HRRPUVCUT":
                        self.hrrpuv_cutoff = float(smv_file.readline().strip())
                    elif keyword == "TOFFSET":
                        offsets = smv_file.readline().strip().split()
                        self.default_texture_origin = tuple(float(offsets[i]) for i in range(3))
                    elif keyword == "CLASS_OF_PARTICLES":
                        self.particles.append(self._register_particle(smv_file))
                    elif keyword.startswith("GEOM"):
                        self._load_geoms(smv_file, keyword)
                    elif keyword.startswith("GRID"):
                        self.meshes.append(self._load_mesh(smv_file, keyword))
                    elif keyword == "SURFACE":
                        self.surfaces.append(self._load_surface(smv_file))
                    elif keyword == "DEVICE":
                        name, device = self._register_device(smv_file)
                        if name in self.devices:
                            if type(self.devices) == list:
                                self.devices[name].append(device)
                            else:
                                self.devices[name] = [self.devices[name], device]
                        else:
                            self.devices[name] = device
                    elif keyword.startswith("SLC"):
                        self._load_slice(smv_file, keyword)
                    elif "ISOG" in keyword:
                        self._load_isosurface(smv_file, keyword)
                    elif keyword.startswith("PL3D"):
                        self._load_data_3d(smv_file, keyword)
                    elif keyword.startswith("BNDF"):
                        self._load_boundary_data(smv_file, keyword, cell_centered=False)
                    elif keyword.startswith("BNDC"):
                        self._load_boundary_data(smv_file, keyword, cell_centered=True)
                    elif keyword.startswith("BNDE"):
                        self._load_boundary_data_geom(smv_file, keyword)
                    elif keyword.startswith("PRT5"):
                        self._load_particle_data(smv_file, keyword)
                    elif keyword.startswith("SHOW_OBST"):
                        self._toggle_obst(smv_file, keyword)
                    elif keyword.startswith("HIDE_OBST"):
                        self._toggle_obst(smv_file, keyword)

                self.cpu = self._load_CPU_data()
                if device_tmp != "":
                    self._load_DEVC_data(device_tmp)

            # POST INIT (post read)
            self.out_file_path = os.path.join(self.root_path, self.chid + ".out")
            self.ventilations: List[Ventilation] = list(self.ventilations.values())
            self.obstructions = ObstructionCollection(self.obstructions)

            # Combine the gathered temporary information into data collections
            self.geom_data = GeometryCollection(self.geom_data)
            self.slices = SliceCollection(
                Slice(self.root_path, slice_data[0]["id"], slice_data[0]["cell_centered"],
                      slice_data[0]["times"], slice_data[1:]) for slice_data in
                self.slices.values())
            self.data_3d = Plot3DCollection(self.data_3d.keys(), self.data_3d.values())
            self.isosurfaces = IsosurfaceCollection(self.isosurfaces.values())
            if self.particles is None:
                self.particles = ParticleCollection((), ())
            self.meshes = MeshCollection(self.meshes)

            if settings.ENABLE_CACHING:
                # Hash will be saved to simulation pickle file and compared to new hash when loading
                # the pickled simulation again in the next run of the prpogram.
                self._hash = create_hash(self.smv_file_path)
                pickle.dump(self,
                            open(Simulation._get_pickle_filename(self.root_path, self.chid), 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _get_pickle_filename(cls, root_path: str, chid: str) -> AnyStr:
        """Get the filename used to save the pickled simulation.
        """
        return os.path.join(root_path, chid + ".pickle")

    def _load_mesh(self, smv_file: TextIO, line: str) -> Mesh:
        """Load information for a single mesh from the smv file at current pointer position.
        """
        mesh_id = line.split()[1]

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

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "VENT"
        self._load_vents(smv_file, mesh)

        return mesh

    @log_error("obst")
    def _load_obstructions(self, smv_file: TextIO, mesh: Mesh):
        temp_data = list()
        n = int(smv_file.readline().strip())

        if n > 0:
            self._subobstructions[mesh] = list()

        for _ in range(n):
            line = smv_file.readline().strip().split()
            ext = [float(line[i]) for i in range(6)]
            # The ordinal is negative if the obstruction was created due to a hole. We will ignore that for now
            obst_ordinal = abs(int(line[6]))

            side_surfaces = tuple(self.surfaces[int(line[i]) - 1] for i in range(7, 13))
            if len(line) > 13:
                texture_origin = (float(line[13]), float(line[14]), float(line[15]))
            else:
                texture_origin = self.default_texture_origin
            temp_data.append((obst_ordinal, Extent(*ext), side_surfaces, texture_origin))

        for tmp in temp_data:
            obst_ordinal, extent, side_surfaces, texture_origin = tmp

            line = smv_file.readline().strip().split()
            bound_indices = (
                int(float(line[0])) - 1, int(float(line[1])) - 1, int(float(line[2])) - 1,
                int(float(line[3])) - 1, int(float(line[4])) - 1, int(float(line[5])) - 1)
            color_index = int(line[6])
            block_type = int(line[7])
            rgba = tuple(float(line[i]) for i in range(8, 12)) if color_index == -3 else ()

            obst = next((o for o in self.obstructions if obst_ordinal == o.id), None)
            if obst is None:
                obst = Obstruction(obst_ordinal, color_index, block_type, texture_origin, rgba)
                self.obstructions.append(obst)

            if not any(obst_ordinal == o.id for o in mesh.obstructions):
                mesh.obstructions.append(obst)

            subobst = SubObstruction(side_surfaces, bound_indices, extent)

            self._subobstructions[mesh].append(subobst)
            obst._subobstructions.append(subobst)

    def _toggle_obst(self, smv_file: TextIO, line: str):
        line = line.split()
        mesh = self.meshes[int(line[-1]) - 1]

        obst_index, time = smv_file.readline().split()
        time = float(time)
        subobst = self._subobstructions[mesh][int(obst_index) - 1]

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
        mesh = self.meshes[mesh_index]

        # Read in index ranges for x, y and z
        bound_indices = [int(i.strip()) for i in line.split('&')[1].split('!')[0].strip().split()]
        extent, dimension = self._indices_to_extent(bound_indices, mesh)

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
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

        if slice_index not in self.slices:
            self.slices[slice_index] = [
                {"cell_centered": cell_centered, "times": times, "id": slice_id}]
        self.slices[slice_index].append(
            {"dimension": dimension, "extent": extent, "mesh": mesh, "filename": filename,
             "quantity": quantity, "label": label, "unit": unit})

    @log_error("bndf")
    def _load_boundary_data(self, smv_file: TextIO, line: str, cell_centered: bool):
        """Loads the boundary data at current pointer position.
        """
        line = line.split()
        mesh_index = int(line[1]) - 1
        mesh = self.meshes[mesh_index]

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        bid = int(filename.split('_')[-1][:-3]) - 1

        file_path = os.path.join(self.root_path, filename)

        patches = dict()

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
                          patch_offset, offset, n_t)

                # Skip obstacles with index 0, which just gives the extent of the (whole) mesh faces
                # These might be needed in case of "closed" mesh faces
                if obst_index != 0:
                    obst_index -= 1  # Account for fortran indexing
                    if obst_index not in patches:
                        patches[obst_index] = list()
                    patches[obst_index].append(p)
                patch_offset += fdtype.new(
                    (('f', str(p.dimension.shape(cell_centered=False))),)).itemsize

        for obst_index, p in patches.items():
            for patch in p:
                patch._post_init(patch_offset)

            self._subobstructions[mesh][obst_index]._add_patches(bid, cell_centered, quantity, label, unit, p, times,
                                                                 n_t, lower_bounds, upper_bounds)

    @log_error("geom")
    def _load_boundary_data_geom(self, smv_file: TextIO, line: str):
        line = line.split()
        mesh_index = int(line[1]) - 1
        # Meshes are not loaded yet
        # mesh = self.meshes[mesh_index]

        filename_be = smv_file.readline().strip()
        filename_gbf = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
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

        if bid >= len(self.geom_data):
            self.geom_data.append(GeomBoundary(Quantity(quantity, label, unit), times, n_t))
        self.geom_data[bid]._add_data(mesh_index, file_path_be, file_path_gbf, lower_bounds,
                                      upper_bounds)

    @log_error("pl3d")
    def _load_data_3d(self, smv_file: TextIO, line: str):
        """Loads the pl3d at current pointer position.
        """
        line = line.strip().split()

        time = float(line[1])

        mesh_index = int(line[2]) - 1

        filename = smv_file.readline().strip()
        quantities = list()
        for _ in range(5):
            quantity = smv_file.readline().strip()
            label = smv_file.readline().strip()
            unit = smv_file.readline().strip()
            quantities.append(Quantity(quantity, label, unit))

        if time not in self.data_3d:
            self.data_3d[time] = Plot3D(self.root_path, time, quantities)
        self.data_3d[time]._add_subplot(filename, self.meshes[mesh_index])

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
        label = smv_file.readline().strip()
        unit = smv_file.readline().strip()
        if double_quantity:
            v_quantity = smv_file.readline().strip()
            v_label = smv_file.readline().strip()
            v_unit = smv_file.readline().strip()

        if iso_id not in self.isosurfaces:
            with open(iso_file_path, 'rb') as infile:
                nlevels = fdtype.read(infile, fdtype.INT, 3)[2][0][0]

                dtype_header_levels = fdtype.new((('f', nlevels),))
                levels = fdtype.read(infile, dtype_header_levels, 1)[0]
        if double_quantity:
            if iso_id not in self.isosurfaces:
                self.isosurfaces[iso_id] = Isosurface(iso_id, double_quantity, quantity, label,
                                                      unit, levels, v_quantity=v_quantity,
                                                      v_label=v_label, v_unit=v_unit)
            self.isosurfaces[iso_id]._add_subsurface(self.meshes[mesh_index], iso_file_path,
                                                     viso_file_path=viso_file_path)
        else:
            if iso_id not in self.isosurfaces:
                self.isosurfaces[iso_id] = Isosurface(iso_id, double_quantity, quantity, label,
                                                      unit, levels)
            self.isosurfaces[iso_id]._add_subsurface(self.meshes[mesh_index], iso_file_path)

    @log_error("part")
    def _register_particle(self, smv_file: TextIO) -> Particle:
        particle_class = smv_file.readline().strip()
        color = tuple(float(c) for c in smv_file.readline().strip().split())

        n_q = int(smv_file.readline().strip())
        quantities = list()
        for _ in range(n_q):
            quantity = smv_file.readline().strip()
            label = smv_file.readline().strip()
            unit = smv_file.readline().strip()
            quantities.append(Quantity(quantity, label, unit))
        return Particle(particle_class, quantities, color)

    def _load_particle_meta(self, particles: Union[List[Particle], ParticleCollection],
                            file_path: str, mesh: Mesh) -> List[float]:
        with open(file_path, 'r') as bnd_file:
            line = bnd_file.readline().strip().split()
            n_classes = int(line[1])
            times = list()
            n_q = list()
            for i in range(n_classes):
                line = bnd_file.readline().strip().split()
                n_q.append(int(line[0]))
                for _ in range(n_q[-1]):
                    bnd_file.readline()
                particles[i].n_particles[mesh] = list()
            bnd_file.seek(0)

            for line in bnd_file:
                times.append(float(line.strip().split()[0]))
                for i in range(n_classes):
                    particle = particles[i]
                    particle.n_particles[mesh].append(
                        int(bnd_file.readline().strip().split()[1].strip()))
                    for q in range(n_q[i]):
                        line = bnd_file.readline().strip().split()
                        quantity = particle.quantities[q].quantity
                        particle.lower_bounds[quantity].append(float(line[0]))
                        particle.upper_bounds[quantity].append(float(line[1]))
        return times

    @log_error("part")
    def _load_particle_data(self, smv_file: TextIO, line: str):
        file_path = os.path.join(self.root_path, smv_file.readline().strip())

        mesh_index = int(line.split()[1].strip()) - 1
        mesh = self.meshes[mesh_index]

        times = self._load_particle_meta(self.particles, file_path + '.bnd', mesh)
        if type(self.particles) == list:
            self.particles = ParticleCollection(times, self.particles)

        self.particles._file_paths[mesh] = file_path

        n_classes = int(smv_file.readline().strip())
        for i in range(n_classes):
            smv_file.readline()

    @log_error("devc")
    def _register_device(self, smv_file: TextIO) -> Tuple[str, Device]:
        line = smv_file.readline().strip().split('%')
        name = line[0].strip()
        quantity_label = line[1].strip()
        line = smv_file.readline().strip().split('#')[0].split()
        position = (float(line[0]), float(line[1]), float(line[2]))
        orientation = (float(line[3]), float(line[4]), float(line[5]))
        return name, Device(Quantity(name, quantity_label, ""), position, orientation)

    @log_error("devc")
    def _load_DEVC_data(self, file_path: str):
        with open(file_path, 'r') as infile:
            units = infile.readline().split(',')
            names = [name.replace('"', '').replace('\n', '').strip() for name in
                     infile.readline().split(',')]
            values = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
            for k in range(len(names)):
                devc = self.devices[names[k]]
                devc.quantity.unit = units[k]
                size = values.shape[0]
                devc.data = np.empty((size,), dtype=np.float32)
                for i in range(size):
                    devc.data[i] = values[i][k]

        line_path = file_path.replace("devc", "steps")
        if os.path.exists(line_path):
            with open(line_path, 'r') as infile:
                infile.readline()
                names = infile.readline().split(',')
                data = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
                for k, key in enumerate(names):
                    if key in self.devices:
                        devc = self.devices[key]
                        for i in range(len(devc)):
                            devc[i] = data[k, i]

    @log_error("csv")
    def _load_HRR_data(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path, 'r') as infile:
            infile.readline()
            keys = infile.readline().split(',')
            values = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
            return self._transform_csv_data(keys, values, [np.float32] * len(keys))

    @log_error("csv")
    def _load_step_data(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path, 'r') as infile:
            infile.readline()
            keys = infile.readline().split(',')
            dtypes = [np.float32] * len(keys)
            dtypes[0] = int
            dtypes[1] = np.dtype("datetime64[ms]")
            values = np.genfromtxt(infile, delimiter=',', dtype=dtypes, autostrip=True)
            return self._transform_csv_data(keys, values, dtypes)

    @log_error("csv")
    def _load_CPU_data(self) -> Dict[str, np.ndarray]:
        file_path = os.path.join(self.root_path, self.chid + "_cpu.csv")
        if os.path.exists(file_path):
            with open(file_path, 'r') as infile:
                keys = infile.readline().split(",")
                dtypes = [np.float32] * len(keys)
                dtypes[0] = int
                values = np.genfromtxt(infile, delimiter=',', dtype=dtypes, autostrip=True)
                if len(values.shape) != 0:
                    return self._transform_csv_data(keys, values, dtypes)
                else:
                    return self._transform_csv_data(keys, values.reshape((1,)), dtypes)

    def _transform_csv_data(self, keys, values, dtypes):
        size = values.shape[0]
        data = {keys[i]: np.empty((size,), dtype=dtypes[i]) for i in range(len(keys))}
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
        self.isosurfaces.clear_cache()
        self.particles.clear_cache()
        self.obstructions.clear_cache()

        if clear_persistent_cache:
            os.remove(Simulation._get_pickle_filename(self.root_path, self.chid))
