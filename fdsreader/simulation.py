import os
from typing import List, TextIO, Dict, AnyStr

import numpy as np
import logging
import pickle

from fdsreader.bndf import Boundary
from fdsreader.isof import Isosurface
from fdsreader.isof.IsosurfaceCollection import IsosurfaceCollection
from fdsreader.plot3d import Plot3D
from fdsreader.plot3d.Plot3dCollection import Plot3DCollection
from fdsreader.slcf import Slice
from fdsreader.slcf.SliceCollection import SliceCollection
from fdsreader.utils import Mesh, Dimension, Surface, Quantity, Obstruction, Ventilation, Extent
from fdsreader.utils.data import create_hash, get_smv_file
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings
from fdsreader.utils.fds_classes.obstruction import Patch


class Simulation:
    """Master class managing all data for a given simulation.

    :ivar smv_file_path: Path to the .smv file of the simulation.
    :ivar root_path: Path to the root directory of the simulation.
    :ivar fds_version: Version of FDS the simulation was performed with.
    :ivar chid: Name (ID) of the simulation.
    :ivar hrrpuv_cutoff: The hrrpuv_cutoff value.
    :ivar default_texture_origin: The default origin used for textures with no explicit origin.
    :ivar out_file_path: Path to the .out file of the simulation.
    :ivar surfaces: List containing all surfaces defined in this simulation.
    :ivar meshes: List containing all meshes (grids) defined in this simulation.
    """

    def __new__(cls, path: str):
        if settings.ENABLE_CACHING:
            smv_file_path = get_smv_file(path)
            root_path = os.path.dirname(path)

            with open(smv_file_path, 'r') as infile:
                for line in infile:
                    if line.strip() == "CHID":
                        chid = infile.readline().strip()
                        break

            pickle_file_path = Simulation._get_pickle_filename(root_path, chid)

            if path and os.path.isfile(path):
                with open(pickle_file_path) as f:
                    sim = pickle.load(f)
                if not isinstance(sim, cls):
                    os.remove(pickle_file_path)
                else:
                    if sim._hash == create_hash(smv_file_path):
                        return sim

        return super(Simulation, cls).__new__(cls)

    def __init__(self, path: str):
        """
        :param path: Either the path to the directory containing the simulation data or direct path
            to the .smv file for the simulation in case that multiple simulation output was written to
            the same directory.
        """
        # Check if the file has already been instantiated via a cached pickle file
        if not hasattr(self, "_hash"):
            self.smv_file_path = get_smv_file(path)

            self.root_path = os.path.dirname(self.smv_file_path)

            self.meshes: List[Mesh] = list()
            self.surfaces: List[Surface] = list()
            self.obstructions: Dict[int, Obstruction] = dict()
            self.ventilations: Dict[int, Ventilation] = dict()

            # First collect all meta-information for any FDS data to later combine the gathered
            # information into data collections
            self.slices = dict()
            self.data_3d = dict()
            self.isosurfaces = dict()

            with open(self.smv_file_path, 'r') as smv_file:
                for line in smv_file:
                    keyword = line.strip()
                    if keyword == "VERSION":
                        self.fds_version = smv_file.readline().strip()
                    elif keyword == "CHID":
                        self.chid = smv_file.readline().strip()
                    elif keyword == "HRRPUVCUT":
                        self.hrrpuv_cutoff = float(smv_file.readline().strip())
                    elif keyword == "TOFFSET":
                        offsets = smv_file.readline().strip().split()
                        self.default_texture_origin = tuple(float(offsets[i]) for i in range(3))
                    elif "GRID" in keyword:
                        self.meshes.append(self._load_mesh(smv_file, keyword))
                    elif keyword == "SURFACE":
                        self.surfaces.append(self._load_surface(smv_file))
                    elif "SLC" in keyword:
                        self._load_slice(smv_file, keyword)
                    elif "ISOG" in keyword:
                        self._load_isosurface(smv_file, keyword)
                    elif keyword == "PL3D":
                        self._load_data_3d(smv_file, keyword)
                    elif "BND" in keyword:
                        self._load_boundary_data(smv_file, keyword)

            self.out_file_path = os.path.join(self.root_path, self.chid + ".out")

            # Now combine the gathered temporary information into data collections
            self.slices = SliceCollection(
                Slice(self.root_path, slice_data[0]["cell_centered"], slice_data[1:]) for slice_data
                in self.slices.values())
            self.data_3d = Plot3DCollection(self.data_3d.keys(), self.data_3d.values())
            self.isosurfaces = IsosurfaceCollection(self.isosurfaces.values())

            if settings.ENABLE_CACHING:
                # Hash will be saved to simulation pickle file and compared to new hash when loading
                # the pickled simulation again in the next run of the program.
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
        logging.debug("Found MESH with id: %s", mesh_id)

        grid_numbers = smv_file.readline().strip().split()
        grid_dimensions = {'x': int(grid_numbers[0]) + 1, 'y': int(grid_numbers[1]) + 1,
                           'z': int(grid_numbers[2]) + 1}
        logging.debug("Number of cells: %i x %i x %i", *grid_dimensions.values())

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
                coordinates[dim][i] = smv_file.readline().split()[1]

        mesh = Mesh(coordinates, extents, mesh_id)

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "OBST"
        self._load_obstructions(smv_file, mesh)

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "VENT"
        self._load_vents(smv_file, mesh)

        # TODO: Offset und closed/open vents

        return mesh

    def _load_obstructions(self, smv_file: TextIO, mesh: Mesh):
        temp_data = list()
        n = int(smv_file.readline().strip())
        for _ in range(n):
            line = smv_file.readline().strip().split()
            extent = Extent(*[float(line[i]) for i in range(6)])
            obst_id = int(line[6])

            side_surfaces = tuple(self.surfaces[int(line[i]) - 1] for i in range(7, 13))
            if len(line) > 13:
                texture_origin = (float(line[13]), float(line[14]), float(line[15]))
                temp_data.append((obst_id, extent, side_surfaces, texture_origin))
            else:
                temp_data.append((obst_id, extent, side_surfaces))

            for tmp in temp_data:
                line = smv_file.readline().strip().split()
                # Todo: What does bound index mean?
                bound_indices = (int(line[0]), int(line[1]), int(line[2]), int(line[3]),
                                 int(line[4]), int(line[5]))
                color_index = int(line[6])
                block_type = int(line[7])
                if color_index == -3:
                    rgba = tuple(float(line[i]) for i in range(8, 12))
                else:
                    rgba = ()

                if len(tmp) == 4:
                    obst_id, extent, side_surfaces, texture_origin = tmp
                else:
                    obst_id, extent, side_surfaces = tmp
                    texture_origin = self.default_texture_origin

                if obst_id not in self.obstructions:
                    self.obstructions[obst_id] = Obstruction(obst_id, side_surfaces, bound_indices,
                                                             color_index, block_type,
                                                             texture_origin, rgba=rgba)
                self.obstructions[obst_id]._extents[mesh] = extent

    def _load_vents(self, smv_file: TextIO, mesh: Mesh):
        line = smv_file.readline()
        n, n_dummies = int(line[0]), int(line[1])

        temp_data = list()

        def read_common_info():
            line = smv_file.readline().strip().split()
            return line, Extent(*[float(line[i]) for i in range(6)]), int(line[6]), self.surfaces[
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
            line, extent, vent_id, surface = read_common_info()
            texture_origin = (float(line[8]), float(line[9]), float(line[10]))
            temp_data.append((extent, vent_id, surface, texture_origin))

        for _ in range(n_dummies):
            _, extent, vent_id, surface = read_common_info()
            temp_data.append((extent, vent_id, surface))

        for v in range(n):
            if v < n - n_dummies:
                extent, vent_id, surface, texture_origin = temp_data[v]
            else:
                extent, vent_id, surface = temp_data[v]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            if vent_id not in self.obstructions:
                self.ventilations[vent_id] = Ventilation(vent_id, surface, bound_indices,
                                                         color_index, draw_type, rgba=rgba,
                                                         texture_origin=texture_origin)
            self.ventilations[vent_id]._add_subventilation(mesh, extent)

        smv_file.readline()
        assert smv_file.readline() == "CVENT"

        n = int(smv_file.readline().strip())
        temp_data.clear()
        for _ in range(n):
            line, extent, vid, surface = read_common_info()
            circular_vent_origin = (float(line[12]), float(line[13]), float(line[14]))
            radius = float(line[15])
            temp_data.append((extent, vid, surface, texture_origin, circular_vent_origin, radius))

        for v in range(n):
            extent, vent_id, surface, texture_origin, circular_vent_origin, radius = temp_data[v]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            if vent_id not in self.obstructions:
                self.ventilations[vent_id] = Ventilation(vent_id, surface, bound_indices,
                                                         color_index, draw_type, rgba=rgba,
                                                         texture_origin=texture_origin,
                                                         circular_vent_origin=circular_vent_origin,
                                                         radius=radius)
            self.ventilations[vent_id]._add_subventilation(mesh, extent)

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

    def _load_slice(self, smv_file: TextIO, line: str):
        """Loads the slice at current pointer position.
        """
        if "SLCC" in line:
            cell_centered = True
        else:
            cell_centered = False

        slice_id = int(line.split('!')[1].strip().split()[0])

        mesh_index = int(line.split('&')[0].strip().split()[1]) - 1

        # Read in index ranges for x, y and z
        index_ranges = [int(i.strip()) for i in
                        line.split('&')[1].split('!')[0].strip().split()]

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        if slice_id not in self.slices:
            self.slices[slice_id] = [{"cell_centered": cell_centered}]
        self.slices[slice_id].append(
            {"extent": Dimension(*index_ranges), "mesh": self.meshes[mesh_index],
             "filename": filename, "quantity": quantity, "label": label, "unit": unit})

        logging.debug("Found SLICE with id: :i", slice_id)

    def _load_boundary_data(self, smv_file: TextIO, line: str):
        """Loads the boundary data at current pointer position.
        """
        line = line.split()
        if line[0] == 'BNDC':
            cell_centered = True
        else:
            cell_centered = False
        mesh_index = int(line[1]) - 1

        mesh = self.meshes[mesh_index]

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        bid = int(filename.split('_')[-1][:-3])

        file_path = os.path.join(self.root_path, filename)

        patches = dict()
        total_dim_size = fdtype.FLOAT.itemsize

        with open(file_path, 'rb') as infile:
            # Offset of the binary file to the end of the file header.
            offset = 3 * fdtype.new((('c', 30),)).itemsize
            infile.seek(offset)

            n_patches = fdtype.read(infile, fdtype.INT, 1)[0][0][0]
            dtype_patches = fdtype.new((('i', 9),))
            patch_infos = fdtype.read(infile, dtype_patches, n_patches)[0]

            offset += fdtype.INT.itemsize + dtype_patches.itemsize * n_patches

            for patch_info in patch_infos:
                co = mesh.coordinates
                dimension = Dimension(co[0][patch_info[0]], co[0][patch_info[1]],
                                   co[1][patch_info[2]],
                                   co[1][patch_info[3]], co[2][patch_info[4]],
                                   co[2][patch_info[5]])
                orientation = patch_info[6]
                obst_id = patch_info[7] + 1
                if obst_id not in patches:
                    patches[obst_id] = list()
                p = Patch(file_path, dimension, orientation, cell_centered, total_dim_size)
                patches[obst_id].append(p)
                total_dim_size += fdtype.new((('f', str(p.shape)),)).itemsize

            t_n = (os.stat(file_path).st_size - offset) // total_dim_size

            dtype_time = np.dtype(settings.FORTRAN_DATA_TYPE_FLOAT)
            times = np.empty((t_n,), dtype=dtype_time)
            for t in range(t_n):
                infile.seek(offset + t * total_dim_size)
                times[t] = fdtype.read(infile, fdtype.FLOAT, 1)[0][0][0]

        for obst_id, patches in patches.items():
            for patch in patches:
                patch._post_init(t_n, total_dim_size)
            self.obstructions[obst_id]._add_patches(bid, cell_centered, quantity, label, unit, mesh, patches, times, t_n)

    def _load_data_3d(self, smv_file: TextIO, line: str):
        """Loads the plot3d at current pointer position.
        """
        # Todo: Also read SMOKG3D data?
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

    def _load_isosurface(self, smv_file: TextIO, line: str):
        """Loads the isosurface at current pointer position.
        """
        double_quantity = line[0] == 'T'
        mesh_index = int(line.strip().split()[1]) - 1

        iso_filename = smv_file.readline().strip()
        iso_id = int(iso_filename.split('_')[-1][:-4])

        if double_quantity:
            viso_filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
        unit = smv_file.readline().strip()
        if double_quantity:
            v_quantity = smv_file.readline().strip()
            v_label = smv_file.readline().strip()
            v_unit = smv_file.readline().strip()

        if double_quantity:
            if iso_id not in self.isosurfaces:
                self.isosurfaces[iso_id] = Isosurface(iso_id,
                                                 os.path.dirname(self.smv_file_path),
                                                 double_quantity, quantity, label, unit,
                                                 v_quantity=v_quantity, v_label=v_label,
                                                 v_unit=v_unit)
            self.isosurfaces[iso_id]._add_subsurface(self.meshes[mesh_index], iso_filename,
                                                viso_filename=viso_filename)
        else:
            if iso_id not in self.isosurfaces:
                self.isosurfaces[iso_id] = Isosurface(iso_id,
                                                 os.path.dirname(self.smv_file_path),
                                                 double_quantity, quantity, label, unit)
            self.isosurfaces[iso_id]._add_subsurface(self.meshes[mesh_index], iso_filename)

    def clear_cache(self, clear_persistent_cache=False):
        """Remove all data from the internal cache that has been loaded so far to free memory.

        :param clear_persistent_cache: Whether to clear the persistent simulation cache as well.
        """
        if hasattr(self, "_slices"):
            self.slices.clear_cache()
        if hasattr(self, "_boundaries"):
            self._boundaries.clear_cache()  # Todo
        if hasattr(self, "_plot3ds"):
            self.data_3d.clear_cache()
        if hasattr(self, "_isosurfaces"):
            self.isosurfaces.clear_cache()

        if clear_persistent_cache:
            os.remove(Simulation._get_pickle_filename(self.root_path, self.chid))
