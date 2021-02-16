import operator
import os
from functools import reduce
from typing import BinaryIO, Dict, Union, List, Tuple

import numpy as np
from pyvista import PolyData

from fdsreader.utils import Quantity, Mesh
from fdsreader import settings
import fdsreader.utils.fortran_data as fdtype


class SubSurface:
    """Part of an isosurface with data for a specific mesh.

    :ivar mesh: The mesh containing all data for this :class:`SubSurface`.
    :ivar file_path: Path to the binary data file.
    :ivar v_file_path: Path to the binary data file containing color data.
    :ivar n_vertices: The number of vertices for this subsurface.
    :ivar n_triangles: The number of triangles for this subsurface.
    :ivar n_t: Total number of time steps for which output data has been written.
    """

    def __init__(self, mesh: Mesh, iso_filepath: str, times: List, viso_filepath: str = ""):
        self.mesh = mesh

        self.file_path = iso_filepath
        if viso_filepath != "":
            self.v_file_path = viso_filepath

        with open(self.file_path, 'rb') as infile:
            nlevels = fdtype.read(infile, fdtype.INT, 3)[2][0][0]

            dtype_header_levels = fdtype.new((('f', nlevels),))
            self.levels = fdtype.read(infile, dtype_header_levels, 1)[0]

            dtype_header_zeros = fdtype.combine(fdtype.INT, fdtype.new((('i', 2),)))
            self._offset = fdtype.INT.itemsize * 3 + dtype_header_levels.itemsize + \
                           dtype_header_zeros.itemsize

            self.times = times
            self.n_vertices = list()
            self.n_triangles = list()

            if not settings.LAZY_LOAD:
                self._load_data(infile)

        if self.has_color_data:
            if not settings.LAZY_LOAD:
                with open(self.v_file_path, 'rb') as infile:
                    self._load_vdata(infile)

    @property
    def vertices(self):
        """Property to lazy load all vertices for all triangles of any level.
        """
        if not hasattr(self, "_vertices"):
            with open(self.file_path, 'rb') as infile:
                self._load_data(infile)
        return self._vertices

    @property
    def triangles(self):
        """Property to lazy load all triangles of any level.
        """
        if not hasattr(self, "_triangles"):
            with open(self.file_path, 'rb') as infile:
                self._load_data(infile)
        return self._triangles

    @property
    def surfaces(self):
        """Property to lazy load a list that maps triangles to an isosurface for a specific level.
        The list has the size n_triangles, while the indices correspond to indices of the triangles.
        """
        if not hasattr(self, "_surfaces"):
            with open(self.file_path, 'rb') as infile:
                self._load_data(infile)
        return self._surfaces

    @property
    def has_color_data(self):
        """Defines whether there is color data for this subsurface or not.
        """
        return hasattr(self, "v_file_path")

    @property
    def colors(self):
        """Property to lazy load the color data that might be associated with the isosurfaces.
        """
        if self.has_color_data:
            if not hasattr(self, "_colors"):
                with open(self.v_file_path, 'rb') as infile:
                    self._load_vdata(infile)
            return self._colors
        else:
            raise UserWarning("The isosurface does not have any associated color-data. Use the"
                              " attribute 'has_color_data' to check if an isosurface has associated"
                              " color-data.")

    def _load_data(self, infile: BinaryIO):
        """Loads data for the subsurface which is given in an iso file.
        """
        dtype_time = fdtype.new((('f', 1), ('i', 1)))
        dtype_dims = fdtype.new((('i', 2),))

        self._vertices = list()
        self._triangles = list()
        self._surfaces = list()

        infile.seek(self._offset)
        time_data = fdtype.read(infile, dtype_time, 1)

        while time_data.size != 0:
            self.times.append(time_data[0][0][0])

            dims_data = fdtype.read(infile, dtype_dims, 1)
            n_vertices = dims_data[0][0][0]
            n_triangles = dims_data[0][0][1]

            if n_vertices > 0:
                dtype_vertices = fdtype.new((('f', 3 * n_vertices),))
                dtype_triangles = fdtype.new((('i', 3 * n_triangles),))
                dtype_surfaces = fdtype.new((('i', n_triangles),))

                self._vertices.append(
                    fdtype.read(infile, dtype_vertices, 1)[0][0].reshape((n_vertices, 3)).astype(
                        float))
                self._triangles.append(
                    fdtype.read(infile, dtype_triangles, 1)[0][0].reshape((n_triangles, 3)).astype(
                        int) - 1)
                self._surfaces.append(fdtype.read(infile, dtype_surfaces, 1)[0][0].astype(int) - 1)
                self.n_vertices.append(n_vertices)
                self.n_triangles.append(n_triangles)
            else:
                self._vertices.append(np.empty((0, 3)))
                self._triangles.append(np.empty((0, 3)))
                self._surfaces.append(np.empty((0,)))
                self.n_vertices.append(0)
                self.n_triangles.append(0)

            time_data = fdtype.read(infile, dtype_time, 1)

        self.n_t = len(self.times)

    def _load_vdata(self, infile: BinaryIO):
        """Loads all color data for all isosurfaces in a given viso file.
        """
        self._colors = np.empty((self.n_t,), dtype=object)
        t_offset = fdtype.FLOAT.itemsize
        dtype_nverts = fdtype.new((('i', 4),)).itemsize

        infile.seek(fdtype.INT.itemsize * 2)
        for t in range(self.n_t):
            infile.seek(t_offset, os.SEEK_CUR)
            n_vertices = fdtype.read(infile, dtype_nverts, 1)[0][0][2]
            self._colors[t] = fdtype.read(infile, fdtype.new((('f', n_vertices),)), 1)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "times"):
            del self.times
        if hasattr(self, "_vertices"):
            del self._vertices
        if hasattr(self, "_triangles"):
            del self._triangles
        if hasattr(self, "_surfaces"):
            del self._surfaces
        if hasattr(self, "_colors"):
            del self._colors


class Isosurface:
    """Isosurface file data container including metadata. Consists of a list of vertices forming a
        list of triangles. Can optionally have additional color data for the surfaces.

    :ivar id: The ID of this isosurface.
    :ivar root_path: Path to the directory containing all isosurface files.
    :ivar quantity: Quantity object containing information about the quantity calculated for this
        isosurface with the corresponding label and unit.
    :ivar v_quantity: Information about the color quantity.
    :ivar levels: All isosurface levels.
    """

    def __init__(self, isosurface_id: int, root_path: str, double_quantity: bool, quantity: str,
                 label: str, unit: str, v_quantity: str = "", v_label: str = "", v_unit: str = ""):
        self.id = isosurface_id
        self.root_path = root_path
        self.quantity = Quantity(quantity, label, unit)
        self._double_quantity = double_quantity

        self._times = list()

        self._subsurfaces: Dict[Mesh, SubSurface] = dict()

        if self._double_quantity:
            self.v_quantity = Quantity(v_quantity, v_label, v_unit)

    def _add_subsurface(self, mesh: Mesh, iso_filename: str, viso_filename: str = "") -> SubSurface:
        if viso_filename != "":
            subsurface = SubSurface(mesh, os.path.join(self.root_path, iso_filename), self._times,
                                    os.path.join(self.root_path, viso_filename))
        else:
            subsurface = SubSurface(mesh, os.path.join(self.root_path, iso_filename), self._times)
        self._subsurfaces[mesh] = subsurface

        return subsurface

    def __getitem__(self, mesh: Mesh) -> SubSurface:
        """Returns the :class:`SubSurface` that contains data for the given mesh.
        """
        return self._subsurfaces[mesh]

    @property
    def vertices(self) -> Dict[Mesh, List[np.ndarray]]:
        """Gets all vertices per mesh.
        """
        return {mesh: subsurface.vertices for mesh, subsurface in self._subsurfaces.items()}

    @property
    def triangles(self) -> Dict[Mesh, List[np.ndarray]]:
        """Gets all triangles per mesh.
        """
        return {mesh: subsurface.triangles for mesh, subsurface in self._subsurfaces.items()}

    @property
    def surfaces(self) -> Dict[Mesh, List[np.ndarray]]:
        """Gets all surfaces per mesh.
        """
        return {mesh: subsurface.surfaces for mesh, subsurface in self._subsurfaces.items()}

    def to_global(self, time: Union[int, float]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Creates an array containing all global vertices and a list containing numpy arrays with
            triangles for each surface level.

        :param time: Either the index of the timestep or an actual time value. In the latter case
            data for the nearest matching timestep will be used.
        """
        if type(time) == float:
            time = self.get_nearest_timestep(time)

        if time > len(self.times):
            time = len(self.times) - 1
        if time < 0:
            time = 0

        n_vertices = sum(
            x[time].shape[0] if len(x[time].shape) > 0 else 0 for x in self.vertices.values())
        vertices = np.empty((n_vertices, 3))
        verts_counter = 0
        for mesh in self._subsurfaces.keys():
            tmp = verts_counter
            verts = self.vertices[mesh][time]
            verts_counter += verts.shape[0]
            vertices[tmp:verts_counter] = verts + tmp

        triangles = list()
        num_levels = max(
            max(np.max(t) if t.shape[0] != 0 else 0 for t in s) for s in self.surfaces.values()) + 1
        for surf in range(num_levels):
            n_triangles = sum(np.count_nonzero(s[time] == surf) for s in self.surfaces.values())
            triangles.append(np.empty((n_triangles, 3), dtype=int))
            tris_counter = 0
            if n_triangles > 0:
                for mesh in self._subsurfaces.keys():
                    tmp = tris_counter
                    tris = self.triangles[mesh][time][self.surfaces[mesh][time] == surf]
                    tris_counter += tris.shape[0]
                    triangles[surf][tmp:tris_counter] = tris + tmp

        return vertices, triangles

    def get_pyvista_mesh(self, vertices: np.ndarray, triangles: np.ndarray) -> PolyData:
        """Creates a PyVista mesh from the data.
        """
        triangles = np.hstack(np.append(np.full((triangles.shape[0], 1), 3), triangles, axis=1))
        return PolyData(vertices, triangles)

    def join_pyvista_meshes(self, meshes: List[PolyData]) -> PolyData:
        """Combines multiple PyVista meshes.
        """
        return reduce(operator.add, meshes)

    def export(self, file_path: str, mesh: PolyData):
        """Export the isosurface for a single timestep into one of many formats.

        :param file_path: Absolute path to the the file which should we written. The file ending
            denotes the file format to export to (supports pretty much every common format).
        :param time: Either the index of the timestep or an actual time value. In the latter case
            data for the nearest matching timestep will be used.
        """
        if "vtk" in file_path or "vtp" in file_path:
            return mesh.save(file_path)
        else:
            return mesh.save_meshio(file_path)

    def get_nearest_timestep(self, time: float) -> int:
        """Calculates the nearest timestep for which data has been output.
        """
        idx = np.searchsorted(self.times, time, side="left")
        if time > 0 and (idx == len(self.times) or np.math.fabs(
                time - self.times[idx - 1]) < np.math.fabs(time - self.times[idx])):
            return idx - 1
        else:
            return idx

    @property
    def has_color_data(self) -> bool:
        """Defines whether there is color data for this isosurface or not.
        """
        return self._double_quantity

    @property
    def times(self) -> List[float]:
        """List containing all times for which data has been recorded.
        """
        if len(self._times) == 0:
            # Implicitly load the data for one subsurface and read times
            _ = next(iter(self._subsurfaces.values())).vertices
        return self._times

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subsurface in self._subsurfaces.values():
            subsurface.clear_cache()
