import os
from typing import BinaryIO, List

import numpy as np

from fdsreader.utils import Quantity, settings, Mesh
import fdsreader.utils.fortran_data as fdtype


class SubSurface:
    """Part of an isosurface with data for a specific mesh.

    :ivar mesh: The mesh containing all data for this :class:`SubSurface`.
    :ivar file_path: Path to the binary data file.
    :ivar v_file_path: Path to the binary data file containing color data.
    :ivar n_vertices: The number of vertices for this subsurface.
    :ivar n_triangles: The number of triangles for this subsurface.
    :ivar t_n: Total number of time steps for which output data has been written.
    :ivar _offset: Offset of the binary file to the end of the file header.
    :cvar _v_offset: Offset of the binary file containing color data to the end of the file header.
    """
    _v_offset = fdtype.INT.itemsize * 2 + fdtype.FLOAT.itemsize + fdtype.new(
        (('i', 4),)).itemsize

    def __init__(self, mesh: Mesh, iso_filepath: str, viso_filepath: str = ""):
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

            self.times = list()
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

        vertices = list()
        triangles = list()
        surfaces = list()

        infile.seek(self._offset)
        time_data = fdtype.read(infile, dtype_time, 1)

        while time_data.size != 0:
            self.times.append(time_data[0][0][0])

            dims_data = fdtype.read(infile, dtype_dims, 1)
            n_vertices = dims_data[0][0][0]
            n_triangles = dims_data[0][0][1]

            dtype_vertices = fdtype.new((('f', 3 * n_vertices),))
            dtype_triangles = fdtype.new((('i', 3 * n_triangles),))
            dtype_surfaces = fdtype.new((('i', n_triangles),))

            vertices.append(fdtype.read(infile, dtype_vertices, 1)[0][0].reshape((n_vertices, 3)))
            triangles.append(
                fdtype.read(infile, dtype_triangles, 1)[0][0].reshape((n_triangles, 3)))
            surfaces.append(fdtype.read(infile, dtype_surfaces, 1)[0][0])

            self.n_vertices.append(n_vertices)
            self.n_triangles.append(n_triangles)
            time_data = fdtype.read(infile, dtype_time, 1)

        self._vertices = np.array(vertices)
        self._triangles = np.array(triangles)
        self._surfaces = np.array(surfaces)

        self.t_n = len(self.times)

    def _load_vdata(self, infile: BinaryIO):
        """Loads all color data for all isosurfaces in a given viso file.
        """
        # Todo: Fix this
        # dtype_color = fdtype.new((('f', self.n_vertices),))
        # infile.seek(self._v_offset)
        # self._colors = fdtype.read(infile, dtype_color, self.n_vertices)


class Isosurface:
    """Isosurface file data container including metadata. Consists of a list of vertices forming a list
         of triangles. Can optionally have additional color data for the surfaces.

    :ivar id: The ID of this isosurface.
    :ivar root_path: Path to the directory containing all isosurface files.
    :ivar quantity: Information about the quantity.
    :ivar v_quantity: Information about the color quantity.
    :ivar levels: All isosurface levels
    :ivar _double_quantity: Defines whether there is color data for this isosurface or not.
    """

    def __init__(self, isosurface_id: int, root_path: str, double_quantity: bool, quantity: str,
                 label: str, unit: str, v_quantity: str = "", v_label: str = "", v_unit: str = ""):
        self.id = isosurface_id
        self.root_path = root_path
        self.quantity = Quantity(quantity, label, unit)
        self._double_quantity = double_quantity

        self._times = None

        self._subsurfaces: List[SubSurface] = list()

        if self._double_quantity:
            self.v_quantity = Quantity(v_quantity, v_label, v_unit)

    def _add_subsurface(self, mesh: Mesh, iso_filename: str, viso_filename: str = "") -> SubSurface:
        if viso_filename != "":
            subsurface = SubSurface(mesh, os.path.join(self.root_path, iso_filename),
                                    os.path.join(self.root_path, viso_filename))
        else:
            subsurface = SubSurface(mesh, os.path.join(self.root_path, iso_filename))
        self._subsurfaces.append(subsurface)

        return subsurface

    def get_subsurface(self, mesh: Mesh):
        """Returns the :class:`SubSurface` that contains data for the given mesh.
        """
        for iso in self._subsurfaces:
            if iso.mesh.id == mesh.id:
                return iso
        raise KeyError("The provided mesh is not valid for this operation in this simulation!")

    @property
    def has_color_data(self):
        """Defines whether there is color data for this isosurface or not.
        """
        return self._double_quantity

    @property
    def times(self):
        if not hasattr(self, "_times"):
            # Implicitly load the data for one subsurface and read times
            _ = self._subsurfaces[0].vertices
        return self._times
