import os
from typing import BinaryIO

from utils import Quantity, settings
import utils.fortran_data as fdtype


class Isosurface:
    def __init__(self, root_path: str, double_quantity: bool, iso_filename: str, quantity: str,
                 label: str, unit: str, viso_filename: str = "", v_quantity: str = "",
                 v_label: str = "", v_unit: str = ""):

        self._double_quantity = double_quantity
        self.file_path = os.path.join(root_path, iso_filename)
        self.quantity = Quantity(quantity, label, unit)

        if self._double_quantity:
            self.v_quantity = Quantity(v_quantity, v_label, v_unit)
            self.v_file_path = os.path.join(root_path, viso_filename)

        with open(self.file_path, 'rb') as infile:
            nlevels = fdtype.read(infile, fdtype.INT, 3)[2][0]

            dtype_header_levels = fdtype.new((('f', nlevels),))
            self.levels = fdtype.read(infile, dtype_header_levels, 1)[0]

            dtype_header_rest = fdtype.combine(fdtype.INT, fdtype.new((('i', 2),)),
                                               fdtype.new((('f', 1), ('i', 1))))
            self._offset = fdtype.INT.itemsize * 3 + dtype_header_levels.itemsize + \
                           dtype_header_rest.itemsize
            infile.seek(self._offset)

            dtype_dims = fdtype.new((('i', 2),))
            dims_data = fdtype.read(infile, dtype_dims, 1)
            self.n_vertices = dims_data[0][0]
            self.n_triangles = dims_data[0][1]
            print(self.n_vertices, self.n_vertices)

            self._offset += dtype_dims.itemsize

            if not settings.LAZY_LOAD:
                self._load_data(infile)

        if self._double_quantity:
            self._v_offset = fdtype.INT.itemsize * 2 + fdtype.FLOAT.itemsize + fdtype.new(
                (('i', 4),)).itemsize

            if not settings.LAZY_LOAD:
                with open(self.v_file_path, 'rb') as infile:
                    self._load_vdata(infile)

    @property
    def vertices(self):
        if not hasattr(self, "_vertices"):
            with open(self.file_path, 'rb') as infile:
                self._load_data(infile)
        return self._vertices

    def triangles(self):
        if not hasattr(self, "_triangles"):
            with open(self.file_path, 'rb') as infile:
                self._load_data(infile)
        return self._triangles

    @property
    def surfaces(self):
        if not hasattr(self, "_surfaces"):
            with open(self.file_path, 'rb') as infile:
                self._load_data(infile)
        return self._surfaces

    @property
    def has_color_data(self):
        return self._double_quantity

    @property
    def colors(self):
        if self._double_quantity:
            if not hasattr(self, "_colors"):
                with open(self.v_file_path, 'rb') as infile:
                    self._load_vdata(infile)
            return self._colors
        else:
            raise UserWarning("The isosurface does not have any associated color-data. Use the"
                              " attribute 'has_color_data' to check if an isosurface has associated"
                              " color-data.")

    def _load_data(self, infile: BinaryIO):
        dtype_vertices = fdtype.new((('f', 3 * self.n_vertices),))
        dtype_triangles = fdtype.new((('i', 3 * self.n_triangles),))
        dtype_surfaces = fdtype.new((('i', self.n_triangles),))
        infile.seek(self._offset)

        self._vertices = fdtype.read(infile, dtype_vertices, 1)
        self._triangles = fdtype.read(infile, dtype_triangles, 1)
        self._surfaces = fdtype.read(infile, dtype_surfaces, 1)

    def _load_vdata(self, infile: BinaryIO):
        dtype_color = fdtype.new((('f', self.n_vertices),))
        infile.seek(self._v_offset)
        self._colors = fdtype.read(infile, dtype_color, self.n_vertices)
