import os
import numpy as np
import logging
from typing import List, BinaryIO

from utils import FDS_DATA_TYPE_INTEGER, FDS_DATA_TYPE_FLOAT, FDS_DATA_TYPE_CHAR, \
    FDS_FORTRAN_BACKWARD, Quantity, settings


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
            dtype_header_nlevels = np.dtype(f"8{FDS_DATA_TYPE_INTEGER}")
            nlevels = np.fromfile(infile, dtype=dtype_header_nlevels, count=1)[0][7]
            print(nlevels)
            dtype_header_levels = np.dtype(f"2{FDS_DATA_TYPE_INTEGER}, ({nlevels},){FDS_DATA_TYPE_FLOAT}, 3{FDS_DATA_TYPE_INTEGER}")
            self.levels = np.fromfile(infile, dtype=dtype_header_levels, count=1)[0][1]
            print(self.levels)
            dtype_dims = np.dtype(f"{FDS_DATA_TYPE_FLOAT}, 3{FDS_DATA_TYPE_INTEGER}")
            time_data = np.fromfile(infile, dtype=dtype_dims, count=1)

            print(time_data)
            exit(0)
            self.n_vertices = time_data[0][1][1]
            self.n_triangles = time_data[0][1][2]

            self.offset = dtype_header_nlevels.itemsize + dtype_header_levels.itemsize + dtype_dims.itemsize

            if not settings.LAZY_LOAD:
                self._load_data(infile)

        if self._double_quantity:
            v_header_offset = dtype_header_nlevels.itemsize + dtype_header_levels.itemsize + np.dtype(f"2{FDS_DATA_TYPE_INTEGER}").itemsize
            v_dtype_dims = np.dtype(f"{FDS_DATA_TYPE_FLOAT}, 4{FDS_DATA_TYPE_INTEGER}")
            self.v_offset = v_header_offset + v_dtype_dims.itemsize

            with open(self.v_file_path, 'rb') as infile:
                infile.seek(v_header_offset)
                time_data = np.fromfile(infile, dtype=v_dtype_dims, count=1)
                logging.error(time_data)

                self.v_n = 0
                if not settings.LAZY_LOAD:
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
        dtype_vertices = np.dtype(f"3{FDS_DATA_TYPE_FLOAT}")
        dtype_triangles = np.dtype(f"3{FDS_DATA_TYPE_INTEGER}")
        dtype_surfaces = np.dtype(FDS_DATA_TYPE_INTEGER)

        infile.seek(self.offset)

        self._vertices = np.fromfile(infile, dtype=dtype_vertices, count=self.n_vertices)
        self._triangles = np.fromfile(infile, dtype=dtype_triangles, count=self.n_triangles)
        self._surfaces = np.fromfile(infile, dtype=dtype_surfaces, count=self.n_triangles)

    def _load_vdata(self, infile: BinaryIO):
        dtype_color = np.dtype(FDS_DATA_TYPE_FLOAT)
        infile.seek(self.v_offset)
        self._colors = np.fromfile(infile, dtype=dtype_color, count=self.v_n)
