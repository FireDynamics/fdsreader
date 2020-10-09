import logging
import os
from typing import List, Tuple, Union, Iterator
from typing_extensions import Literal

import numpy as np
import numpy.lib.mixins

from utils import FDS_DATA_TYPE_INTEGER, FDS_DATA_TYPE_FLOAT, FDS_DATA_TYPE_CHAR, \
    FDS_FORTRAN_BACKWARD, Extent, Quantity

if FDS_FORTRAN_BACKWARD:
    DTYPE_HEADER = np.dtype(f"30{FDS_DATA_TYPE_CHAR}, {FDS_DATA_TYPE_INTEGER}")
    DTYPE_INDEX = np.dtype(f"6{FDS_DATA_TYPE_INTEGER}, {FDS_DATA_TYPE_INTEGER}")
    DTYPE_TIME = np.dtype(f"{FDS_DATA_TYPE_FLOAT}, {FDS_DATA_TYPE_INTEGER}")
    DTYPE_STRIDE_RAW = f"({{}}){FDS_DATA_TYPE_FLOAT}, {FDS_DATA_TYPE_INTEGER}"
else:
    DTYPE_HEADER = np.dtype("30" + FDS_DATA_TYPE_CHAR)
    DTYPE_INDEX = np.dtype("6" + FDS_DATA_TYPE_INTEGER)
    DTYPE_TIME = np.dtype(FDS_DATA_TYPE_FLOAT)
    DTYPE_STRIDE_RAW = "({})" + FDS_DATA_TYPE_FLOAT


class Slice(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, root_path: str, cell_centered: bool):
        self.root_path = root_path
        self.quantities = list()

        self.cell_centered = cell_centered
        self._subslices = list()

    def _add_subslice(self, filename: str, quantity: str, label: str, unit: str, extent: Extent,
                      mesh_id: int):
        self.quantities.append(Quantity(quantity, label, unit))
        for subslice in self._subslices:
            if subslice.extent == extent:
                subslice.file_names[quantity] = filename
                break
        self._subslices.append(_SubSlice(filename, extent, quantity, mesh_id))

    def __array__(self):
        raise NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        new_slice = Slice(self.root_path, self.cell_centered)
        for i, subslice in enumerate(self._subslices):
            quantity = self.quantities[i]
            new_slice._add_subslice(subslice.filename, quantity.quantity, quantity.label,
                                    quantity.unit, subslice.extent, subslice.mesh_id)
            new_slice._subslices[-1]._data[quantity.quantity] = ufunc(subslice.get_data(), *inputs, **kwargs)



class _SubSlice:
    def __init__(self, filename: str, extent: Extent, quantity: str, mesh_id: int):
        self.mesh_id = mesh_id
        self.extent = extent

        self.file_names = {quantity: filename}
        self._data = dict()

    def get_data(self, quantity: str, root_path: str, cell_centered: bool):
        if quantity not in self._data:
            file_path = os.path.join(root_path, self.file_names[quantity])
            dtype_float = np.dtype(FDS_DATA_TYPE_FLOAT)
            N = self.extent.size(cell_centered=cell_centered)
            stride = np.dtype(DTYPE_STRIDE_RAW.format(N)).itemsize
            offset = 3 * DTYPE_HEADER.itemsize + DTYPE_INDEX.itemsize

            t_n = (os.stat(file_path).st_size - offset) // stride

            self._data[quantity] = np.empty((t_n, self.extent.x, self.extent.y, self.extent.z), dtype=dtype_float)

            with open(file_path, 'r') as infile:
                for t in range(t_n):
                    infile.seek(offset + t * stride)
                    slice_data = np.fromfile(infile, dtype=dtype_float, count=N)
                    self._data[quantity][t, :] = slice_data.reshape((self.extent.x, self.extent.y, self.extent.z))
        return self._data[quantity]
