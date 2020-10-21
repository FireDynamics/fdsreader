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


_HANDLED_FUNCTIONS = {}


def implements(np_function):
    """
    Decorator to register an __array_function__ implementation for Slices.
    """
    def decorator(func):
        _HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


class Slice(numpy.lib.mixins.NDArrayOperatorsMixin):
    """

    """
    def __init__(self, root_path: str, cell_centered: bool):
        self.root_path = root_path
        self.quantities = list()

        self.cell_centered = cell_centered
        self._subslices = list()

    def _add_subslice(self, filename: str, quantity: str, label: str, unit: str, extent: Extent,
                      mesh_id: int):
        """
        :param filename:
        :param quantity:
        :param label:
        :param unit:
        :param extent:
        :param mesh_id:
        """
        self.quantities.append(Quantity(quantity, label, unit))
        for subslice in self._subslices:
            if subslice.extent == extent:
                subslice.file_names[quantity] = filename
                break
        self._subslices.append(_SubSlice(filename, extent, quantity, mesh_id))

    def __array__(self):
        """

        """
        raise UserWarning("Slices can not be converted to numpy arrays, but they support all typical numpy operations"
                          "such as np.multiply.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        
        """
        if method != "__call__":
            logging.warning("The %s method has been used which is not explicitly implemented. Correctness of results is"
                            " not guaranteed. If you require this feature to be implemented please submit an issue"
                            " on github where you explain your use case.", method)
        input_list = list(inputs)
        for i, input in enumerate(inputs):
            if isinstance(input, self.__class__):
                del input_list[i]

        new_slice = Slice(self.root_path, self.cell_centered)
        for i, subslice in enumerate(self._subslices):
            q = self.quantities[i]
            new_slice._add_subslice(subslice.file_names[q.quantity], q.quantity, q.label,
                                    q.unit, subslice.extent, subslice.mesh_id)
            new_slice._subslices[-1]._data[q.quantity] = ufunc(
                subslice.get_data(q.quantity, self.root_path, self.cell_centered), input_list[0], **kwargs)
        return new_slice

    def __array_function__(self, func, types, args, kwargs):
        """

        """
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
            # Note: this allows subclasses that don't override
            # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)


class _SubSlice:
    """

    """
    def __init__(self, filename: str, extent: Extent, quantity: str, mesh_id: int):
        self.mesh_id = mesh_id
        self.extent = extent

        self.file_names = {quantity: filename}
        self._data = dict()

    def get_data(self, quantity: str, root_path: str, cell_centered: bool):
        """

        """
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


# __array_function__ implementations
@implements(np.mean)
def mean(slc: Slice):
    mean = 0
    for subsclice in slc._subslices:
        mean += np.mean(subsclice.get_data())
    return mean / len(slc._subslices)
