import os
import numpy as np
import numpy.lib.mixins
import logging
from typing import List, Dict

from utils import FDS_DATA_TYPE_INTEGER, FDS_DATA_TYPE_FLOAT, FDS_DATA_TYPE_CHAR, \
    FDS_FORTRAN_BACKWARD, Extent, Quantity, settings

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
    Slice file data container including metadata. Consists of multiple subslices, one for each mesh
    the slice cuts through.
    :ivar root_path: Path to the directory containing all slice files.
    :ivar quantities: List with quantity objects containing information about the quantities
    calculated for this slice with the corresponding label and unit
    :ivar cell_centered: Indicates whether centered positioning for data is used
    :ivar _subslices: List of all subslices this slice consists of
    """

    def __init__(self, root_path: str, cell_centered: bool):
        self.root_path = root_path
        self.quantities: List[Quantity] = list()

        self.cell_centered = cell_centered
        self._subslices: List[_SubSlice] = list()

    def _add_subslice(self, filename: str, quantity: str, label: str, unit: str, extent: Extent,
                      mesh_id: int):
        """
        Adds another subslice to the slice.
        :param filename: Name of the slice file
        :param quantity: Quantity of the data
        :param label: Quantity label
        :param unit: Quantity unit
        :param extent: Extent object containing 3-dimensional extent information
        :param mesh_id: Id of the mesh the subslice cuts through
        """
        self.quantities.append(Quantity(quantity, label, unit))
        for subslice in self._subslices:
            if subslice.extent == extent:
                subslice.file_names[quantity] = filename
                break
        self._subslices.append(_SubSlice(filename, extent, quantity, mesh_id))

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            self._subslices[-1].get_data(quantity, self.root_path, self.cell_centered)

    def mean(self, quantity: str = None):
        """
        Calculates the mean over the whole slice.
        :param quantity: The quantity for which the mean should be calculated. If not provided, the
        first found quantity will be used
        :returns: The calculated mean value.
        """
        if quantity is None:
            quantity = self.quantities[0].quantity
        mean_sum = 0
        for subsclice in self._subslices:
            mean_sum += np.mean(subsclice.get_data(quantity, self.root_path, self.cell_centered))
        return mean_sum / len(self._subslices)

    def __array__(self):
        """
        Method that will be called by numpy when trying to convert the object to a numpy ndarray.
        """
        raise UserWarning(
            "Slices can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containg all subslices is"
            " required, please request this functionality by submitting an issue on Github.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Method that will be called by numpy when using a ufunction with a Slice as input.
        :returns: A new slice on which the ufunc has been applied.
        """
        if method != "__call__":
            logging.warning(
                "The %s method has been used which is not explicitly implemented. Correctness of"
                " results is not guaranteed. If you require this feature to be implemented please"
                " submit an issue on Github where you explain your use case.", method)
        input_list = list(inputs)
        for i, inp in enumerate(inputs):
            if isinstance(inp, self.__class__):
                del input_list[i]

        new_slice = Slice(self.root_path, self.cell_centered)
        for i, subslice in enumerate(self._subslices):
            q = self.quantities[i]
            new_slice._add_subslice(subslice.file_names[q.quantity],
                                    q.quantity, q.label,
                                    q.unit, subslice.extent, subslice.mesh_id)
            new_slice._subslices[-1]._data[q.quantity] = ufunc(
                subslice.get_data(q.quantity, self.root_path, self.cell_centered),
                input_list[0],
                **kwargs)
        return new_slice

    def __array_function__(self, func, types, args, kwargs):
        """
        Method that will be called by numpy when using an array function with a Slice as input.
        :returns: The output of the array function.
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
    Part of a slice that cuts through a single mesh.
    :ivar mesh_id: Id of the mesh the subslice cuts through
    :ivar extent: Extent object containing 3-dimensional extent information
    :ivar file_names: File names for the corresponding slice file depending on the quantity.
    :ivar _data: Dictionary that maps quantity to data.
    """

    def __init__(self, filename: str, extent: Extent, quantity: str, mesh_id: int):
        self.mesh_id = mesh_id
        self.extent = extent

        self.file_names = {quantity: filename}
        self._data: Dict[str, np.ndarray] = dict()

    def get_data(self, quantity: str, root_path: str, cell_centered: bool):
        """
        Method to lazy load the slice's data.
        :param quantity: Quantity of the data
        :param root_path: Path to the directory containing all slice files.
        :param cell_centered: Indicates whether centered positioning for data is used.
        """
        if quantity not in self._data:
            file_path = os.path.join(root_path, self.file_names[quantity])
            dtype_float = np.dtype(FDS_DATA_TYPE_FLOAT)
            n = self.extent.size(cell_centered=cell_centered)
            stride = np.dtype(DTYPE_STRIDE_RAW.format(n)).itemsize
            offset = 3 * DTYPE_HEADER.itemsize + DTYPE_INDEX.itemsize

            t_n = (os.stat(file_path).st_size - offset) // stride

            self._data[quantity] = np.empty((t_n, self.extent.x, self.extent.y, self.extent.z),
                                            dtype=dtype_float)

            with open(file_path, 'rb') as infile:
                for t in range(t_n):
                    infile.seek(offset + t * stride)
                    slice_data = np.fromfile(infile, dtype=dtype_float, count=n)
                    self._data[quantity][t, :] = slice_data.reshape(
                        (self.extent.x, self.extent.y, self.extent.z))
        return self._data[quantity]


# __array_function__ implementations
@implements(np.mean)
def mean(slc: Slice):
    return [slc.mean(q.quantity) for q in slc.quantities]
