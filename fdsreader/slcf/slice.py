import os
import numpy as np
import logging
from typing import List, Dict

from fdsreader.utils import Extent, Quantity, settings, Mesh
import fdsreader.utils.fortran_data as fdtype

_HANDLED_FUNCTIONS = {}


def implements(np_function):
    """
    Decorator to register an __array_function__ implementation for Slices.
    """

    def decorator(func):
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class Slice(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Slice file data container including metadata. Consists of multiple subslices, one for each mesh
     the slice cuts through.

    :ivar root_path: Path to the directory containing all slice files.
    :ivar quantities: List with quantity objects containing information about the
        quantities calculated for this slice with the corresponding label and unit.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar times: Numpy array containing all times for which data has been recorded.
    """

    def __init__(self, root_path: str, cell_centered: bool):
        self.root_path = root_path
        self.cell_centered = cell_centered

        self.quantities: List[Quantity] = list()
        self.times = None
        # List of all subslices this slice consists of (one per mesh).
        self._subslices: List[SubSlice] = list()

    def _add_subslice(self, filename: str, quantity: str, label: str, unit: str, extent: Extent,
                      mesh: Mesh):
        """
        Adds another subslice to the slice.

        :param filename: Name of the slice file.
        :param quantity: Quantity of the data.
        :param label: Quantity label.
        :param unit: Quantity unit.
        :param extent: Extent object containing 3-dimensional extent information.
        :param mesh: The mesh the subslice cuts through.
        """
        self.quantities.append(Quantity(quantity, label, unit))
        for subslice in self._subslices:
            if subslice.extent == extent:
                subslice.file_names[quantity] = filename
                break

        dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new(
            (('f', extent.size(cell_centered=self.cell_centered)),)))

        if self.times is None:
            t_n = (os.stat(os.path.join(self.root_path, filename)).st_size - SubSlice._offset) \
                  // dtype_data.itemsize
            self.times = np.empty(shape=(t_n,))
            self.times[0] = -1

        self._subslices.append(
            SubSlice(filename, self.root_path, self.cell_centered, extent, quantity, mesh,
                     self.times))

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            self._subslices[-1].get_data(quantity)

    def get_subslice(self, mesh: Mesh):
        """
        Returns the SubSlice that cuts through the given mesh.

        :param mesh:
        """
        for slc in self._subslices:
            if slc.mesh.id == mesh.id:
                return slc
        return None

    def mean(self, quantity: str = None):
        """
        Calculates the mean over the whole slice.

        :param quantity: The quantity for which the mean should be calculated. If not provided, the
         first found quantity will be used.
        :returns: The calculated mean value.
        """
        if quantity is None:
            quantity = self.quantities[0].quantity
        mean_sum = 0
        for subsclice in self._subslices:
            mean_sum += np.mean(subsclice.get_data(quantity))
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
                                    q.unit, subslice.extent, subslice.mesh)
            new_slice._subslices[-1]._data[q.quantity] = ufunc(
                subslice.get_data(q.quantity),
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
            # Note: this allows subclasses that don't override __array_function__ to handle Slices.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)


class SubSlice:
    """
    Part of a slice that cuts through a single mesh.

    :ivar root_path: Path to the directory containing the slice file.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar mesh: The mesh the subslice cuts through.
    :ivar extent: Extent object containing 3-dimensional extent information.
    :ivar file_names: File names for the corresponding slice file depending on the quantity.
    """

    _offset = 3 * fdtype.new((('c', 30),)).itemsize + fdtype.new((('i', 6),)).itemsize

    def __init__(self, filename: str, root_path: str, cell_centered: bool, extent: Extent,
                 quantity: str, mesh: Mesh, times):
        self.mesh = mesh
        self.extent = extent
        self.root_path = root_path
        self.cell_centered = cell_centered

        self.file_names = {quantity: filename}
        self._data: Dict[str, np.ndarray] = dict()  # Dictionary that maps quantity to data.
        self._times = times

    def get_data(self, quantity: str) -> np.ndarray:
        """
        Method to lazy load the slice's data for a specific quantity.
        """
        if quantity not in self._data:
            file_path = os.path.join(self.root_path, self.file_names[quantity])
            n = self.extent.size(cell_centered=self.cell_centered)
            shape = (self.extent.x - 1 if self.cell_centered else self.extent.x,
                     self.extent.y - 1 if self.cell_centered else self.extent.y,
                     self.extent.z - 1 if self.cell_centered else self.extent.z)

            dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((('f', n),)))

            fill_times = self._times[0] == -1
            t_n = self._times.shape[0]

            self._data[quantity] = np.empty((t_n,) + shape, dtype=np.float32)

            with open(file_path, 'rb') as infile:
                infile.seek(self._offset)
                for i, data in enumerate(fdtype.read(infile, dtype_data, t_n)):
                    if fill_times:
                        self._times[i] = data[0][0]
                    self._data[quantity][i, :] = data[1].reshape(shape)
        return self._data[quantity]


# __array_function__ implementations
@implements(np.mean)
def mean(slc: Slice):
    return [slc.mean(q.quantity) for q in slc.quantities]
