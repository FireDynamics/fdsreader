import logging
import os
from typing import List
import numpy as np

from fdsreader.utils import Quantity, Mesh
import fdsreader.utils.fortran_data as fdtype

_HANDLED_FUNCTIONS = {np.mean: (lambda pl: pl.mean)}


def implements(np_function):
    """Decorator to register an __array_function__ implementation for Slices.
    """

    def decorator(func):
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class Plot3D(np.lib.mixins.NDArrayOperatorsMixin):
    """Plot3d file data container including metadata. Consists of multiple subplots, one for each mesh.

    :ivar root_path: Path to the directory containing all slice files.
    :ivar time: The point in time when this data has been recorded.
    :ivar quantities: List with quantity objects containing information about recorded quantities
     calculated for this Plot3D with the corresponding label and unit.
    """
    def __init__(self, root_path: str, time: float, quantities: List[Quantity]):
        self.root_path = root_path
        self.time = time
        self.quantities = quantities

        # List of all subplots this slice consists of (one per mesh).
        self._subplots: List[_SubPlot3D] = list()

    def _add_subplot(self, filename: str, mesh: Mesh):
        self._subplots.append(_SubPlot3D(os.path.join(self.root_path, filename), mesh))

    def mean(self) -> np.ndarray:
        """Calculates the mean over each quantity individually of the whole Plot3D.

        :returns: The calculated mean values.
        """
        mean_sums = np.zeros((5,))
        for subplot in self._subplots:
            mean_sums += np.mean(subplot.get_data(), axis=(0, 1, 2))
        return mean_sums / len(self._subplots)

    def __array__(self):
        """Method that will be called by numpy when trying to convert the object to a numpy ndarray.
        """
        raise UserWarning(
            "Plot3Ds can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containg all subplots is"
            " required, please request this functionality by submitting an issue on Github.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Method that will be called by numpy when using a ufunction with a Plot3D as input.

        :returns: A new plot3d on which the ufunc has been applied.
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

        new_pl3d = Plot3D(self.root_path, self.time, self.quantities)
        for i, subplot in enumerate(self._subplots):
            new_pl3d._add_subplot(subplot.file_path, subplot.mesh)
            new_pl3d._subplots[-1]._data = ufunc(subplot.get_data(), input_list[0], **kwargs)
        return new_pl3d

    def __array_function__(self, func, types, args, kwargs):
        """Method that will be called by numpy when using an array function with a Slice as input.

        :returns: The output of the array function.
        """
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
            # Note: this allows subclasses that don't override __array_function__ to handle Plot3Ds.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)


class _SubPlot3D:
    """Subplot of a plot3d output for a single mesh.

    :ivar file_path: Path to the binary data file.
    :ivar mesh: The mesh containing the data.
    """
    # Offset of the binary file to the end of the file header.
    _offset = fdtype.new((('i', 3),)).itemsize + fdtype.new((('i', 4),)).itemsize

    def __init__(self, file_path: str, mesh: Mesh):
        self.file_path = file_path
        self.mesh = mesh

    def get_data(self) -> np.ndarray:
        """Method to lazy load the 3D data for a single mesh.

        :returns: 4D numpy array wiht (x,y,z,q) as dimensions, while q represents the 5 quantities.
        """
        if not hasattr(self, "_data"):
            with open(self.file_path, 'rb') as infile:
                dtype_data = fdtype.new((('f', self.mesh.extent.size(cell_centered=False) * 5),))
                infile.seek(self._offset)
                self._data = fdtype.read(infile, dtype_data, 1)[0][0].reshape(
                    (self.mesh.extent.x, self.mesh.extent.y, self.mesh.extent.z, 5))
        return self._data


# __array_function__ implementations
# ...
