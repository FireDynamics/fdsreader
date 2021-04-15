import logging
import os
from copy import deepcopy
from typing import Dict, Sequence
import numpy as np

from fdsreader.utils import Quantity, Mesh
from fdsreader import settings
import fdsreader.utils.fortran_data as fdtype

_HANDLED_FUNCTIONS = {np.mean: (lambda pl: pl.mean)}


def implements(np_function):
    """Decorator to register an __array_function__ implementation for Slices.
    """

    def decorator(func):
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class SubPlot3D:
    """Subplot of a pl3d output for a single mesh.

    :ivar mesh: The mesh containing the data.
    """
    # Offset of the binary file to the end of the file header.
    _offset = fdtype.new((('i', 3),)).itemsize + fdtype.new((('i', 4),)).itemsize

    def __init__(self, file_path: str, mesh: Mesh):
        self.file_path = file_path  # Path to the binary data file
        self.mesh = mesh

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the 3D data for each quantity of a single mesh.

        :returns: 4D numpy array wiht (x,y,z,q) as dimensions, while q represents the 5 quantities.
        """
        if not hasattr(self, "_data"):
            with open(self.file_path, 'rb') as infile:
                dtype_data = fdtype.new((('f', self.mesh.dimension.size() * 5),))
                infile.seek(self._offset)
                self._data = fdtype.read(infile, dtype_data, 1)[0][0].reshape(
                    self.mesh.dimension.shape() + (5,), order='F')
        return self._data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "_data"):
            del self._data


class Plot3D(np.lib.mixins.NDArrayOperatorsMixin):
    """Plot3d file data container including metadata. Consists of multiple subplots, one for each
        mesh.

    :ivar root_path: Path to the directory containing all slice files.
    :ivar time: The point in time when this data has been recorded.
    :ivar quantities: List with quantity objects containing information about recorded quantities
     calculated for this Plot3D with the corresponding label and unit.
    """

    def __init__(self, root_path: str, time: float, quantities: Sequence[Quantity]):
        self.root_path = root_path
        self.time = time
        self.quantities = quantities

        # List of all subplots this Plot3D consists of (one per mesh).
        self._subplots: Dict[Mesh, SubPlot3D] = dict()

    def _add_subplot(self, filename: str, mesh: Mesh) -> SubPlot3D:
        subplot = SubPlot3D(os.path.join(self.root_path, filename), mesh)
        self._subplots[mesh] = subplot

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            _ = subplot.data

        return subplot

    def __getitem__(self, mesh: Mesh):
        """Returns the :class:`SubPlot` that contains data for the given mesh.
        """
        return self._subplots[mesh]

    @implements(np.mean)
    def mean(self) -> np.ndarray:
        """Calculates the mean over each quantity individually of the whole Plot3D.

        :returns: The calculated mean values.
        """
        mean_sums = np.zeros((5,))
        for subplot in self._subplots.values():
            mean_sums += np.mean(subplot.data, axis=(0, 1, 2))
        return mean_sums / len(self._subplots)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subplot in self._subplots.values():
            subplot.clear_cache()

    def __array__(self):
        """Method that will be called by numpy when trying to convert the object to a numpy ndarray.
        """
        raise UserWarning(
            "Plot3Ds can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containg all subplots is"
            " required, please request this functionality by submitting an issue on Github.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Method that will be called by numpy when using a ufunction with a Plot3D as input.

        :returns: A new pl3d on which the ufunc has been applied.
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

        new_pl3d = deepcopy(self)
        for subplot in self._subplots.values():
            subplot._data = ufunc(subplot.data, input_list[0], **kwargs)
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

# __array_function__ implementations
# ...
