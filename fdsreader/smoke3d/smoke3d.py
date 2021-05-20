import logging
import os
from copy import deepcopy
from typing import Dict
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


class SubSmoke3D:
    """Part of a smoke3d output for a single mesh.

    :ivar mesh: The mesh containing the data.
    """

    def __init__(self, file_path: str, mesh: Mesh, upper_bounds: np.ndarray, times: np.ndarray, max_length: int):
        self.file_path = file_path  # Path to the binary data file
        self.mesh = mesh
        self.upper_bounds = upper_bounds
        self.times = times
        self._max_length = max_length

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the Smoke3D data of a single mesh.
        """
        if not hasattr(self, "_data"):
            with open(self.file_path, 'rb') as infile:
                dtype_header = fdtype.new((('i', 8),))
                dtype_nchars = fdtype.new((('i', 2),))

                header = fdtype.read(infile, dtype_header, 1)[0][0]
                nx, ny, nz = int(header[3]), int(header[5]), int(header[7])
                data_shape = (nx + 1, ny + 1, nz + 1)

                self._data = np.empty((self.times.size,) + data_shape)

                for t in range(self.times.size):
                    fdtype.read(infile, fdtype.FLOAT, 1)  # Skip time value
                    nchars_out = int(fdtype.read(infile, dtype_nchars, 1)[0][0][1])

                    if nchars_out > 0:
                        dtype_data = fdtype.new((('u', nchars_out),))

                        rle_data = fdtype.read(infile, dtype_data, 1)[0][0]
                        decoded_data = np.empty(((nx + 1) * (ny + 1) * (nz + 1)))

                        # Decode run-length-encoded data (see "RLE" subroutine in smvv.f90)
                        i = 0
                        mark = np.uint8(255)
                        out_pos = 0
                        while i < nchars_out:
                            if rle_data[i] == mark:
                                value = rle_data[i+1]
                                repeats = rle_data[i+2]
                                i += 3
                            else:
                                value = rle_data[i]
                                repeats = 1
                                i += 1
                            decoded_data[out_pos:out_pos+repeats] = value
                            out_pos += repeats

                        self._data[t, :, :, :] = decoded_data.reshape(data_shape, order='F')

        return self._data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "_data"):
            del self._data


class Smoke3D(np.lib.mixins.NDArrayOperatorsMixin):
    """Smoke3D file data container including metadata. Consists of multiple subplots, one for each
        mesh.

    :ivar root_path: Path to the directory containing all slice files.
    :ivar time: The point in time when this data has been recorded.
    :ivar quantities: List with quantity objects containing information about recorded quantities
     calculated for this Smoke3D with the corresponding label and unit.
    """

    def __init__(self, root_path: str, times: np.ndarray, quantity: Quantity):
        self.root_path = root_path
        self.times = times
        self.quantity = quantity

        # List of all subplots this Smoke3D consists of (one per mesh).
        self._subsmokes: Dict[Mesh, SubSmoke3D] = dict()

    def _add_subsmoke(self, filename: str, mesh: Mesh, upper_bounds: np.ndarray, max_length: int) -> SubSmoke3D:
        subplot = SubSmoke3D(os.path.join(self.root_path, filename), mesh, upper_bounds, self.times, max_length)
        self._subsmokes[mesh] = subplot

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            _ = subplot.data

        return subplot

    def __getitem__(self, mesh: Mesh):
        """Returns the :class:`SubPlot` that contains data for the given mesh.
        """
        return self._subsmokes[mesh]

    def get_subsmoke(self, mesh: Mesh):
        """Returns the :class:`SubPlot` that contains data for the given mesh.
        """
        return self[mesh]

    @property
    def vmax(self):
        """Maximum value of all data at any time.
        """
        curr_max = max(np.max(subsmoke3d.upper_bounds) for subsmoke3d in self._subsmokes.values())
        if curr_max == np.float32(-1e33):
            return max(np.max(subsmoke3d.data) for subsmoke3d in self._subsmokes.values())
        return curr_max

    @implements(np.mean)
    def mean(self) -> np.ndarray:
        """Calculates the mean for each quantity individually of the whole Smoke3D.

        :returns: The calculated mean values.
        """
        mean_sums = np.zeros((5,))
        for subplot in self._subsmokes.values():
            mean_sums += np.mean(subplot.data, axis=(0, 1, 2))
        return mean_sums / len(self._subsmokes)

    @implements(np.std)
    def std(self) -> np.ndarray:
        """Calculates the standard deviation for each quantity individually of the whole Smoke3D.

        :returns: The calculated standard deviations.
        """
        stds = np.zeros((5,))
        n_q = len(self.quantities)
        for q in range(n_q):
            mean = self.mean
            sum = np.sum([np.sum(np.power(subplot.data[:, :, :, q] - mean, 2)) for subplot in self._subsmokes.values()])
            N = np.sum([subplot.data.size / n_q for subplot in self._subsmokes.values()])
            return np.sqrt(sum / N)
        return stds

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subplot in self._subsmokes.values():
            subplot.clear_cache()

    def __array__(self):
        """Method that will be called by numpy when trying to convert the object to a numpy ndarray.
        """
        raise UserWarning(
            "Smoke3Ds can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containg all subplots is"
            " required, please request this functionality by submitting an issue on Github.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Method that will be called by numpy when using a ufunction with a Smoke3D as input.

        :returns: A new smoke3d on which the ufunc has been applied.
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

        new_smoke3d = deepcopy(self)
        for subplot in self._subsmokes.values():
            subplot._data = ufunc(subplot.data, input_list[0], **kwargs)
        return new_smoke3d

    def __array_function__(self, func, types, args, kwargs):
        """Method that will be called by numpy when using an array function with a Slice as input.

        :returns: The output of the array function.
        """
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
            # Note: this allows subclasses that don't override __array_function__ to handle Smoke3Ds.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

# __array_function__ implementations
# ...
