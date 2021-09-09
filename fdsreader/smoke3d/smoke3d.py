import logging
import os
from copy import deepcopy
from typing import Dict
from typing_extensions import Literal
import numpy as np

from fdsreader.fds_classes import Mesh
from fdsreader.utils import Quantity
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
    :ivar upper_bounds: Numpy ndarray containing the maxmimum data value for each timestep.
    :ivar times: Numpy ndarray containing all time steps for which data has been written out.
    """

    def __init__(self, file_path: str, mesh: Mesh, upper_bounds: np.ndarray, times: np.ndarray):
        self.mesh = mesh
        self.upper_bounds = upper_bounds
        self.times = times

        self._file_path = file_path  # Path to the binary data file

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the Smoke3D data of a single mesh.
        """
        if not hasattr(self, "_data"):
            with open(self._file_path, 'rb') as infile:
                dtype_header = fdtype.new((('i', 8),))
                dtype_nchars = fdtype.new((('i', 2),))

                header = fdtype.read(infile, dtype_header, 1)[0][0]
                nx, ny, nz = int(header[3]), int(header[5]), int(header[7])
                data_shape = (nx + 1, ny + 1, nz + 1)

                self._data = np.empty((self.times.size,) + data_shape, dtype=np.float32)

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
                                value = rle_data[i + 1]
                                repeats = rle_data[i + 2]
                                i += 3
                            else:
                                value = rle_data[i]
                                repeats = 1
                                i += 1
                            decoded_data[out_pos:out_pos + repeats] = value
                            out_pos += repeats

                        self._data[t, :, :, :] = decoded_data.reshape(data_shape, order='F')

        return self._data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "_data"):
            del self._data


class Smoke3D(np.lib.mixins.NDArrayOperatorsMixin):
    """Smoke3D file data container including metadata. Consists of multiple subsmokes, one for each
        mesh.

    :ivar times: Numpy ndarray containing all time steps for which data has been written out.
    :ivar quantity: :class:`Quantity` object containing information about the recorded quantity and its unit.
    """

    def __init__(self, root_path: str, times: np.ndarray, quantity: Quantity):
        self._root_path = root_path
        self.times = times
        self.quantity = quantity

        # List of all subsmokes this Smoke3D consists of (one per mesh).
        self._subsmokes: Dict[Mesh, SubSmoke3D] = dict()

    def _add_subsmoke(self, filename: str, mesh: Mesh, upper_bounds: np.ndarray) -> SubSmoke3D:
        subsmoke = SubSmoke3D(os.path.join(self._root_path, filename), mesh, upper_bounds, self.times)
        self._subsmokes[mesh] = subsmoke

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            _ = subsmoke.data

        return subsmoke

    def __getitem__(self, mesh: Mesh):
        """Returns the :class:`SubSmoke` that contains data for the given mesh.
        """
        return self.get_subsmoke(mesh)

    def get_subsmoke(self, mesh: Mesh):
        """Returns the :class:`SubSmoke` that contains data for the given mesh.
        """
        return self._subsmokes[mesh]

    @property
    def subsmokes(self):
        return self._subsmokes.items()

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
        """Calculates the mean value of all Smoke3D data for this quantity.
        """
        return np.sum([np.mean(subsmoke.data) for subsmoke in self._subsmokes.values()]) / len(self._subsmokes)

    @implements(np.std)
    def std(self) -> np.ndarray:
        """Calculates the standard deviation of all Smoke3D data for this quantity.
        """
        mean = self.mean
        sum = np.sum([np.sum(np.power(subsmoke.data - mean, 2)) for subsmoke in self._subsmokes.values()])
        N = np.sum([subsmoke.data.size for subsmoke in self._subsmokes.values()])
        return np.sqrt(sum / N)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subsmoke in self._subsmokes.values():
            subsmoke.clear_cache()

    def __array__(self):
        """Method that will be called by numpy when trying to convert the object to a numpy ndarray.
        """
        raise UserWarning(
            "Smoke3Ds can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containg all subsmokes is"
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
        for subsmoke in self._subsmokes.values():
            subsmoke._data = ufunc(subsmoke.data, input_list[0], **kwargs)
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
