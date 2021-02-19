import os
from copy import deepcopy

import numpy as np
import logging
from typing import Dict, Collection, Tuple
from typing_extensions import Literal

from fdsreader.utils import Dimension, Quantity, Mesh, Extent
from fdsreader import settings
import fdsreader.utils.fortran_data as fdtype

_HANDLED_FUNCTIONS = {}


def implements(np_function):
    """Decorator to register an __array_function__ implementation for Slices.
    """

    def decorator(func):
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class SubSlice:
    """Part of a slice that cuts through a single mesh.

    :ivar mesh: The mesh the subslice cuts through.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar dimension: :class:`Dimension` object containing information about steps in each dimension.
    """

    _offset = 3 * fdtype.new((('c', 30),)).itemsize + fdtype.new((('i', 6),)).itemsize

    def __init__(self, parent_slice, filename: str, dimension: Dimension, extent: Extent,
                 mesh: Mesh, has_vector_data: bool):
        self.mesh = mesh
        self.dimension = dimension
        self.extent = extent
        self.parent_slice = parent_slice

        self.filename = filename

        if has_vector_data:
            self.vector_filenames = dict()
            self._vector_data = dict()

    @property
    def shape(self) -> Tuple[int, int]:
        """2D-shape of the slice.
        """
        shape = self.dimension.shape(cell_centered=self.parent_slice.cell_centered)
        if self.parent_slice.orientation != 0:
            return shape[0], shape[1]
        return shape

    def _load_data(self, file_path: str, data_out: np.ndarray, n_t: int):
        n = self.dimension.size(cell_centered=self.parent_slice.cell_centered)
        dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((('f', n),)))

        with open(file_path, 'rb') as infile:
            infile.seek(self._offset)
            for i, data in enumerate(fdtype.read(infile, dtype_data, n_t)):
                data_out[i, :] = data[1].reshape(self.shape, order='F')

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the slice's data.
        """
        if not hasattr(self, "_data"):
            n_t = self.parent_slice.times.shape[0]

            file_path = os.path.join(self.parent_slice.root_path, self.filename)
            self._data = np.empty((n_t,) + self.shape, dtype=np.float32)
            self._load_data(file_path, self._data, n_t)
        return self._data

    @property
    def vector_data(self) -> Dict[str, np.ndarray]:
        """Method to lazy load the slice's vector data if it exists.
        """
        if not hasattr(self, "_vector_data"):
            raise AttributeError("There is no vector data available for this slice.")
        if len(self._vector_data) == 0:
            n_t = self.parent_slice.times.shape[0]

            for direction in self.vector_filenames.keys():
                file_path = os.path.join(self.parent_slice.root_path,
                                         self.vector_filenames[direction])
                self._vector_data[direction] = np.empty((n_t,) + self.shape, dtype=np.float32)
                self._load_data(file_path, self._vector_data[direction], n_t)
        return self._vector_data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "_data"):
            del self._data
        if hasattr(self, "_vector_data"):
            del self._vector_data

    def __repr__(self):
        return f"SubSlice(shape={self.shape}, mesh={self.mesh.id}, extent={self.extent})"


class Slice(np.lib.mixins.NDArrayOperatorsMixin):
    """Slice file data container including metadata. Consists of multiple subslices, one for each
        mesh the slice cuts through.

    :ivar root_path: Path to the directory containing all slice files.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar quantity: Quantity object containing information about the quantity calculated for this
        slice with the corresponding label and unit.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar orientation: Orientation [1,2,3] of the slice in case it is 2D, 0 otherwise.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    """

    def __init__(self, root_path: str, slice_id: str, cell_centered: bool, times: np.ndarray,
                 multimesh_data: Collection[Dict]):
        self.root_path = root_path
        self.cell_centered = cell_centered

        self.times = times

        self.id = slice_id

        # List of all subslices this slice consists of (one per mesh).
        self._subslices: Dict[Mesh, SubSlice] = dict()

        vector_temp = dict()
        for mesh_data in multimesh_data:
            if "-VELOCITY" in mesh_data["quantity"]:
                vector_temp[mesh_data["mesh"]] = dict()

        for mesh_data in multimesh_data:
            if "-VELOCITY" not in mesh_data["quantity"]:
                self.quantity = Quantity(mesh_data["quantity"], mesh_data["label"],
                                         mesh_data["unit"])
                self._subslices[mesh_data["mesh"]] = SubSlice(self, mesh_data["filename"],
                                                              mesh_data["dimension"],
                                                              mesh_data["extent"],
                                                              mesh_data["mesh"],
                                                              mesh_data["mesh"] in vector_temp)
            else:
                vector_temp[mesh_data["mesh"]][mesh_data["quantity"]] = mesh_data["filename"]

        for mesh, vector_filenames in vector_temp.items():
            if "U-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["u"] = vector_filenames["U-VELOCITY"]
            if "V-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["v"] = vector_filenames["V-VELOCITY"]
            if "W-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["w"] = vector_filenames["W-VELOCITY"]

        vals = self._subslices.values()
        self.extent = Extent(min(vals, key=lambda e: e.extent.x_start).extent.x_start,
                             max(vals, key=lambda e: e.extent.x_end).extent.x_end,
                             min(vals, key=lambda e: e.extent.y_start).extent.y_start,
                             max(vals, key=lambda e: e.extent.y_end).extent.y_end,
                             min(vals, key=lambda e: e.extent.z_start).extent.z_start,
                             max(vals, key=lambda e: e.extent.z_end).extent.z_end)

        if self.extent.x_start == self.extent.x_end:
            self.orientation = 1
        elif self.extent.y_start == self.extent.y_end:
            self.orientation = 2
        elif self.extent.z_start == self.extent.z_end:
            self.orientation = 3
        else:
            self.orientation = 0

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            for _, subslice in self._subslices.items():
                _ = subslice.data

    def __getitem__(self, mesh: Mesh):
        """Returns the :class:`SubSlice` that cuts through the given mesh.
        """
        return self._subslices[mesh]

    @property
    def extent_dirs(self) -> Tuple[
        Literal['x', 'y', 'z'], Literal['x', 'y', 'z'], Literal['x', 'y', 'z']]:
        """The directions in which there is an extent. All three dimensions in case the slice is 3D.
        """
        ior = self.orientation
        if ior == 0:
            return 'x', 'y', 'z'
        elif ior == 1:
            return 'y', 'z'
        elif ior == 2:
            return 'x', 'z'
        else:
            return 'x', 'y'

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subslice in self._subslices.values():
            subslice.clear_cache()

    def mean(self):
        """Calculates the mean over the whole slice.

        :returns: The calculated mean value.
        """
        return np.mean([np.mean(subsclice.data) for subsclice in self._subslices.values()])

    def __array__(self):
        """Method that will be called by numpy when trying to convert the object to a numpy ndarray.
        """
        raise UserWarning(
            "Slices can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containg all subslices is"
            " required, please request this functionality by submitting an issue on Github.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Method that will be called by numpy when using a ufunction with a Slice as input.

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
        if len(input_list) == 0:
            raise UserWarning(
                f"The {method} operation is not implemented for multiple slices as input yet. If"
                " you require this feature, please request this functionality by submitting an"
                " issue on Github.")

        new_slice = deepcopy(self)
        for subslice in new_slice._subslices.values():
            subslice._data = ufunc(subslice.data, input_list[0], **kwargs)
        return new_slice

    def __array_function__(self, func, types, args, kwargs):
        """Method that will be called by numpy when using an array function with a Slice as input.

        :returns: The output of the array function.
        """
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
            # Note: this allows subclasses that don't override __array_function__ to handle Slices.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __repr__(self):
        if self.orientation == 0:  # 3D-Slice
            return f"Slice([3D] cell_centered={self.cell_centered}, extent={self.extent}, times=[{self.times[0]:.2f},{self.times[1]:.2f},...,{self.times[-1]:.2f}])"
        else:  # 2D-Slice
            return f"Slice([2D] cell_centered={self.cell_centered}, extent={self.extent}, extent_dirs={self.extent_dirs}, orientation={self.orientation}, times=[{self.times[0]:.2f},{self.times[1]:.2f},...,{self.times[-1]:.2f}])"

# __array_function__ implementations
