from __future__ import annotations
import os
from copy import deepcopy

import numpy as np
import logging
from typing import Dict, Collection

from fdsreader.utils import Extent, Quantity, settings, Mesh
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

    :ivar root_path: Path to the directory containing the slice file.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar mesh: The mesh the subslice cuts through.
    :ivar extent: Extent object containing 3-dimensional extent information. Values are indexes for
        the actual grid values of the mesh.
    :ivar file_names: File names for the corresponding slice file depending on the quantity.
    """

    _offset = 3 * fdtype.new((('c', 30),)).itemsize + fdtype.new((('i', 6),)).itemsize

    def __init__(self, filename: str, root_path: str, cell_centered: bool, extent: Extent,
                 mesh: Mesh, times: np.ndarray):
        self.mesh = mesh
        self.extent = extent
        self.root_path = root_path
        self.cell_centered = cell_centered

        self.filename = filename
        self._times = times

        self.shape = (self.extent.x - 1 if self.cell_centered else self.extent.x,
                      self.extent.y - 1 if self.cell_centered else self.extent.y,
                      self.extent.z - 1 if self.cell_centered else self.extent.z)

        if True:
            self.vector_filenames = dict()
            self._vector_data = dict()

    def _load_data(self, file_path: str, data_out: np.ndarray, t_n: int, fill_times: bool = False):
        n = self.extent.size(cell_centered=self.cell_centered)
        dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((('f', n),)))

        with open(file_path, 'rb') as infile:
            infile.seek(self._offset)
            for i, data in enumerate(fdtype.read(infile, dtype_data, t_n)):
                if fill_times:
                    self._times[i] = data[0][0]
                data_out[i, :] = data[1].reshape(self.shape)

    @property
    def data(self) -> np.ndarray:
        """
        Method to lazy load the slice's data for a specific quantity.
        """
        if not hasattr(self, "_data"):
            t_n = self._times.shape[0]

            file_path = os.path.join(self.root_path, self.filename)
            self._data = np.empty((t_n,) + self.shape, dtype=np.float32)
            self._load_data(file_path, self._data, t_n, fill_times=self._times[0] == -1)
        return self._data

    @property
    def vector_data(self) -> Dict[str, np.ndarray]:
        if not hasattr(self, "_vector_data"):
            raise AttributeError("There is no vector data available for this slice.")
        if len(self._vector_data) == 0:
            t_n = self._times.shape[0]

            for direction in ('u', 'v', 'w'):
                file_path = os.path.join(self.root_path, self.vector_filenames[direction])
                self._vector_data[direction] = np.empty((t_n,) + self.shape, dtype=np.float32)
                self._load_data(file_path, self._vector_data[direction], t_n)
        return self._vector_data


class Slice(np.lib.mixins.NDArrayOperatorsMixin):
    """Slice file data container including metadata. Consists of multiple subslices, one for each
        mesh the slice cuts through.

    :ivar root_path: Path to the directory containing all slice files.
    :ivar quantities: List with quantity objects containing information about the
        quantities calculated for this slice with the corresponding label and unit.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar times: Numpy array containing all times for which data has been recorded.
    """

    def __init__(self, root_path: str, cell_centered: bool, multimesh_data: Collection[Dict]):
        self.root_path = root_path
        self.cell_centered = cell_centered

        n = next(iter(multimesh_data))["extent"].size(cell_centered=self.cell_centered)
        dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((('f', n),)))
        t_n = (os.stat(os.path.join(self.root_path, next(iter(multimesh_data))[
            "filename"])).st_size - SubSlice._offset) // dtype_data.itemsize

        self._times = np.empty((t_n,), dtype=np.float32)
        self._times[0] = -1  # Mark as uninitialized

        # List of all subslices this slice consists of (one per mesh).
        self._subslices: Dict[Mesh, SubSlice] = dict()

        vector_temp = dict()
        for mesh_data in multimesh_data:
            if "-VELOCITY" not in mesh_data["quantity"]:
                self.quantity = Quantity(mesh_data["quantity"], mesh_data["label"],
                                         mesh_data["unit"])
                self._subslices[mesh_data["mesh"]] = SubSlice(mesh_data["filename"], self.root_path,
                                                              self.cell_centered,
                                                              mesh_data["extent"],
                                                              mesh_data["mesh"], self._times)
            else:
                if mesh_data["mesh"] in vector_temp:
                    vector_temp[mesh_data["mesh"]][mesh_data["quantity"]] = mesh_data["filename"]
                else:
                    vector_temp[mesh_data["mesh"]] = {mesh_data["quantity"]: mesh_data["filename"]}

        for mesh, vector_filenames in vector_temp.items():
            self._subslices[mesh].vector_filenames["u"] = vector_filenames["U-VELOCITY"]
            self._subslices[mesh].vector_filenames["v"] = vector_filenames["V-VELOCITY"]
            self._subslices[mesh].vector_filenames["w"] = vector_filenames["W-VELOCITY"]

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            for _, subslice in self._subslices.items():
                _ = subslice.data

    def get_subslice(self, mesh: Mesh):
        """Returns the :class:`SubSlice` that cuts through the given mesh.
        """
        return self._subslices[mesh]

    @property
    def times(self):
        if self._times is None:
            raise AssertionError("Time data is not available before initializing the first"
                                 " subslice. This indicates that this function has been called"
                                 " mid-initialization, which should not happen!")
        elif self._times[0] == -1:
            # Implicitly load the data for one subslice, which (as a side effect) sets time data
            _ = next(iter(self._subslices.values())).data
        return self._times

    # def copy(self) -> __class__:
    #     """Performs a deep copy of a slice.
    #
    #     :returns: A deep copy of the original slice.
    #     """

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

# __array_function__ implementations
