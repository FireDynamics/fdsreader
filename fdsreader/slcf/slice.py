import os
from copy import deepcopy, copy

import numpy as np
import logging
from typing import Dict, Collection, Tuple, Union, List
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

    def __init__(self, parent_slc, filename: str, dimension: Dimension, extent: Extent, mesh: Mesh):
        self.mesh = mesh
        self.dimension = dimension
        self.extent = extent
        self.parent_slice = parent_slc

        self.filename = filename

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

    @property
    def orientation(self):
        """Orientation [1,2,3] of the slice in case it is 2D, 0 otherwise.
        """
        return self.parent_slice.orientation

    def _load_data(self, file_path: str, data_out: np.ndarray):
        # Both cases (cell_centered True/False) output the same number of data points
        n = self.dimension.size(cell_centered=False)
        dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((('f', n),)))

        load_times = False
        if self.parent_slice.n_t == -1:
            load_times = True
            self.parent_slice.n_t = (os.stat(
                file_path).st_size - self._offset) // dtype_data.itemsize
            self.parent_slice.times = np.empty(self.parent_slice.n_t)

        with open(file_path, 'rb') as infile:
            infile.seek(self._offset)
            for t, data in enumerate(fdtype.read(infile, dtype_data, self.parent_slice.n_t)):
                if load_times:
                    self.parent_slice.times[t] = data[0][0]
                data = data[1].reshape(self.dimension.shape(cell_centered=False), order='F')
                if self.parent_slice.cell_centered:
                    data_out[t, :] = data[:-1, :-1]  # Ignore ghost points
                else:
                    data_out[t, :] = data

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the slice's data.
        """
        if not hasattr(self, "_data"):
            file_path = os.path.join(self.parent_slice.root_path, self.filename)
            self._data = np.empty((self.parent_slice.n_t,) + self.shape, dtype=np.float32)
            self._load_data(file_path, self._data)
        return self._data

    @property
    def vector_data(self) -> Dict[str, np.ndarray]:
        """Method to lazy load the slice's vector data if it exists.
        """
        if not hasattr(self, "_vector_data"):
            raise AttributeError("There is no vector data available for this slice.")
        if len(self._vector_data) == 0:
            for direction in self.vector_filenames.keys():
                file_path = os.path.join(self.parent_slice.root_path,
                                         self.vector_filenames[direction])
                self._vector_data[direction] = np.empty((self.parent_slice.n_t,) + self.shape,
                                                        dtype=np.float32)
                self._load_data(file_path, self._vector_data[direction])
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
    :ivar n_t: Total number of time steps for which output data has been written.
    :ivar orientation: Orientation [1,2,3] of the slice in case it is 2D, 0 otherwise.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    """

    def __init__(self, root_path: str, slice_id: str, cell_centered: bool, times: np.ndarray,
                 multimesh_data: Collection[Dict]):
        self.root_path = root_path
        self.cell_centered = cell_centered

        self.times = times

        if times is not None:
            self.n_t = times.size
        else:
            self.n_t = -1

        self.id = slice_id

        # List of all subslices this slice consists of (one per mesh).
        self._subslices: Dict[Mesh, SubSlice] = dict()

        vector_temp = dict()
        for mesh_data in multimesh_data:
            if "-VELOCITY" in mesh_data["quantity"]:
                vector_temp[mesh_data["mesh"]] = dict()

        for mesh_data in multimesh_data:
            if mesh_data["mesh"] not in self._subslices:
                self.quantity = Quantity(mesh_data["quantity"], mesh_data["label"],
                                         mesh_data["unit"])
                self._subslices[mesh_data["mesh"]] = SubSlice(self, mesh_data["filename"],
                                                              mesh_data["dimension"],
                                                              mesh_data["extent"],
                                                              mesh_data["mesh"])

            if "-VELOCITY" in mesh_data["quantity"]:
                vector_temp[mesh_data["mesh"]][mesh_data["quantity"]] = mesh_data["filename"]

        for mesh, vector_filenames in vector_temp.items():
            if "U-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["u"] = vector_filenames["U-VELOCITY"]
            if "V-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["v"] = vector_filenames["V-VELOCITY"]
            if "W-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["w"] = vector_filenames["W-VELOCITY"]

        # Iterate over all subslices and remove duplicated ones which will be created when the slice
        # cuts exactly through two mesh borders. Only required for non-cell-centered slices.
        if not self.cell_centered:
            extents_tmp = list()
            remove_tmp = list()
            for mesh, sslc in self._subslices.items():
                if sslc.extent not in extents_tmp:
                    extents_tmp.append(sslc.extent)
                else:
                    remove_tmp.append(mesh)
            for mesh in remove_tmp:
                del self._subslices[mesh]

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

    def get_subslice(self, key: Union[int, Mesh]) -> SubSlice:
        """Returns the :class:`SubSlice` that cuts through the given mesh. When an int is provided
            the nth SubSlice will be returned.
        """
        return self[key]

    def __getitem__(self, key: Union[int, Mesh]) -> SubSlice:
        """Returns the :class:`SubSlice` that cuts through the given mesh. When an int is provided
            the nth SubSlice will be returned.
        """
        if type(key) == int:
            return tuple(self._subslices.values())[key]
        return self._subslices[key]

    def __len__(self):
        return len(self._subslices)

    @property
    def subslices(self) -> List[SubSlice]:
        """Get a list with all SubSlices.
        """
        return list(self._subslices.values())

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

    def get_nearest_timestep(self, time: float) -> int:
        """Calculates the nearest timestep for which data has been output for this slice.
        """
        idx = np.searchsorted(self.times, time, side="left")
        if time > 0 and (idx == len(self.times) or np.math.fabs(
                time - self.times[idx - 1]) < np.math.fabs(time - self.times[idx])):
            return idx - 1
        else:
            return idx

    def get_nearest_index(self, dimension: Literal['x', 'y', 'z'], value: float) -> int:
        """Get the nearest mesh coordinate index in a specific dimension.
        """
        coords = self.coordinates[dimension]
        idx = np.searchsorted(coords, value, side="left")
        if idx > 0 and (idx == coords.size or np.math.fabs(value - coords[idx - 1]) < np.math.fabs(
                value - coords[idx])):
            return idx - 1
        else:
            return idx

    @property
    def meshes(self) -> List[Mesh]:
        """Returns a list of all meshes this slice cuts through.
        """
        return list(self._subslices.keys())

    @property
    def coordinates(self) -> Dict[Literal['x', 'y', 'z'], np.ndarray]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
            For cell-centered slices, the coordinates are adjusted to represent cell-centered coordinates.
        """
        coords = {'x': set(), 'y': set(), 'z': set()}
        for dim in ('x', 'y', 'z'):
            for mesh in self._subslices.keys():
                co = mesh.coordinates[dim]
                # In case the slice is cell-centered, we will shift the coordinates by half a cell
                # and remove the last coordinate
                if self.cell_centered:
                    co = co[:-1]
                    co += (co[1] - co[0]) / 2
                coords[dim].update(co)
            coords[dim] = np.array(sorted(list(coords[dim])))

        return coords

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subslice in self._subslices.values():
            subslice.clear_cache()

    def sort_subslices_cartesian(self):
        """Returns all subslices sorted in cartesian coordinates (2D-slices only).
        """
        assert self.type == '2D', "The sort_subslices_cartesian method only works on 2D-slices!"

        slices = list(self._subslices.values())
        slices_cart = [[slices[0]]]
        orientation = abs(slices[0].orientation)
        if orientation == 1:  # x
            slices.sort(key=lambda p: (p.extent.y_start, p.extent.z_start))
        elif orientation == 2:  # y
            slices.sort(key=lambda p: (p.extent.x_start, p.extent.z_start))
        elif orientation == 3:  # z
            slices.sort(key=lambda p: (p.extent.x_start, p.extent.y_start))

        if orientation == 1:
            for slc in slices[1:]:
                if slc.extent.y_start == slices_cart[-1][-1].extent.y_start:
                    slices_cart[-1].append(slc)
                else:
                    slices_cart.append([slc])
        else:
            for slc in slices[1:]:
                if slc.extent.x_start == slices_cart[-1][-1].extent.x_start:
                    slices_cart[-1].append(slc)
                else:
                    slices_cart.append([slc])
        return slices_cart

    def to_global(self) -> np.ndarray:
        """Creates a global 2D-numpy ndarray from all subslices (2D-slices only).
            Note: This method will only return valid output if evenly sized and spaced meshes were
            used in the simulation. Other cases will require individual custom combination logic.
        """
        assert self.type == '2D', "The to_global method only works on 2D-slices!"

        slices = self.sort_subslices_cartesian()

        shape_dim1 = sum([slice_row[0].shape[0] for slice_row in slices])
        if not self.cell_centered:
            shape_dim1 -= len(slices) - 1
        shape_dim2 = sum([slc.shape[1] for slc in slices[0]])
        if not self.cell_centered:
            shape_dim2 -= len(slices[0]) - 1
        slc_array = np.empty(shape=(self.n_t, shape_dim1, shape_dim2))
        dim1_pos = 0
        dim2_pos = 0
        for i, slice_row in enumerate(slices):
            d1 = slice_row[0].shape[0]
            if not self.cell_centered:
                d1 -= 1 if i + 1 != len(slices) else 0
            for j, slc in enumerate(slice_row):
                d2 = slc.shape[1]
                if not self.cell_centered:
                    d2 -= 1 if j + 1 != len(slice_row) else 0
                slc_array[:, dim1_pos:dim1_pos + d1, dim2_pos:dim2_pos + d2] = slc.data[:, :d1, :d2]
                dim2_pos += d2
            dim1_pos += d1
            dim2_pos = 0
        return slc_array

    @property
    def type(self) -> Literal['2D', '3D']:
        if self.orientation == 0:
            return '3D'
        return '2D'

    @implements(np.mean)
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
            " required, use the 'to_global' method and use the returned numpy-array explicitly.")

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
        if self.type == '3D':  # 3D-Slice
            return f"Slice([3D] cell_centered={self.cell_centered}, extent={self.extent})"
        else:  # 2D-Slice
            return f"Slice([2D] cell_centered={self.cell_centered}, extent={self.extent}, extent_dirs={self.extent_dirs}, orientation={self.orientation})"

# __array_function__ implementations
