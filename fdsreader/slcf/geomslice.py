import os
from copy import deepcopy

import numpy as np
import logging
from typing import Dict, Collection, Union, List
from typing_extensions import Literal

from fdsreader.fds_classes import Mesh
from fdsreader.utils import Quantity, Extent
from fdsreader import settings
import fdsreader.utils.fortran_data as fdtype

_HANDLED_FUNCTIONS = {}


def implements(np_function):
    """Decorator to register an __array_function__ implementation for GeomSlices.
    """

    def decorator(func):
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


class SubGeomSlice:
    """Part of a geomslice that cuts through a single mesh.

    :ivar mesh: The mesh the subgeomslice cuts through.
    """

    def __init__(self, parent_slice, filename: str, geom_filename: str, extent: Extent, mesh: Mesh):
        self._parent_slice = parent_slice
        self.mesh = mesh
        self.extent = extent

        self.file_path = os.path.join(parent_slice._root_path, filename)
        self.geom_file_path = os.path.join(parent_slice._root_path, geom_filename)

        # self.vector_filenames = dict()
        # self._vector_data = dict()

    @property
    def orientation(self) -> Literal[1, 2, 3]:
        """Orientation [1,2,3] of the geomslice in case it is 2D, 0 otherwise.
        """
        return self._parent_slice.orientation

    @property
    def times(self):
        return self._parent_slice.times


    @property
    def n_t(self) -> int:
        """Get the number of timesteps for which data was output.
        """
        return self._parent_slice.n_t

    def _load_geom_data(self):
        with open(self.geom_file_path, 'rb') as infile:
            dtype_meta = fdtype.new((('i', 3),))
            infile.seek(2 * fdtype.INT.itemsize + dtype_meta.itemsize + fdtype.FLOAT.itemsize)
            self.n_verts, self.n_faces, n_vols = fdtype.read(infile, dtype_meta, 1)[0][0]

            if self.n_verts > 0 and self.n_faces > 0:
                dtype_verts = fdtype.new((('f', 3 * self.n_verts),))
                dtype_faces = fdtype.new((('i', 3 * self.n_faces),))
                # dtype_locations = fdtype.new((('i', self.n_faces),))
                # dtype_zero_floats = fdtype.new((('f', 3 * self.n_faces * 2),))
                self._vertices = fdtype.read(infile, dtype_verts, 1)[0][0].reshape((self.n_verts, 3), order='F')
                self._faces = fdtype.read(infile, dtype_faces, 1)[0][0].reshape((self.n_faces, 3), order='F')
            else:
                self._vertices = np.array([])
                self._faces = np.array([])

    def _load_data(self):
        with open(self.file_path, 'rb') as infile:
            infile.seek(2 * fdtype.INT.itemsize)

            if self.n_verts > 0 and self.n_faces > 0:
                dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((('i', 4),)), fdtype.new((('f', self.n_faces),)))
            else:
                dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((('i', 4),)))

            load_times = self.n_t == -1
            if load_times:
                self._parent_slice.n_t = (os.stat(self.file_path).st_size - 2 * fdtype.INT.itemsize) // dtype_data.itemsize
                self._parent_slice.times = np.empty(self.n_t)

            self._data = np.empty((self.n_t, self.n_faces), dtype=np.float32)

            if self.n_verts > 0 and self.n_faces > 0:
                for t, data in enumerate(fdtype.read(infile, dtype_data, self.n_t)):
                    if load_times:
                        self.times[t] = data[0][0]
                    self._data[t, :] = data[2]

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the geomslice's data.
        """
        if not hasattr(self, "_data"):
            _ = self.vertices  # Make sure geom data has been loaded already
            self._load_data()
        return self._data

    @property
    def vertices(self) -> np.ndarray:
        """Method to lazy load the geomslice's data.
        """
        if not hasattr(self, "_vertices"):
            self._load_geom_data()
        return self._vertices

    @property
    def faces(self) -> np.ndarray:
        """Method to lazy load the geomslice's data.
        """
        if not hasattr(self, "_faces"):
            self._load_geom_data()
        return self._faces

    # @property
    # def vector_data(self) -> Dict[str, np.ndarray]:
    #     """Method to lazy load the geomslice's vector data if it exists.
    #     """
    #     if not hasattr(self, "_vector_data"):
    #         raise AttributeError("There is no vector data available for this geomslice.")
    #     if len(self._vector_data) == 0:
    #         for direction in self.vector_filenames.keys():
    #             file_path = os.path.join(self._parent_slice._root_path,
    #                                      self.vector_filenames[direction])
    #             self._vector_data[direction] = np.empty((self.n_t,) + self.shape,
    #                                                     dtype=np.float32)
    #             self._load_data(file_path, self._vector_data[direction])
    #     return self._vector_data

    @property
    def vmin(self):
        return np.min(self.data)

    @property
    def vmax(self):
        return np.max(self.data)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "_data"):
            del self._data
            # del self._vector_data
            # self._vector_data = dict()

    def __repr__(self):
        return f"SubGeomSlice(mesh={self.mesh.id})"


class GeomSlice(np.lib.mixins.NDArrayOperatorsMixin):
    """Slice file data container including metadata. Consists of multiple subgeomslices, one for each
        mesh the geomslice cuts through.

    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar quantity: Quantity object containing information about the quantity calculated for this
        geomslice with the corresponding short_name and unit.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar n_t: Total number of time steps for which output data has been written.
    :ivar orientation: Orientation [1,2,3] of the geomslice in case it is 2D, 0 otherwise.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    """

    def __init__(self, root_path: str, geomslice_id: str, times: np.ndarray, multimesh_data: Collection[Dict]):
        self._root_path = root_path

        self.times = times

        if times is not None:
            self.n_t = times.size
        else:
            self.n_t = -1

        self.id = geomslice_id

        # List of all subgeomslices this geomslice consists of (one per mesh).
        self._subgeomslices: Dict[str, SubGeomSlice] = dict()

        vector_temp = dict()
        for mesh_data in multimesh_data:
            if "-VELOCITY" in mesh_data["quantity"]:
                vector_temp[mesh_data["mesh"].id] = dict()

        for mesh_data in multimesh_data:
            if mesh_data["mesh"].id not in self._subgeomslices:
                self.quantity = Quantity(mesh_data["quantity"], mesh_data["short_name"], mesh_data["unit"])
                self._subgeomslices[mesh_data["mesh"].id] = SubGeomSlice(self, mesh_data["filename"], mesh_data["geomfilename"], mesh_data["extent"], mesh_data["mesh"])

            # if "-VELOCITY" in mesh_data["quantity"]:
            #     vector_temp[mesh_data["mesh"]][mesh_data["quantity"]] = mesh_data["filename"]

        # for mesh, vector_filenames in vector_temp.items():
        #     if "U-VELOCITY" in vector_filenames:
        #         self._subgeomslices[mesh].vector_filenames["u"] = vector_filenames["U-VELOCITY"]
        #     if "V-VELOCITY" in vector_filenames:
        #         self._subgeomslices[mesh].vector_filenames["v"] = vector_filenames["V-VELOCITY"]
        #     if "W-VELOCITY" in vector_filenames:
        #         self._subgeomslices[mesh].vector_filenames["w"] = vector_filenames["W-VELOCITY"]

        # # Iterate over all subgeomslices and remove duplicated ones which will be created when the geomslice
        # # cuts exactly through two mesh borders. Only required for non-cell-centered geomslices.
        # if not self.cell_centered:
        #     extents_tmp = list()
        #     remove_tmp = list()
        #     for mesh, sslc in self._subgeomslices.items():
        #         if sslc.extent not in extents_tmp:
        #             extents_tmp.append(sslc.extent)
        #         else:
        #             remove_tmp.append(mesh)
        #     for mesh in remove_tmp:
        #         del self._subgeomslices[mesh]
        #
        vals = self._subgeomslices.values()
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
            for _, subgeomslice in self._subgeomslices.items():
                _ = subgeomslice.data

    def get_subgeomslice(self, key: Union[int, str, Mesh]) -> SubGeomSlice:
        """Returns the :class:`SubGeomSlice` that cuts through the given mesh. When an int is
            provided the nth SubGeomSlice will be returned.
        """
        return self[key]

    def __getitem__(self, key: Union[int, str, Mesh]) -> SubGeomSlice:
        """Returns the :class:`SubGeomSlice` that cuts through the given mesh. When an int is
            provided the nth SubGeomSlice will be returned.
        """
        if type(key) == int:
            return tuple(self._subgeomslices.values())[key]
        if type(key) == str:
            return self._subgeomslices[key]
        return self._subgeomslices[key.id]

    def __len__(self):
        return len(self._subgeomslices)

    @property
    def subgeomslices(self) -> List[SubGeomSlice]:
        """Get a list with all SubGeomSlices.
        """
        return list(self._subgeomslices.values())

    # @property
    # def extent_dirs(self) -> Tuple[Literal['x', 'y', 'z'], Literal['x', 'y', 'z'], Literal['x', 'y', 'z']]:
    #     """The directions in which there is an extent. All three dimensions in case the geomslice is 3D.
    #     """
    #     ior = self.orientation
    #     if ior == 0:
    #         return 'x', 'y', 'z'
    #     elif ior == 1:
    #         return 'y', 'z'
    #     elif ior == 2:
    #         return 'x', 'z'
    #     else:
    #         return 'x', 'y'

    def get_nearest_timestep(self, time: float) -> int:
        """Calculates the nearest timestep for which data has been output for this geomslice.
        """
        idx = np.searchsorted(self.times, time, side="left")
        if time > 0 and (idx == len(self.times) or np.math.fabs(
                time - self.times[idx - 1]) < np.math.fabs(time - self.times[idx])):
            return idx - 1
        else:
            return idx

    # def get_nearest_index(self, dimension: Literal['x', 'y', 'z'], value: float) -> int:
    #     """Get the nearest mesh coordinate index in a specific dimension.
    #     """
    #     coords = self.coordinates[dimension]
    #     idx = np.searchsorted(coords, value, side="left")
    #     if idx > 0 and (idx == coords.size or np.math.fabs(value - coords[idx - 1]) < np.math.fabs(
    #             value - coords[idx])):
    #         return idx - 1
    #     else:
    #         return idx

    @property
    def meshes(self) -> List[Mesh]:
        """Returns a list of all meshes this geomslice cuts through.
        """
        return [subgeomslc.mesh for subgeomslc in self._subgeomslices]

    # @property
    # def coordinates(self) -> Dict[Literal['x', 'y', 'z'], np.ndarray]:
    #     """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
    #         For cell-centered geomslices, the coordinates are adjusted to represent cell-centered coordinates.
    #     """
    #     coords = {'x': set(), 'y': set(), 'z': set()}
    #     for dim in ('x', 'y', 'z'):
    #         for mesh in self._subgeomslices.keys():
    #             co = mesh.coordinates[dim].copy()
    #             # In case the geomslice is cell-centered, we will shift the coordinates by half a cell
    #             # and remove the last coordinate
    #             if self.cell_centered:
    #                 co = co[:-1]
    #                 co -= (co[1] - co[0]) / 2
    #             coords[dim].update(co)
    #         coords[dim] = np.array(sorted(list(coords[dim])))
    #
    #     return coords

    @property
    def vmin(self):
        return np.min(self)

    @property
    def vmax(self):
        return np.max(self)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subgeomslice in self._subgeomslices.values():
            subgeomslice.clear_cache()

    @property
    def vertices(self):
        n_verts = sum(subgeomslice.vertices.shape[0] for subgeomslice in self._subgeomslices.values())
        vertices = np.empty((n_verts, 3))

        counter = 0
        for subgeomslice in self._subgeomslices.values():
            size = subgeomslice.vertices.shape[0]
            vertices[:, counter:counter+size] = subgeomslice.vertices
            counter += size

        return vertices

    @property
    def faces(self):
        n_faces = sum(x.faces.shape[0] for x in self._subgeomslices.values())
        faces = np.empty((n_faces, 3))

        counter = 0
        for subgeomslice in self._subgeomslices.values():
            size = subgeomslice.faces.shape[0]
            faces[:, counter:counter + size] = subgeomslice.faces
            counter += size

        return faces

    @property
    def data(self):
        n = sum(x.data.shape[1] for x in self._subgeomslices.values())
        data = np.empty((next(iter(self._subgeomslices.values())).n_t, n))

        counter = 0
        for subgeomslice in self._subgeomslices.values():
            size = subgeomslice.data.shape[1]
            data[:, counter:counter + size] = subgeomslice.data
            counter += size

        return data

    @implements(np.min)
    def _min(self):
        return min(subgeomsclice.vmin for subgeomsclice in self._subgeomslices.values())

    @implements(np.max)
    def _max(self):
        return max(subgeomsclice.vmax for subgeomsclice in self._subgeomslices.values())

    @implements(np.mean)
    def mean(self):
        """Calculates the mean over the whole geomslice.

        :returns: The calculated mean value.
        """
        return np.mean([np.mean(subgeomsclice.data) for subgeomsclice in self._subgeomslices.values()])

    @implements(np.std)
    def std(self):
        """Calculates the standard deviation over the whole geomslice.

        :returns: The calculated standard deviation.
        """
        mean = self.mean
        sum = np.sum([np.sum(np.power(subgeomsclice.data - mean, 2)) for subgeomsclice in self._subgeomslices.values()])
        N = np.sum([subgeomsclice.data.size for subgeomsclice in self._subgeomslices.values()])
        return np.sqrt(sum / N)

    def __array__(self):
        """Method that will be called by numpy when trying to convert the object to a numpy ndarray.
        """
        raise UserWarning(
            "Slices can not be converted to numpy arrays, but they support all typical numpy"
            " operations such as np.multiply. If a 'global' array containing all subgeomslices is"
            " required, use the 'to_global' method and use the returned numpy-array explicitly.")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Method that will be called by numpy when using a ufunction with a GeomSlice as input.

        :returns: A new geomslice on which the ufunc has been applied.
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
                f"The {method} operation is not implemented for multiple geomslices as input yet. If"
                " you require this feature, please request this functionality by submitting an"
                " issue on Github.")

        new_slice = deepcopy(self)
        for subgeomslice in new_slice._subgeomslices.values():
            subgeomslice._data = ufunc(subgeomslice.data, input_list[0], **kwargs)
        return new_slice

    def __array_function__(self, func, types, args, kwargs):
        """Method that will be called by numpy when using an array function with a GeomSlice as input.

        :returns: The output of the array function.
        """
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
            # Note: this allows subclasses that don't override __array_function__ to handle GeomSlices.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __repr__(self):
        # if self.type == '3D':  # 3D-Slice
        #     return f"GeomSlice([3D] quantity={self.quantity}, cell_centered={self.cell_centered}, extent={self.extent})"
        # else:  # 2D-Slice
        #     return f"GeomSlice([2D] quantity={self.quantity}, cell_centered={self.cell_centered}, extent={self.extent}, " \
        #            f"extent_dirs={self.extent_dirs}, orientation={self.orientation})"
        return f"GeomSlice(quantity={self.quantity})"

# __array_function__ implementations
