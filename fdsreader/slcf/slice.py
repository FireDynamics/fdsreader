import math
import os
from copy import deepcopy

import numpy as np
import logging
from typing import Dict, Collection, Tuple, Union, List
from typing_extensions import Literal

from fdsreader.fds_classes import Mesh
from fdsreader.utils import Dimension, Quantity, Extent
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
        self._parent_slice = parent_slc
        self.mesh = mesh
        self.dimension = dimension
        self.extent = extent

        self.filename = filename

        self.vector_filenames = dict()
        self._vector_data = dict()

    def get_coordinates(self, ignore_cell_centered: bool = False) -> Dict[Literal['x', 'y', 'z'], np.ndarray]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
            For cell-centered slices, the coordinates can be adjusted to represent cell-centered coordinates.

            :param ignore_cell_centered: Whether to shift the coordinates when the slice is cell_centered or not.
        """
        # orientation = ('x', 'y', 'z')[self.orientation - 1] if self.orientation != 0 else ''
        # coords = {'x': set(), 'y': set(), 'z': set()}
        coords: Dict[Literal['x', 'y', 'z'], np.ndarray] = {}
        for dim in ('x', 'y', 'z'):
            co = self.mesh.coordinates[dim].copy()
            # In case the slice is cell-centered, we will shift the coordinates by half a cell
            # and remove the last coordinate
            if self.cell_centered and not ignore_cell_centered:
                co = co[:-1]
                co += abs(co[1] - co[0]) / 2

            coords[dim] = co[np.where(np.logical_and(co >= self.extent[dim][0], co <= self.extent[dim][1]))]
            if coords[dim].size == 0:
                coords[dim] = np.array([co[np.argmin(np.abs(co - self.extent[dim][0]))]])

        return coords

    @property
    def shape(self) -> Tuple[int, int]:
        """2D-shape of the slice.
        """
        shape = self.dimension.shape(cell_centered=self.cell_centered)
        if self.orientation != 0:
            if len(shape) < 2:
                return shape[0], 1
            return shape[0], shape[1]
        return shape

    @property
    def orientation(self) -> Literal[0, 1, 2, 3]:
        """Orientation [1,2,3] of the slice in case it is 2D, 0 otherwise.
        """
        return self._parent_slice.orientation

    @property
    def cell_centered(self) -> bool:
        """Indicates whether centered positioning for data is used.
        """
        return self._parent_slice.cell_centered

    @property
    def times(self):
        return self._parent_slice.times

    @property
    def n_t(self):
        return self._parent_slice.n_t

    def _load_data(self, file_path: str, data_out: np.ndarray):
        # Both cases (cell_centered True/False) output the same number of data points
        n = self.dimension.size(cell_centered=False)
        dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((('f', n),)))

        load_times = self.n_t == -1
        if load_times:
            self._parent_slice.n_t = (os.stat(file_path).st_size - self._offset) // dtype_data.itemsize
            self._parent_slice.times = np.empty(self.n_t)

        with open(file_path, 'rb') as infile:
            infile.seek(self._offset)
            for t, data in enumerate(fdtype.read(infile, dtype_data, self.n_t)):
                if load_times:
                    self.times[t] = data[0][0]
                data = data[1].reshape(self.dimension.shape(cell_centered=False), order='F')
                if self.cell_centered:
                    data_out[t, :] = data[1:, 1:]  # Ignore ghost points
                else:
                    data_out[t, :] = data

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the slice's data.
        """
        if not hasattr(self, "_data"):
            file_path = os.path.join(self._parent_slice._root_path, self.filename)
            self._data = np.empty((self.n_t,) + self.shape, dtype=np.float32)
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
                file_path = os.path.join(self._parent_slice._root_path,
                                         self.vector_filenames[direction])
                self._vector_data[direction] = np.empty((self.n_t,) + self.shape,
                                                        dtype=np.float32)
                self._load_data(file_path, self._vector_data[direction])
        return self._vector_data

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
            del self._vector_data
            self._vector_data = dict()

    def __repr__(self):
        return f"SubSlice(shape={self.shape}, mesh={self.mesh.id}, extent={self.extent})"


class Slice(np.lib.mixins.NDArrayOperatorsMixin):
    """Slice file data container including metadata. Consists of multiple subslices, one for each
        mesh the slice cuts through. In case a slice cuts right through the border of two meshes, the generated data
        would be duplicated. For edge-centered slices a random of both generated slices will be discarded as the data
        is completely identical. For cell-centered slices the data of both slices will be saved as :class:`SubSlice` s,
        therefore it might seem as if all slices would be duplicated, in reality however the slices might contain
        different data.

    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar quantity: Quantity object containing information about the quantity calculated for this
        slice with the corresponding short_name and unit.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar n_t: Total number of time steps for which output data has been written.
    :ivar orientation: Orientation [1,2,3] of the slice in case it is 2D, 0 otherwise.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    """

    def __init__(self, root_path: str, slice_id: str, cell_centered: bool, times: np.ndarray,
                 multimesh_data: Collection[Dict]):
        self._root_path = root_path
        self.cell_centered = cell_centered

        self.times = times

        if times is not None:
            self.n_t = times.size
        else:
            self.n_t = -1

        self.id = slice_id

        # List of all subslices this slice consists of (one per mesh).
        self._subslices: Dict[str, SubSlice] = dict()

        vector_temp = dict()
        for mesh_data in multimesh_data:
            if "-VELOCITY" in mesh_data["quantity"]:
                vector_temp[mesh_data["mesh"].id] = dict()

        for mesh_data in multimesh_data:
            if mesh_data["mesh"].id not in self._subslices:
                self.quantity = Quantity(mesh_data["quantity"], mesh_data["short_name"], mesh_data["unit"])
                self._subslices[mesh_data["mesh"].id] = SubSlice(self, mesh_data["filename"], mesh_data["dimension"],
                                                              mesh_data["extent"], mesh_data["mesh"])

            if "-VELOCITY" in mesh_data["quantity"]:
                vector_temp[mesh_data["mesh"].id][mesh_data["quantity"]] = mesh_data["filename"]

        for mesh, vector_filenames in vector_temp.items():
            if "U-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["u"] = vector_filenames["U-VELOCITY"]
            if "V-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["v"] = vector_filenames["V-VELOCITY"]
            if "W-VELOCITY" in vector_filenames:
                self._subslices[mesh].vector_filenames["w"] = vector_filenames["W-VELOCITY"]

        subslices = self._subslices.values()
        orientations = list()
        # Check if all subslices are 2D. If so, the whole slice is 2D, even though it would have a 3D bounding box as
        # meshes can have different resolutions, therefore the subslices will not necessarily align
        for subslice in subslices:
            if subslice.extent.x_start == subslice.extent.x_end:
                orientations.append(1)
            elif subslice.extent.y_start == subslice.extent.y_end:
                orientations.append(2)
            elif subslice.extent.z_start == subslice.extent.z_end:
                orientations.append(3)
            else:
                orientations.append(0)
        self.orientation = 0 if any(o != orientations[0] for o in orientations) else orientations[0]

        x_start, x_end, y_start, y_end, z_start, z_end = min(subslices, key=lambda e: e.extent.x_start).extent.x_start, \
                                                         max(subslices, key=lambda e: e.extent.x_end).extent.x_end, \
                                                         min(subslices, key=lambda e: e.extent.y_start).extent.y_start, \
                                                         max(subslices, key=lambda e: e.extent.y_end).extent.y_end, \
                                                         min(subslices, key=lambda e: e.extent.z_start).extent.z_start, \
                                                         max(subslices, key=lambda e: e.extent.z_end).extent.z_end
        self.extent = Extent(x_start, x_start if self.orientation == 1 else x_end,
                             y_start, y_start if self.orientation == 2 else y_end,
                             z_start, z_start if self.orientation == 3 else z_end)

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            for _, subslice in self._subslices.items():
                _ = subslice.data

    def get_subslice(self, key: Union[int, str, Mesh]) -> SubSlice:
        """Returns the :class:`SubSlice` that cuts through the given mesh. When an int is provided
            the nth SubSlice will be returned.
        """
        return self[key]

    def __getitem__(self, key: Union[int, str, Mesh]) -> SubSlice:
        """Returns the :class:`SubSlice` that cuts through the given mesh. When an int is provided
            the nth SubSlice will be returned.
        """
        if type(key) == int:
            return tuple(self._subslices.values())[key]
        elif type(key) == str:
            key = tuple(mesh for mesh in self.meshes if mesh.id == key)[0]
        return self._subslices[key.id]

    def __len__(self):
        return len(self._subslices)

    @property
    def subslices(self) -> List[SubSlice]:
        """Get a list with all SubSlices.
        """
        return list(self._subslices.values())

    @property
    def extent_dirs(self) -> Tuple[Literal['x', 'y', 'z'], Literal['x', 'y', 'z'], Literal['x', 'y', 'z']]:
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
        coords = self.get_coordinates()[dimension]
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
        return [subslc.mesh for subslc in self._subslices.values()]

    def get_coordinates(self, ignore_cell_centered: bool = False) -> Dict[Literal['x', 'y', 'z'], np.ndarray]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
            For cell-centered slices, the coordinates can be adjusted to represent cell-centered coordinates.

            :param ignore_cell_centered: Whether to shift the coordinates when the slice is cell_centered or not.
        """
        orientation = ('x', 'y', 'z')[self.orientation-1] if self.orientation != 0 else ''
        coords = {'x': list(), 'y': list(), 'z': list()}
        for dim in ('x', 'y', 'z'):
            if orientation == dim:
                coords[dim] = np.array([self.extent[dim][0]])
                continue
            for slc in self._subslices.values():
                co = slc.get_coordinates(ignore_cell_centered)[dim]
                coords[dim].extend(co)

            coords[dim] = np.sort(np.array(coords[dim]))
            # Remove duplicates, caused by either same coordinate values in a specific dimensions or
            # floating point inaccuraciesas floats get written out with limited precision from FDS
            # (sometimes only 6 decimal places, e.g. 1.0 and 1.000001)
            diff = coords[dim][1:] - coords[dim][:-1]
            coords[dim] = np.concatenate((coords[dim][:1], coords[dim][1:][diff > 0.000002]), axis=0)

            if len(coords[dim]) == 0:
                slice_coordinate = self.extent[dim][0]
                nearest_coordinate = np.inf
                for mesh in self._subslices.keys():
                    mesh_coords = mesh.coordinates[dim]
                    idx = np.searchsorted(mesh_coords, slice_coordinate, side="left")
                    if idx > 0 and (idx == mesh_coords.size or np.math.fabs(
                            slice_coordinate - mesh_coords[idx - 1]) < np.math.fabs(
                            slice_coordinate - mesh_coords[idx])):
                        idx = idx + 1
                    if mesh_coords[idx] - slice_coordinate < nearest_coordinate - slice_coordinate:
                        nearest_coordinate = mesh_coords[idx]
                coords[dim] = np.array([nearest_coordinate])

        return coords

    @property
    def vmin(self):
        return np.min(self)

    @property
    def vmax(self):
        return np.max(self)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subslice in self._subslices.values():
            subslice.clear_cache()

    def get_2d_slice_from_3d(self, slice_direction: Literal['x', 'y', 'z', 1, 2, 3], value: float):
        """Creates a 2D-Slice from a 3D-Slice by slicing the Slice.
        :param slice_direction: The direction in which to cut through the slice.
        :param value: The position at which to start cutting through the slice.
        """
        if self.type == "2D":
            logging.error("Trying to slice a 2D-slice, which might not yield the desired result.")

        if type(slice_direction) == str:
            slice_dim = {'x': 1, 'y': 2, 'z': 3}[slice_direction]
        else:
            slice_dim = slice_direction
            slice_direction = ('x', 'y', 'z')[slice_dim - 1]
        new_slice = deepcopy(self)

        to_remove = list()
        for mesh in new_slice._subslices.keys():
            subslc = new_slice._subslices[mesh]
            # Check if the new slice cuts through the subslice
            cut_index = mesh.coordinate_to_index((value,), (slice_direction,),
                                                 cell_centered=self.cell_centered)
            cut_value = mesh.get_nearest_coordinate((value,), (slice_direction,),
                                                    cell_centered=self.cell_centered)[0]
            if mesh.coordinates[slice_direction][0] <= value <= mesh.coordinates[slice_direction][-1] and \
                    subslc.extent[slice_dim - 1][0] <= cut_value <= subslc.extent[slice_dim - 1][1]:
                shape = subslc.dimension.shape(cell_centered=self.cell_centered)
                indices = [slice(subslc.n_t)]

                if subslc.dimension.x == 1 or subslc.dimension.y == 1 or subslc.dimension.z == 1:
                    indices.extend((slice(shape[0]), slice(shape[1])))
                else:
                    indices.extend((slice(shape[0]), slice(shape[1]), slice(shape[2])))
                    indices[slice_dim - 1] = slice(cut_index[0], cut_index[0] + 1, 1)

                subslc._data = subslc.data[tuple(indices)]
                for direction in subslc.vector_filenames.keys():
                    subslc._vector_data[direction] = subslc.vector_data[direction][tuple(indices)]
                subslc.extent._extents[slice_dim - 1] = (0, 0)
            else:
                to_remove.append(mesh)

        for mesh in to_remove:
            del new_slice._subslices[mesh]

        new_slice.orientation = slice_dim
        subslices = new_slice._subslices.values()
        x_start, x_end, y_start, y_end, z_start, z_end = min(subslices, key=lambda e: e.extent.x_start).extent.x_start, \
                                                         max(subslices, key=lambda e: e.extent.x_end).extent.x_end, \
                                                         min(subslices, key=lambda e: e.extent.y_start).extent.y_start, \
                                                         max(subslices, key=lambda e: e.extent.y_end).extent.y_end, \
                                                         min(subslices, key=lambda e: e.extent.z_start).extent.z_start, \
                                                         max(subslices, key=lambda e: e.extent.z_end).extent.z_end
        new_slice.extent = Extent(x_start, x_start if new_slice.orientation == 1 else x_end,
                                  y_start, y_start if new_slice.orientation == 2 else y_end,
                                  z_start, z_start if new_slice.orientation == 3 else z_end)

        return new_slice

    def sort_subslices_cartesian(self):
        """Returns all subslices sorted in cartesian coordinates.
        """
        slices = list(self._subslices.values())
        slices_cart = [[slices[0]]]
        orientation = abs(slices[0].orientation)
        if orientation == 0:
            # 3D-slices have an orientation of 0, so we have to find the orientation ourself
            if slices[0].extent.x_start == slices[1].extent.x_start:
                orientation = 1
            elif slices[0].extent.y_start == slices[1].extent.y_start:
                orientation = 2
            elif slices[0].extent.z_start == slices[1].extent.z_start:
                orientation = 3

        if orientation == 1:  # x
            slices.sort(key=lambda p: (p.extent.y_start, p.extent.z_start))
        elif orientation == 2:  # y
            slices.sort(key=lambda p: (p.extent.x_start, p.extent.z_start))
        else:  # z
            slices.sort(key=lambda p: (p.extent.x_start, p.extent.y_start))

        # Form a list of lists (2D-array) of the sorted slices
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

    def to_global(self, masked: bool = False, fill: float = 0, return_coordinates: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, Dict[Literal['x', 'y', 'z'], np.ndarray]], Tuple[np.ndarray, np.ndarray, Dict[Literal['x', 'y', 'z'], np.ndarray], Dict[Literal['x', 'y', 'z'], np.ndarray]]]:
        """Creates a global numpy ndarray from all subslices (only tested for 2D-slices).
            Note: This method might create a sparse np-array that consumes lots of memory.
            Attention: Two global slices are returned in cases where cell-centered slices cut right
            through one or more mesh borders. If there is a cell-centered slice that cuts right
            through the border of two meshes (mesh1 and mesh2), there will actually be two slices
            that could be equally relevant for the user. The one will cells on mesh1 and the one
            with cells on mesh2. As there are no cells in between (i.e. where the slice "should" be)
            and cell-centered slices output values at the centers of each cell, FDS simply outputs
            two slices, one on each side of the mesh borders. The fdsreader will not discard any
            data, both slices that are output by FDS are sent to the user for him to decide which
            one to use.

            :param masked: Whether to apply the obstruction mask to the slice or not.
            :param fill: The fill value to use for masked slice entries. Only used when masked=True.
            :param return_coordinates: If true, return the matching coordinate for each value on the
                generated grid.
        """
        subslice_sets = [dict(), dict()]

        dimension = ['x', 'y', 'z'][self.orientation - 1]
        base_coord = next(iter(self._subslices.values())).get_coordinates(ignore_cell_centered=False)[dimension][0]

        for mesh, slc in self._subslices.items():
            coords = slc.get_coordinates(ignore_cell_centered=False)
            if coords[dimension][0] == base_coord:
                subslice_sets[0][mesh] = slc
            else:
                subslice_sets[1][mesh] = slc
        if len(subslice_sets[1]) == 0:
            subslice_sets.pop(1)

        first_result_grid = None
        for subslices in subslice_sets:
            coord_min = {'x': math.inf, 'y': math.inf, 'z': math.inf}
            coord_max = {'x': -math.inf, 'y': -math.inf, 'z': -math.inf}
            for dim in ('x', 'y', 'z'):
                for slc in subslices.values():
                    co = slc.mesh.coordinates[dim]
                    co = co[np.where(np.logical_and(co >= slc.extent[dim][0], co <= slc.extent[dim][1]))]
                    coord_min[dim] = min(co[0], coord_min[dim])
                    coord_max[dim] = max(co[-1], coord_max[dim])

            # The global grid will use the finest mesh as base and duplicate values of the coarser meshes
            # Therefore we first find the finest mesh and calculate the step size in each dimension
            step_sizes_min = {'x': coord_max['x'] - coord_min['x'],
                          'y': coord_max['y'] - coord_min['y'],
                          'z': coord_max['z'] - coord_min['z']}
            step_sizes_max = {'x': 0, 'y': 0, 'z': 0}
            steps = dict()
            global_max = {'x': -math.inf, 'y': -math.inf, 'z': -math.inf}

            for dim in ('x', 'y', 'z'):
                for sslc in subslices.values():
                    step_size = sslc.mesh.coordinates[dim][1] - sslc.mesh.coordinates[dim][0]
                    step_sizes_min[dim] = min(step_size, step_sizes_min[dim])
                    step_sizes_max[dim] = max(step_size, step_sizes_max[dim])
                    global_max[dim] = max(sslc.mesh.coordinates[dim][-1], global_max[dim])

            for dim in ('x', 'y', 'z'):
                if step_sizes_min[dim] == 0:
                    step_sizes_min[dim] = math.inf
                    steps[dim] = 1
                else:
                    steps[dim] = max(int(round((coord_max[dim] - coord_min[dim]) / step_sizes_min[dim])), 1) + (0 if self.cell_centered else 1)  # + step_sizes_max[dim] / step_sizes_min[dim]

            grid = np.full((self.n_t, steps['x'], steps['y'], steps['z']), np.nan)

            start_idx = dict()
            end_idx = dict()
            for slc in subslices.values():
                slc_data = slc.data if slc.orientation == 0 else np.expand_dims(slc.data, axis=slc.orientation)
                for axis in (0, 1, 2):
                    dim = ('x', 'y', 'z')[axis]
                    if axis == slc.orientation - 1:
                        start_idx[dim] = 0
                        end_idx[dim] = 1
                        continue
                    n_repeat = max(int(round((slc.mesh.coordinates[dim][1] - slc.mesh.coordinates[dim][0]) / step_sizes_min[dim])), 1)

                    start_idx[dim] = int(round((slc.mesh.coordinates[dim][0] - coord_min[dim]) / step_sizes_min[dim]))
                    end_idx[dim] = int(round((slc.mesh.coordinates[dim][-1] - coord_min[dim]) / step_sizes_min[dim]))

                    # We ignore border points unless they are actually on the border of the simulation space as all
                    # other border points actually appear twice, as the subslices overlap. This only
                    # applies for face_centered slices, as cell_centered slices will not overlap.
                    if not self.cell_centered:
                        if axis != slc.orientation - 1:
                            reduced_shape = list(slc_data.shape)
                            reduced_shape[axis + 1] -= 1
                            reduced_data_slices = tuple(slice(s) for s in reduced_shape)
                            slc_data = slc_data[reduced_data_slices]

                        # Temporarily save border points to add them back to the array again later
                        if slc.mesh.coordinates[dim][-1] == global_max[dim]:
                            end_idx[dim] += 1
                            temp_data_slices = [slice(s) for s in slc_data.shape]
                            temp_data_slices[axis + 1] = slice(slc_data.shape[axis + 1] - 1, None)
                            temp_data = slc_data[tuple(temp_data_slices)]

                    if n_repeat > 1:
                        slc_data = np.repeat(slc_data, n_repeat, axis=axis + 1)

                    # Add border points back again if needed
                    if not self.cell_centered and slc.mesh.coordinates[dim][-1] == global_max[dim]:
                        slc_data = np.concatenate((slc_data, temp_data), axis=axis + 1)

                if masked:
                    mask = slc.mesh.get_obstruction_mask_slice(slc)
                    slc_data = np.where(mask, slc_data, fill)

                grid[:, start_idx['x']: end_idx['x'], start_idx['y']: end_idx['y'],
                start_idx['z']: end_idx['z']] = slc_data.reshape(
                    (self.n_t, end_idx['x'] - start_idx['x'], end_idx['y'] - start_idx['y'], end_idx['z'] - start_idx['z']))

            if return_coordinates:
                coordinates = dict()
                for dim_index, dim in enumerate(('x', 'y', 'z')):
                    coordinates[dim] = np.linspace(coord_min[dim], coord_max[dim], grid.shape[dim_index+1])

            # Remove empty dimensions, but make sure to note remove the time dimension if there is only a single timestep
            grid = np.squeeze(grid)
            if len(grid.shape) == 2:
                grid = grid[np.newaxis, :, :]

            if first_result_grid is not None:
                if return_coordinates:
                    return first_result_grid, grid, first_result_coordinates, coordinates
                else:
                    return first_result_grid, grid
            else:
                first_result_grid = grid
                if return_coordinates:
                    first_result_coordinates = coordinates

        if return_coordinates:
            return first_result_grid, first_result_coordinates
        else:
            return first_result_grid

    @property
    def type(self) -> Literal['2D', '3D']:
        if self.orientation == 0:
            return '3D'
        return '2D'

    @implements(np.amin)
    def _min(self):
        return min(subsclice.vmin for subsclice in self._subslices.values())

    @implements(np.amax)
    def _max(self):
        return max(subsclice.vmax for subsclice in self._subslices.values())

    @implements(np.mean)
    def mean(self):
        """Calculates the mean over the whole slice.

        :returns: The calculated mean value.
        """
        return np.mean([np.mean(subsclice.data) for subsclice in self._subslices.values()])

    @implements(np.std)
    def std(self):
        """Calculates the standard deviation over the whole slice.

        :returns: The calculated standard deviation.
        """
        mean = self.mean
        sum = np.sum([np.sum(np.power(subsclice.data - mean, 2)) for subsclice in self._subslices.values()])
        N = np.sum([subsclice.data.size for subsclice in self._subslices.values()])
        return np.sqrt(sum / N)

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
            return f"Slice([3D] quantity={self.quantity}, cell_centered={self.cell_centered}, extent={self.extent})"
        else:  # 2D-Slice
            return f"Slice([2D] quantity={self.quantity}, cell_centered={self.cell_centered}, extent={self.extent}, " \
                   f"extent_dirs={self.extent_dirs}, orientation={self.orientation})"

# __array_function__ implementations
