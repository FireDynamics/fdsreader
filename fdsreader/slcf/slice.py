import logging
import math
import os
from copy import deepcopy
from typing import Collection, Dict, List, Tuple, Union

import numpy as np
from typing_extensions import Literal

import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings
from fdsreader.fds_classes import Mesh
from fdsreader.utils import Dimension, Extent, NumpyArrayMixin, Quantity
from fdsreader.utils.data import resample_mesh_axis
from fdsreader.utils.extent import extent_index_range

_HANDLED_FUNCTIONS = {}


def implements(np_function):
    """Decorator to register an __array_function__ implementation for Slices."""

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

    _offset = 3 * fdtype.new((("c", 30),)).itemsize + fdtype.new((("i", 6),)).itemsize

    def __init__(
        self,
        parent_slc,
        filename: str,
        dimension: Dimension,
        extent: Extent,
        mesh: Mesh,
        bound_indices: Tuple[int, int, int, int, int, int] | None = None,
    ):
        self._parent_slice = parent_slc
        self.mesh = mesh
        self.dimension = dimension
        self.extent = extent
        # Node indices i1, i2, j1, j2, k1, k2 of the slice in its mesh, as in the .smv file
        self.bound_indices = bound_indices

        self.filename = filename

        self.vector_filenames = dict()
        self._vector_data = dict()

    def get_coordinates(self, ignore_cell_centered: bool = False) -> Dict[Literal["x", "y", "z"], np.ndarray]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
        For cell-centered slices, the coordinates can be adjusted to represent cell-centered coordinates.

        :param ignore_cell_centered: Whether to shift the coordinates when the slice is cell_centered or not.
        """
        cell_centered = self.cell_centered and not ignore_cell_centered
        ranges = self.index_ranges(cell_centered)
        coords: Dict[Literal["x", "y", "z"], np.ndarray] = {}
        for dim in ("x", "y", "z"):
            co = self.mesh.coordinates[dim]
            # In case the slice is cell-centered, we will shift the coordinates by half a cell
            # and remove the last coordinate. Compute the per-cell offset before trimming, since a
            # mesh with only a single cell along this axis leaves just one coordinate afterwards.
            if cell_centered:
                co = co[:-1] + np.diff(co) / 2
            coords[dim] = co[ranges[dim]].copy()

        return coords

    def index_ranges(self, cell_centered: bool | None = None) -> Dict[Literal["x", "y", "z"], slice]:
        """Indices of the nodes, or of the cells, of the mesh that this subslice covers along each axis.

        FDS writes a cell-centred slice ``SLCC ... i1 i2`` for the cells between nodes i1 and i2, and a
        cell-centred plane ``SLCC ... k k`` for the cell below node k.

        :param cell_centered: Whether to give cells or nodes; by default as the slice is written.
        """
        if cell_centered is None:
            cell_centered = self.cell_centered
        ranges = dict()
        for axis, dim in enumerate(("x", "y", "z")):
            if self.bound_indices is None:  # not read from an .smv file
                ranges[dim] = extent_index_range(self.mesh.coordinates[dim], self.extent[dim], cell_centered)
                continue
            low, high = self.bound_indices[2 * axis], self.bound_indices[2 * axis + 1]
            if not cell_centered:
                ranges[dim] = slice(low, high + 1)
            elif high > low:
                ranges[dim] = slice(low, high)
            else:
                ranges[dim] = slice(max(low - 1, 0), max(low, 1))
        return ranges

    @property
    def shape(self) -> Tuple[int, int]:
        """2D-shape of the slice."""
        shape = self.dimension.shape(cell_centered=self.cell_centered)
        if self.orientation != 0:
            if len(shape) < 2:
                return shape[0], 1
            return shape[0], shape[1]
        return shape

    @property
    def orientation(self) -> Literal[0, 1, 2, 3]:
        """Orientation [1,2,3] of the slice in case it is 2D, 0 otherwise."""
        return self._parent_slice.orientation

    @property
    def cell_centered(self) -> bool:
        """Indicates whether centered positioning for data is used."""
        return self._parent_slice.cell_centered

    @property
    def times(self):
        return self._parent_slice.times

    @property
    def n_t(self) -> int:
        """Get the number of timesteps for which data was output."""
        return self._parent_slice.n_t

    def _load_data(self, file_path: str, data_out: np.ndarray):
        # Both cases (cell_centered True/False) output the same number of data points
        n = self.dimension.size(cell_centered=False)
        dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((("f", n),)))

        with open(file_path, "rb") as infile:
            infile.seek(self._offset)
            _, data_col = fdtype.read_columns(infile, dtype_data, self.n_t)

            # Every record's data field was written by Fortran in column-major (F) order;
            # reshaping with order="F" applies that per-record reshape across the whole (n_t, n)
            # block at once, without a Python-level loop.
            shape = self.dimension.shape(cell_centered=False)
            full = data_col.reshape((self.n_t,) + shape, order="F")
            if self.cell_centered:
                # Ignore ghost points on every axis (2 for 2D slices, 3 for 3D slices)
                data_out[:] = full[(slice(None),) + (slice(1, None),) * (full.ndim - 1)]
            else:
                data_out[:] = full

    def _load_times(self) -> np.ndarray:
        # Read in (only) the times for which SLCF data is available
        n = self.dimension.size(cell_centered=False)
        dtype_data = fdtype.combine(fdtype.FLOAT, fdtype.new((("f", n),)))

        file_path = os.path.join(self._parent_slice._root_path, self.filename)

        n_t = (os.stat(file_path).st_size - self._offset) // dtype_data.itemsize

        with open(file_path, "rb") as infile:
            infile.seek(self._offset)
            times_col, _ = fdtype.read_columns(infile, dtype_data, n_t)

        return times_col.reshape(-1)

    @property
    def data(self) -> np.ndarray:
        """Method to lazy load the slice's data."""
        if not hasattr(self, "_data"):
            file_path = os.path.join(self._parent_slice._root_path, self.filename)
            self._data = np.empty((self.n_t,) + self.shape, dtype=np.float32)
            self._load_data(file_path, self._data)
        return self._data

    @property
    def vector_data(self) -> Dict[str, np.ndarray]:
        """Method to lazy load the slice's vector data if it exists."""
        if not hasattr(self, "_vector_data"):
            raise AttributeError("There is no vector data available for this slice.")
        if len(self._vector_data) == 0:
            for direction in self.vector_filenames.keys():
                file_path = os.path.join(self._parent_slice._root_path, self.vector_filenames[direction])
                self._vector_data[direction] = np.empty((self.n_t,) + self.shape, dtype=np.float32)
                self._load_data(file_path, self._vector_data[direction])
        return self._vector_data

    @property
    def vmin(self):
        return np.min(self.data)

    @property
    def vmax(self):
        return np.max(self.data)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        if hasattr(self, "_data"):
            del self._data
            del self._vector_data
            self._vector_data = dict()

    def __repr__(self):
        return f"SubSlice(shape={self.shape}, mesh={self.mesh.id}, extent={self.extent})"


class Slice(NumpyArrayMixin):
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

    _handled_functions = _HANDLED_FUNCTIONS
    _HANDLED_FUNCTIONS[np.mean] = NumpyArrayMixin.mean
    _HANDLED_FUNCTIONS[np.std] = NumpyArrayMixin.std

    def _array_subitems(self):
        return self._subslices.values()

    def __init__(self, root_path: str, slice_id: str, cell_centered: bool, multimesh_data: Collection[Dict]):
        self._root_path = root_path
        self.cell_centered = cell_centered

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
                self._subslices[mesh_data["mesh"].id] = SubSlice(
                    self,
                    mesh_data["filename"],
                    mesh_data["dimension"],
                    mesh_data["extent"],
                    mesh_data["mesh"],
                    mesh_data.get("bound_indices"),
                )

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

        x_start, x_end, y_start, y_end, z_start, z_end = (
            min(subslices, key=lambda e: e.extent.x_start).extent.x_start,
            max(subslices, key=lambda e: e.extent.x_end).extent.x_end,
            min(subslices, key=lambda e: e.extent.y_start).extent.y_start,
            max(subslices, key=lambda e: e.extent.y_end).extent.y_end,
            min(subslices, key=lambda e: e.extent.z_start).extent.z_start,
            max(subslices, key=lambda e: e.extent.z_end).extent.z_end,
        )
        self.extent = Extent(
            x_start,
            x_start if self.orientation == 1 else x_end,
            y_start,
            y_start if self.orientation == 2 else y_end,
            z_start,
            z_start if self.orientation == 3 else z_end,
        )

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        # (this also triggers computing `times`/`n_t` below as a side effect).
        if not settings.LAZY_LOAD:
            for _, subslice in self._subslices.items():
                _ = subslice.data

    @property
    def times(self) -> np.ndarray:
        """Numpy array containing all times for which data has been recorded."""
        if not hasattr(self, "_times"):
            # Read in the available time steps, using arbitrarily the first sub-slice
            self._times = self.subslices[0]._load_times()
        return self._times

    @times.setter
    def times(self, value: np.ndarray):
        self._times = value

    @property
    def n_t(self) -> int:
        """Total number of time steps for which output data has been written."""
        if not hasattr(self, "_n_t"):
            self._n_t = len(self.times)
        return self._n_t

    @n_t.setter
    def n_t(self, value: int):
        self._n_t = value

    def get_subslice(self, key: Union[int, str, Mesh]) -> SubSlice:
        """Returns the :class:`SubSlice` that cuts through the given mesh. When an int is provided
        the nth SubSlice will be returned.
        """
        return self[key]

    def __getitem__(self, key: Union[int, str, Mesh]) -> SubSlice:
        """Returns the :class:`SubSlice` that cuts through the given mesh. When an int is provided
        the nth SubSlice will be returned.
        """
        if isinstance(key, int):
            return tuple(self._subslices.values())[key]
        elif isinstance(key, str):
            key = tuple(mesh for mesh in self.meshes if mesh.id == key)[0]
        return self._subslices[key.id]

    def __len__(self):
        return len(self._subslices)

    @property
    def subslices(self) -> List[SubSlice]:
        """Get a list with all SubSlices."""
        return list(self._subslices.values())

    @property
    def extent_dirs(self) -> Tuple[Literal["x", "y", "z"], Literal["x", "y", "z"], Literal["x", "y", "z"]]:
        """The directions in which there is an extent. All three dimensions in case the slice is 3D."""
        ior = self.orientation
        if ior == 0:
            return "x", "y", "z"
        elif ior == 1:
            return "y", "z"
        elif ior == 2:
            return "x", "z"
        else:
            return "x", "y"

    def get_nearest_timestep(self, time: float) -> int:
        """Calculates the nearest timestep for which data has been output for this slice."""
        idx = np.searchsorted(self.times, time, side="left")
        if time > 0 and (
            idx == len(self.times) or math.fabs(time - self.times[idx - 1]) < math.fabs(time - self.times[idx])
        ):
            return idx - 1
        else:
            return idx

    def get_nearest_index(self, dimension: Literal["x", "y", "z"], value: float) -> int:
        """Get the nearest mesh coordinate index in a specific dimension."""
        coords = self.get_coordinates()[dimension]
        idx = np.searchsorted(coords, value, side="left")
        if idx > 0 and (idx == coords.size or math.fabs(value - coords[idx - 1]) < math.fabs(value - coords[idx])):
            return idx - 1
        else:
            return idx

    @property
    def meshes(self) -> List[Mesh]:
        """Returns a list of all meshes this slice cuts through."""
        return [subslc.mesh for subslc in self._subslices.values()]

    def get_coordinates(self, ignore_cell_centered: bool = False) -> Dict[Literal["x", "y", "z"], np.ndarray]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
        For cell-centered slices, the coordinates can be adjusted to represent cell-centered coordinates.

        :param ignore_cell_centered: Whether to shift the coordinates when the slice is cell_centered or not.
        """
        orientation = ("x", "y", "z")[self.orientation - 1] if self.orientation != 0 else ""
        coords = {"x": list(), "y": list(), "z": list()}
        for dim in ("x", "y", "z"):
            if orientation == dim:
                first = next(iter(self._subslices.values()))
                coords[dim] = first.get_coordinates(ignore_cell_centered)[dim][:1]
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
            # Every subslice has at least one coordinate on each axis (SubSlice.get_coordinates falls back to the
            # nearest one), so this is only empty for a slice without subslices

        return coords

    @property
    def vmin(self):
        return np.min(self)

    @property
    def vmax(self):
        return np.max(self)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        for subslice in self._subslices.values():
            subslice.clear_cache()

    def get_2d_slice_from_3d(self, slice_direction: Literal["x", "y", "z", 1, 2, 3], value: float):
        """Creates a 2D-Slice from a 3D-Slice by slicing the Slice.
        :param slice_direction: The direction in which to cut through the slice.
        :param value: The position at which to start cutting through the slice.
        """
        if self.type == "2D":
            logging.error("Trying to slice a 2D-slice, which might not yield the desired result.")

        if isinstance(slice_direction, str):
            slice_dim = {"x": 1, "y": 2, "z": 3}[slice_direction]
        else:
            slice_dim = slice_direction
            slice_direction = ("x", "y", "z")[slice_dim - 1]
        new_slice = deepcopy(self)

        to_remove = list()
        for mesh_id, subslc in new_slice._subslices.items():
            mesh = subslc.mesh
            # Check if the new slice cuts through the subslice
            cut_index = mesh.coordinate_to_index((value,), (slice_direction,), cell_centered=self.cell_centered)
            cut_value = mesh.get_nearest_coordinate((value,), (slice_direction,), cell_centered=self.cell_centered)[0]
            if (
                mesh.coordinates[slice_direction][0] <= value <= mesh.coordinates[slice_direction][-1]
                and subslc.extent[slice_dim][0] <= cut_value <= subslc.extent[slice_dim][1]
            ):
                shape = subslc.dimension.shape(cell_centered=self.cell_centered)
                indices = [slice(None)]

                if subslc.dimension.x == 1 or subslc.dimension.y == 1 or subslc.dimension.z == 1:
                    indices.extend((slice(shape[0]), slice(shape[1])))
                else:
                    indices.extend((slice(shape[0]), slice(shape[1]), slice(shape[2])))
                    indices[slice_dim] = slice(cut_index[0], cut_index[0] + 1, 1)

                subslc._data = np.squeeze(subslc.data[tuple(indices)], axis=slice_dim)
                for direction in subslc.vector_filenames.keys():
                    subslc._vector_data[direction] = np.squeeze(
                        subslc.vector_data[direction][tuple(indices)], axis=slice_dim
                    )

                # Change 3D extent to 2D one
                new_extent = subslc.extent.as_list(reduced=False)
                new_extent[(slice_dim - 1) * 2 : slice_dim * 2] = (cut_value, cut_value)
                subslc.extent = Extent(*new_extent)
            else:
                to_remove.append(mesh_id)

        for mesh_id in to_remove:
            del new_slice._subslices[mesh_id]

        new_slice.orientation = slice_dim
        subslices = new_slice._subslices.values()
        x_start, x_end, y_start, y_end, z_start, z_end = (
            min(subslices, key=lambda e: e.extent.x_start).extent.x_start,
            max(subslices, key=lambda e: e.extent.x_end).extent.x_end,
            min(subslices, key=lambda e: e.extent.y_start).extent.y_start,
            max(subslices, key=lambda e: e.extent.y_end).extent.y_end,
            min(subslices, key=lambda e: e.extent.z_start).extent.z_start,
            max(subslices, key=lambda e: e.extent.z_end).extent.z_end,
        )
        new_slice.extent = Extent(
            x_start,
            x_start if new_slice.orientation == 1 else x_end,
            y_start,
            y_start if new_slice.orientation == 2 else y_end,
            z_start,
            z_start if new_slice.orientation == 3 else z_end,
        )

        return new_slice

    def sort_subslices_cartesian(self):
        """Returns all subslices sorted in cartesian coordinates."""
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

    def to_global(
        self, masked: bool = False, fill: float = 0, return_coordinates: bool = False
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, Dict[Literal["x", "y", "z"], np.ndarray]],
        Tuple[
            Tuple[np.ndarray, np.ndarray],
            Tuple[Dict[Literal["x", "y", "z"], np.ndarray], Dict[Literal["x", "y", "z"], np.ndarray]],
        ],
    ]:
        """Creates a global numpy ndarray from all subslices (works for 2D- and 3D-slices).
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
        :param return_coordinates: If true, return the matching coordinate for each value on the generated grid.
        """
        if len(self._subslices) == 0:
            if return_coordinates:
                return np.array([]), {d: np.array([]) for d in ("x", "y", "z")}
            else:
                return np.array([])

        subslice_sets = [dict(), dict()]

        dimension = ["x", "y", "z"][self.orientation - 1]
        base_coord = next(iter(self._subslices.values())).get_coordinates(ignore_cell_centered=False)[dimension][0]

        for mesh, slc in self._subslices.items():
            coords = slc.get_coordinates(ignore_cell_centered=False)
            if self.orientation == 0 or np.isclose(coords[dimension][0], base_coord):
                subslice_sets[0][mesh] = slc
            else:
                subslice_sets[1][mesh] = slc
        if len(subslice_sets[1]) == 0:
            subslice_sets.pop(1)

        first_result_grid = None
        first_result_coordinates = None
        for subslices in subslice_sets:
            coord_min = {"x": math.inf, "y": math.inf, "z": math.inf}
            coord_max = {"x": -math.inf, "y": -math.inf, "z": -math.inf}
            # The nodes each subslice covers: a slice defined with XB may cover only part of its mesh
            nodes = {mesh: slc.get_coordinates(ignore_cell_centered=True) for mesh, slc in subslices.items()}
            for dim in ("x", "y", "z"):
                for mesh_nodes in nodes.values():
                    coord_min[dim] = min(mesh_nodes[dim][0], coord_min[dim])
                    coord_max[dim] = max(mesh_nodes[dim][-1], coord_max[dim])

            # The global grid will use the finest mesh as base and duplicate values of the coarser
            # meshes. Therefore, we first find the finest mesh and calculate the step size in each
            # dimension.
            step_sizes_min = {
                "x": coord_max["x"] - coord_min["x"],
                "y": coord_max["y"] - coord_min["y"],
                "z": coord_max["z"] - coord_min["z"],
            }
            step_sizes_max = {"x": 0, "y": 0, "z": 0}
            steps = dict()

            for dim in ("x", "y", "z"):
                for sslc in subslices.values():
                    step_size = sslc.mesh.coordinates[dim][1] - sslc.mesh.coordinates[dim][0]
                    step_sizes_min[dim] = min(step_size, step_sizes_min[dim])
                    step_sizes_max[dim] = max(step_size, step_sizes_max[dim])

            for dim in ("x", "y", "z"):
                if step_sizes_min[dim] == 0:
                    step_sizes_min[dim] = math.inf
                    steps[dim] = 1
                else:
                    steps[dim] = max(int(round((coord_max[dim] - coord_min[dim]) / step_sizes_min[dim])), 1) + (
                        0 if self.cell_centered else 1
                    )  # + step_sizes_max[dim] / step_sizes_min[dim]

            grid = np.full((self.n_t, steps["x"], steps["y"], steps["z"]), np.nan)

            def place(mesh: str, slc: SubSlice):
                """Maps a subslice onto the global grid, keeping the node rows on all of its upper faces.

                :returns: The data (masked if requested), the obstruction mask (None if not masked), the region of
                    the grid the data covers and the axes on which the subslice's upper face lies inside the
                    simulation space. Cell-centered subslices do not overlap, so no axis is reported for them.
                """
                slc_data = (
                    slc.data.copy() if slc.orientation == 0 else np.expand_dims(slc.data.copy(), axis=slc.orientation)
                )
                mask = slc.mesh.get_obstruction_mask_slice(slc) if masked else None

                slc_nodes = nodes[mesh]
                start_idx = dict()
                end_idx = dict()
                inner_axes = list()
                for axis in (0, 1, 2):
                    dim = ("x", "y", "z")[axis]
                    if axis == slc.orientation - 1:
                        start_idx[dim] = 0
                        end_idx[dim] = 1
                        continue
                    n_repeat = max(
                        int(round((slc.mesh.coordinates[dim][1] - slc.mesh.coordinates[dim][0]) / step_sizes_min[dim])),
                        1,
                    )

                    start_idx[dim] = int(round((slc_nodes[dim][0] - coord_min[dim]) / step_sizes_min[dim]))
                    end_idx[dim] = int(round((slc_nodes[dim][-1] - coord_min[dim]) / step_sizes_min[dim]))

                    if self.cell_centered:
                        if n_repeat > 1:
                            slc_data = np.repeat(slc_data, n_repeat, axis=axis + 1)
                            if masked:
                                mask = np.repeat(mask, n_repeat, axis=axis + 1)
                        continue

                    end_idx[dim] += 1
                    if not np.isclose(slc_nodes[dim][-1], coord_max[dim]):
                        inner_axes.append(axis)
                    slc_data = resample_mesh_axis(slc_data, axis + 1, n_repeat, keep_last=True)
                    if masked:
                        mask = resample_mesh_axis(mask, axis + 1, n_repeat, keep_last=True)

                region = tuple(slice(start_idx[dim], end_idx[dim]) for dim in ("x", "y", "z"))
                shape = (self.n_t,) + tuple(end_idx[dim] - start_idx[dim] for dim in ("x", "y", "z"))
                # If the slice should be masked, we set all cells at which an obstruction is in the
                # simulation space to the fill value set by the user
                if masked:
                    mask = mask.reshape(shape)
                    slc_data = np.where(mask, slc_data.reshape(shape), fill)
                return slc_data.reshape(shape), mask, region, inner_axes

            # Grid nodes some subslice has written, and the node rows of upper faces that lie inside the
            # simulation space, which were left to a neighbouring mesh (node-based slices only)
            written = None if self.cell_centered else np.zeros(grid.shape[1:], dtype=bool)
            inner_faces = list()
            for mesh, slc in subslices.items():
                slc_data, mask, region, inner_axes = place(mesh, slc)

                # The node row on an upper face inside the simulation space coincides with the first row of the
                # neighbouring mesh, which supplies it, so it is left out here. A copy of it is kept for the case
                # that no mesh does (np.take copies, so the rest of the subslice's data can be freed).
                for axis in inner_axes:
                    last = region[axis].stop - 1
                    face_region = tuple(slice(last, last + 1) if a == axis else r for a, r in enumerate(region))
                    face_mask = None if mask is None else np.take(mask, [-1], axis=axis + 1)
                    inner_faces.append((np.take(slc_data, [-1], axis=axis + 1), face_mask, face_region))

                    region = tuple(slice(r.start, last) if a == axis else r for a, r in enumerate(region))
                    without_face = (slice(None),) * (axis + 1) + (slice(None, -1),)
                    slc_data = slc_data[without_face]
                    if mask is not None:
                        mask = mask[without_face]

                grid[(slice(None),) + region] = slc_data
                if written is not None:
                    written[region] = True

            # On a domain that does not fill its bounding box (e.g. L-shaped), no neighbouring mesh supplies some
            # of those rows. Take their nodes from the mesh itself. Checking written rather than NaN keeps NaNs a
            # mesh did write, e.g. masked obstructions with fill=np.nan. Where several meshes supply a node, the
            # first one's value is used, and the node is masked if any of them masks it.
            if inner_faces:
                recovered = np.zeros_like(written)
                for face_data, face_mask, face_region in inner_faces:
                    missing = ~written[face_region]
                    if not missing.any():
                        continue
                    grid_face = grid[(slice(None),) + face_region]
                    first = missing & ~recovered[face_region]
                    grid_face[:, first] = face_data[:, first]
                    recovered[face_region] |= first
                    if face_mask is not None:
                        grid_face[:, missing] = np.where(face_mask[:, missing], grid_face[:, missing], fill)

            if return_coordinates:
                coordinates = dict()
                first = next(iter(subslices.values()))
                for dim_index, dim in enumerate(("x", "y", "z")):
                    if self.cell_centered and dim_index == self.orientation - 1:
                        # The cell layer of the plane, see SubSlice.index_ranges
                        coordinates[dim] = first.get_coordinates()[dim][:1]
                    elif self.cell_centered:
                        half = step_sizes_min[dim] / 2
                        coordinates[dim] = np.linspace(
                            coord_min[dim] + half, coord_max[dim] - half, grid.shape[dim_index + 1]
                        )
                    else:
                        coordinates[dim] = np.linspace(coord_min[dim], coord_max[dim], grid.shape[dim_index + 1])

            # Remove dimension corresponding to orientation for 2D slices
            if self.orientation != 0:
                grid = np.squeeze(grid, axis=self.orientation)

            # If we are returning two slices and the first of the two has been calculated, save that
            # data into intermediate variables and calculate the second array first before returning
            if first_result_grid is not None:
                if return_coordinates:
                    return (first_result_grid, grid), (first_result_coordinates, coordinates)
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
    def type(self) -> Literal["2D", "3D"]:
        if self.orientation == 0:
            return "3D"
        return "2D"

    @implements(np.amin)
    @implements(np.min)
    def _min(self):
        return min(subsclice.vmin for subsclice in self._subslices.values())

    @implements(np.amax)
    @implements(np.max)
    def _max(self):
        return max(subsclice.vmax for subsclice in self._subslices.values())

    def __repr__(self):
        if self.type == "3D":  # 3D-Slice
            return f"Slice([3D] quantity={self.quantity}, cell_centered={self.cell_centered}, extent={self.extent})"
        else:  # 2D-Slice
            return (
                f"Slice([2D] quantity={self.quantity}, cell_centered={self.cell_centered}, extent={self.extent}, "
                f"extent_dirs={self.extent_dirs}, orientation={self.orientation})"
            )


# __array_function__ implementations
