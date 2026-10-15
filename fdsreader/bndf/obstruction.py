import math
import os
import warnings

# Unfortunately, this is necessary due to a cyclic reference. "Mesh" is only needed for static type hints anyway
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Literal

import fdsreader.utils.fortran_data as fdtype
from fdsreader.utils import Dimension, Extent, Quantity
from fdsreader.utils.misc import caller_stacklevel

if TYPE_CHECKING:
    from fdsreader.fds_classes import Mesh


class Patch:
    """Container for the actual data which is stored as rectangular plane with specific orientation
        and extent.

    :ivar dimension: :class:`Dimension` object containing information about steps in each dimension.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar orientation: The direction the patch is facing (x={-1;1}, y={-2;2}, z={-3;3}).
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar _n_t: Total number of time steps for which output data has been written.
    """

    def __init__(
        self,
        file_path: str,
        dimension: Dimension,
        extent: Extent,
        orientation: int,
        cell_centered: bool,
        patch_offset: int,
        initial_offset: int,
        n_t: int,
        mesh: "Mesh",
    ):
        self.file_path = file_path
        self.dimension = dimension
        self.extent = extent
        self.orientation = orientation
        self.cell_centered = cell_centered
        self._patch_offset = patch_offset
        self._initial_offset = initial_offset
        self._time_offset = -1
        self._n_t = n_t
        self.mesh = mesh
        self._boundary_parent: Boundary = None

    def n_t(self, count_duplicates=True) -> int:
        """Get the number of timesteps for which data was output.

        :param count_duplicates: If true, return the total number of data points, even if there is
            duplicate data for a timestep. Duplicate data might be output when restarting the
            simulation in between the simulation run.
        """
        if count_duplicates:
            return self._n_t
        return np.unique(self._boundary_parent.times).size

    @property
    def shape(self) -> Tuple:
        """Convenience function to calculate the shape of the array containing data for this patch."""
        return self.dimension.shape(self.cell_centered)

    @property
    def size(self) -> int:
        """Convenience function to calculate the number of data points in the array for this patch."""
        return self.dimension.size(self.cell_centered)

    def _post_init(self, time_offset: int):
        """Fully initialize the patch as soon as the number of timesteps is known."""
        self._time_offset = time_offset

    def get_coordinates(self, ignore_cell_centered: bool = False) -> Dict[Literal["x", "y", "z"], np.ndarray]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
        For cell-centered boundary data, the coordinates can be adjusted to represent cell-centered coordinates.

        :param ignore_cell_centered: Whether to shift the coordinates when the bndf is cell_centered or not.
        """
        # orientation = ('x', 'y', 'z')[self.orientation - 1] if self.orientation != 0 else ''
        # coords = {'x': set(), 'y': set(), 'z': set()}
        coords: Dict[Literal["x", "y", "z"], np.ndarray] = {}
        for dim in ("x", "y", "z"):
            co = self.mesh.coordinates[dim].copy()
            # In case the slice is cell-centered, we will shift the coordinates by half a cell
            # and remove the last coordinate. Compute the per-cell offset before trimming, since a
            # mesh with only a single cell along this axis leaves just one coordinate afterwards.
            if self.cell_centered and not ignore_cell_centered:
                co = co[:-1] + np.diff(co) / 2

            tol = _tolerance(self.mesh.coordinates[dim])
            coords[dim] = co[(co >= self.extent[dim][0] - tol) & (co <= self.extent[dim][1] + tol)]
            if coords[dim].size == 0:
                coords[dim] = np.array([co[np.argmin(np.abs(co - self.extent[dim][0]))]])

        return coords

    @property
    def data(self):
        """Method to load the quantity data for a single patch."""
        if not hasattr(self, "_data"):
            dtype_data = fdtype.new((("f", self.dimension.size(cell_centered=False)),))

            if self._n_t == -1:
                self._n_t = (os.stat(self.file_path).st_size - self._initial_offset) // self._time_offset
            n_t = self.n_t(count_duplicates=True)

            # This patch's record repeats every `_time_offset` bytes, with `_patch_offset` bytes
            # (the leading time value and any other patches' data) before it and the rest of the
            # timestep's block after it. Reading all timesteps in one go via a dtype that treats
            # those surrounding bytes as raw padding is far faster than seeking + reading a single
            # record at a time in a Python loop.
            #
            # Known inefficiency: when N patches share the same underlying file (the normal case
            # for a mesh/quantity with multiple boundary faces), each patch's own `.data` call
            # re-reads and discards the *other* N-1 patches' bytes as padding, so the file ends up
            # read from disk N times instead of once (e.g. triggered by Boundary.vmin()/vmax()'s
            # fallback over all patches, or cumulatively by Obstruction.get_global_boundary_data_
            # arrays() iterating every orientation). Fixing this properly means caching the bulk
            # read once per file (e.g. on `Boundary`, shared across its patches) instead of once
            # per `Patch` -- a real refactor, not attempted here; tracked as a follow-up.
            pre_pad = self._patch_offset
            post_pad = self._time_offset - self._patch_offset - dtype_data.itemsize
            # A size-0 void field is valid and has no effect, so pre/post padding never needs to
            # be conditionally omitted.
            row_dtype = np.dtype([("_pre", "V" + str(pre_pad)), ("data", dtype_data), ("_post", "V" + str(post_pad))])

            with open(self.file_path, "rb") as infile:
                infile.seek(self._initial_offset)
                rows = np.fromfile(infile, dtype=row_dtype, count=n_t)
            # fdtype.new(...) always auto-names its border/payload/border fields f0/f1/f2.
            all_data = rows["data"]["f1"]

            # Every record's data field was written by Fortran in column-major (F) order;
            # reshaping with order="F" applies that per-record reshape across the whole (n_t, n)
            # block at once, without a Python-level loop.
            shape = self.dimension.shape(cell_centered=False)
            full = all_data.reshape((n_t,) + shape, order="F")
            if self.cell_centered:
                full = full[:, :-1, :-1]

            unique_times_indices = np.unique(self._boundary_parent.times, return_index=True)[1]
            # The on-disk values are float32; match the float64 dtype the old np.empty(...)
            # allocation produced instead of silently narrowing precision.
            self._data = full[unique_times_indices].astype(np.float64)
        return self._data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        if hasattr(self, "_data"):
            del self._data

    def __repr__(self, *args, **kwargs):
        return f"Patch(shape={self.shape}, orientation={self.orientation}, extent={self.extent})"


class Boundary:
    """Container for boundary data specific to one quantity.

    :ivar quantity: Quantity object containing information about the quantity calculated for this
        :class:`Obstruction` with the corresponding short_name and unit.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar lower_bounds: Dictionary with lower bounds for each timestep with meshes as keys.
    :ivar upper_bounds: Dictionary with upper bounds for each timestep with meshes as keys.
    :ivar n_t: Total number of time steps for which output data has been written.
    """

    def __init__(
        self,
        quantity: Quantity,
        cell_centered: bool,
        times: Sequence[float],
        patches: List[Patch],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        self.quantity = quantity
        self.cell_centered = cell_centered
        self._patches = patches
        self.times = times
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def n_t(self, count_duplicates=True) -> int:
        """Get the number of timesteps for which data was output.

        :param count_duplicates: If true, return the total number of data points, even if there is
            duplicate data for a timestep. Duplicate data might be output when restarting the
            simulation in between the simulation run.
        """
        return self._patches[0].n_t(count_duplicates=count_duplicates)

    @property
    def patches(self) -> List[Patch]:
        """All patches, in the order FDS wrote them. A face of the simulation domain may consist of several patches
        with the same orientation, e.g. where a vent begins."""
        return list(self._patches)

    @property
    def orientations(self):
        """Return all orientations for which there is data available."""
        return sorted({p.orientation for p in self._patches})

    def get_nearest_timestep(self, time: float) -> int:
        """Calculates the nearest timestep for which data has been output for this obstruction."""
        idx = np.searchsorted(self.times, time, side="left")
        if time > 0 and (
            idx == len(self.times) or math.fabs(time - self.times[idx - 1]) < math.fabs(time - self.times[idx])
        ):
            return idx - 1
        else:
            return idx

    @property
    def data(self) -> Dict[int, Patch]:
        """The :class:`Patch` in each direction (-3=-z, -2=-y, -1=-x, 1=x, 2=y, 3=z).

        A face of the simulation domain may consist of several patches with the same orientation; this only holds
        the last of them and warns. Use :attr:`patches` or ``get_global_boundary_data_arrays`` for those faces.
        """
        data = {p.orientation: p for p in self._patches}
        if len(data) < len(self._patches):
            warnings.warn(
                f"{self.quantity.name}: several patches face the same direction, Boundary.data only holds the last "
                "of each. Use Boundary.patches, or get_global_boundary_data_arrays to combine them.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        return data

    def vmin(self, orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Minimum value of all patches at any time.

        :param orientation: Optionally filter by patches with a specific orientation; NaN if there is none.
        """
        if orientation == 0:
            curr_min = np.min(self.lower_bounds)
            if curr_min == np.float32(+1e33):
                return float(min(np.min(p.data) for p in self._patches))
            return float(curr_min)
        else:
            return float(min((np.min(p.data) for p in self._patches if p.orientation == orientation), default=np.nan))

    def vmax(self, orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Maximum value of all patches at any time.

        :param orientation: Optionally filter by patches with a specific orientation; NaN if there is none.
        """
        if orientation == 0:
            curr_max = np.max(self.upper_bounds)
            if curr_max == np.float32(-1e33):
                return float(max(np.max(p.data) for p in self._patches))
            return float(curr_max)
        else:
            return float(max((np.max(p.data) for p in self._patches if p.orientation == orientation), default=np.nan))

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        for p in self._patches:
            p.clear_cache()

    def __repr__(self):
        return f"Boundary(Quantity={self.quantity}, Patches={len(self._patches)})"


class SubObstruction:
    """An :class:`Obstruction` consists of 1 or more SubObstructions which can be hidden at specific points in time.

    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar bound_indices: Indices used to define obstruction bounds in terms of mesh locations.
    :ivar side_surfaces: Tuple of six :class:`Surface` s for each side of the cuboid.
    :ivar hide_times: List with points in time from when on the SubObstruction will be hidden.
    :ivar show_times: List with points in time from when on the SubObstruction will be shown.
    """

    def __init__(
        self, side_surfaces: Tuple, bound_indices: Tuple[int, int, int, int, int, int], extent: Extent, mesh: "Mesh"
    ):
        self.extent = extent
        self.side_surfaces = side_surfaces
        self.bound_indices = {
            "x": (bound_indices[0], bound_indices[1]),
            "y": (bound_indices[2], bound_indices[3]),
            "z": (bound_indices[4], bound_indices[5]),
        }
        self.mesh = mesh

        self._boundary_data: Dict[int, Boundary] = dict()

        self.hide_times = list()
        self.show_times = list()

    def _add_patches(
        self,
        bid: int,
        cell_centered: bool,
        quantity: str,
        short_name: str,
        unit: str,
        patches: List[Patch],
        times: Sequence[float],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = Boundary(
                Quantity(quantity, short_name, unit), cell_centered, times, patches, lower_bounds, upper_bounds
            )
            # Add reference to parent boundary class in patches
            for patch in patches:
                patch._boundary_parent = self._boundary_data[bid]

    @property
    def orientations(self):
        """Return all orientations for which there is data available."""
        if self.has_boundary_data:
            return next(iter(self._boundary_data.values())).orientations
        return []

    def get_coordinates(
        self, ignore_cell_centered: bool = False
    ) -> Dict[int, Dict[Literal["x", "y", "z"], np.ndarray]]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
        For cell-centered boundary data, the coordinates can be adjusted to represent cell-centered coordinates.

        :param ignore_cell_centered: Whether to shift the coordinates when the bndf is cell_centered or not.
        """
        if self.has_boundary_data:
            return {
                patch.orientation: patch.get_coordinates(ignore_cell_centered)
                for patch in next(iter(self._boundary_data.values())).patches
            }
        return {}

    def get_nearest_index(self, dimension: Literal["x", "y", "z"], orientation: int, value: float) -> int:
        """Get the nearest mesh coordinate index in a specific dimension for a specific orientation."""
        if self.has_boundary_data:
            coords = self.get_coordinates()[orientation][dimension]
            idx = np.searchsorted(coords, value, side="left")
            if idx > 0 and (idx == coords.size or math.fabs(value - coords[idx - 1]) < math.fabs(value - coords[idx])):
                return idx - 1
            else:
                return idx
        return np.nan

    @property
    def has_boundary_data(self):
        return len(self._boundary_data) != 0

    def get_data(self, quantity: Union[str, Quantity]):
        return _find_boundary(self._boundary_data, quantity)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._boundary_data[item]
        return self.get_data(item)

    def _hide(self, time: float):
        self.hide_times.append(time)
        self.hide_times.sort()

    def _show(self, time: float):
        self.show_times.append(time)
        self.show_times.sort()

    def n_t(self, count_duplicates=True) -> int:
        """Returns the number of timesteps for which boundary data is available.

        :param count_duplicates: If true, return the total number of data points, even if there is
            duplicate data for a timestep. Duplicate data might be output when restarting the
            simulation in between the simulation run.
        """
        if self.has_boundary_data:
            return next(iter(self._boundary_data.values())).n_t(count_duplicates=count_duplicates)
        return 0

    @property
    def times(self):
        """Return all timesteps for which boundary data is available, if any."""
        if self.has_boundary_data:
            return next(iter(self._boundary_data.values())).times
        return np.array([])

    def get_visible_times(self, times: Sequence[float]) -> np.ndarray:
        """Returns a ndarray filtering all time steps when the SubObstruction is visible/not hidden."""
        ret = list()
        hidden = False
        for time in times:
            if time in self.show_times:
                hidden = False
            if time in self.hide_times:
                hidden = True
            if not hidden:
                ret.append(time)
        return np.array(ret)

    def vmin(
        self,
        quantity: Union[str, Quantity],
        orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0,
    ) -> float:
        """Minimum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if self.has_boundary_data:
            return self.get_data(quantity).vmin(orientation)
        return np.nan

    def vmax(
        self,
        quantity: Union[str, Quantity],
        orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0,
    ) -> float:
        """Maximum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if self.has_boundary_data:
            return self.get_data(quantity).vmax(orientation)
        return np.nan

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        for bndf in self._boundary_data.values():
            bndf.clear_cache()

    def __repr__(self):
        return f"SubObstruction(Extent={self.extent})"


class Obstruction:
    """A box-shaped obstruction with specific surfaces (materials) on each side.

    :ivar id: ID of the obstruction.
    :ivar color_index: Type of coloring used to color obstruction.
        -1 - default color
        -2 - invisible
        -3 - use red, green, blue and alpha (rgba attribute)
        n>0 - use n’th color table entry
    :ivar block_type: Defines how the obstruction is drawn.
        -1 - use surface to obtain blocktype
        0 - regular block
        2 - outline
    :ivar texture_origin: Origin position of the texture provided by the surface. When the texture
        does have a pattern, for example windows or bricks, the texture_origin specifies where the
        pattern should begin.
    :ivar rgba: Optional color of the obstruction in form of a 4-element tuple
        (ranging from 0.0 to 1.0).
    """

    def __init__(
        self,
        oid: str,
        color_index: int,
        block_type: int,
        texture_origin: Tuple[float, float, float],
        rgba: Union[Tuple[()], Tuple[float, float, float, float]] = (),
    ):
        self.id = oid
        self.color_index = color_index
        self.block_type = block_type
        self.texture_origin = texture_origin
        if len(rgba) != 0:
            self.rgba = rgba

        # A mesh may contain more than one SubObstruction for this Obstruction (e.g. when a MULT
        # block repeats the same obstruction multiple times within a single mesh), so each mesh id
        # maps to a list of SubObstructions rather than a single one.
        self._subobstructions: Dict[str, List[SubObstruction]] = dict()

    @property
    def _all_subobstructions(self) -> List[SubObstruction]:
        """Flat list of all SubObstructions across all meshes."""
        return [subobst for subobsts in self._subobstructions.values() for subobst in subobsts]

    @property
    def bounding_box(self) -> Extent:
        """:class:`Extent` object representing the bounding box around the Obstruction."""
        extents = [sub.extent for sub in self._all_subobstructions]

        return Extent(
            min(extents, key=lambda e: e.x_start).x_start,
            max(extents, key=lambda e: e.x_end).x_end,
            min(extents, key=lambda e: e.y_start).y_start,
            max(extents, key=lambda e: e.y_end).y_end,
            min(extents, key=lambda e: e.z_start).z_start,
            max(extents, key=lambda e: e.z_end).z_end,
        )

    @property
    def orientations(self):
        """Return all orientations for which there is data available."""
        if self.has_boundary_data:
            orientations = set()
            for subobst in self._all_subobstructions:
                orientations.update(subobst.orientations)
            return sorted(list(orientations))
        return []

    def n_t(self, count_duplicates=True) -> int:
        """Returns the number of timesteps for which boundary data is available.

        :param count_duplicates: If true, return the total number of data points, even if there is
            duplicate data for a timestep. Duplicate data might be output when restarting the
            simulation in between the simulation run.
        """
        for subobst in self._all_subobstructions:
            if subobst.has_boundary_data:
                return subobst.n_t(count_duplicates=count_duplicates)
        return 0

    @property
    def times(self):
        """Return all timesteps for which boundary data is available, if any."""
        for subobst in self._all_subobstructions:
            if subobst.has_boundary_data:
                return subobst.times
        return np.array([])

    def get_visible_times(self, times: Sequence[float]):
        """Returns an ndarray filtering all time steps when theSubObstruction is visible/not hidden."""
        for subobst in self._all_subobstructions:
            return subobst.get_visible_times(times)
        return np.array([])

    def get_coordinates(
        self, ignore_cell_centered: bool = False
    ) -> Dict[int, Dict[Literal["x", "y", "z"], np.ndarray]]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
        For cell-centered boundary data, the coordinates can be adjusted to represent cell-centered coordinates.

        :param ignore_cell_centered: Whether to shift the coordinates when the bndf is cell_centered or not.
        """
        if self.has_boundary_data:
            all_coords = dict()
            for orientation_int in self.orientations:
                orientation = ("x", "y", "z")[abs(orientation_int) - 1]
                coords = {"x": set(), "y": set(), "z": set()}
                for dim in ("x", "y", "z"):
                    if orientation == dim:
                        bounding_box_index = 0 if orientation_int < 0 else 1
                        coords[dim] = np.array([self.bounding_box[dim][bounding_box_index]])
                        continue
                    min_delta = math.inf
                    for subobst in self._all_subobstructions:
                        subobst_coords = subobst.get_coordinates(ignore_cell_centered)
                        if orientation_int not in subobst_coords:
                            continue
                        co = subobst_coords[orientation_int][dim]
                        if len(co) > 1:  # a face one node wide has no spacing along this axis
                            min_delta = min(min_delta, co[1] - co[0])
                        coords[dim].update(co)

                    coords[dim] = np.array(sorted(list(coords[dim])))
                    to_keep = np.full(coords[dim].shape, True)
                    if not math.isfinite(min_delta):  # all faces one node wide: use the grid spacing
                        min_delta = min(
                            np.min(np.diff(subobst.mesh.coordinates[dim])) for subobst in self._all_subobstructions
                        )
                    for i in range(len(coords[dim]) - 1):
                        if abs(coords[dim][i] - coords[dim][i + 1]) < min_delta / 2:
                            to_keep[i + 1] = False
                    coords[dim] = coords[dim][to_keep]

                    if len(coords[dim]) == 0:
                        bounding_box_index = 0 if orientation_int < 0 else 1
                        single_coordinate = self.bounding_box[dim][bounding_box_index]
                        nearest_coordinate = np.inf
                        for subobst in self._all_subobstructions:
                            mesh = subobst.mesh
                            mesh_coords = mesh.coordinates[dim]
                            idx = np.searchsorted(mesh_coords, single_coordinate, side="left")
                            if idx > 0 and (
                                idx == mesh_coords.size
                                or math.fabs(single_coordinate - mesh_coords[idx - 1])
                                < math.fabs(single_coordinate - mesh_coords[idx])
                            ):
                                idx = idx + 1
                            if mesh_coords[idx] - single_coordinate < nearest_coordinate - single_coordinate:
                                nearest_coordinate = mesh_coords[idx]
                        coords[dim] = np.array([nearest_coordinate])
                all_coords[orientation_int] = coords

            return all_coords
        return dict()

    def get_nearest_index(self, dimension: Literal["x", "y", "z"], orientation: int, value: float) -> int:
        """Get the nearest mesh coordinate index in a specific dimension for a specific orientation."""
        if self.has_boundary_data:
            coords = self.get_coordinates()[orientation][dimension]
            idx = np.searchsorted(coords, value, side="left")
            if idx > 0 and (idx == coords.size or math.fabs(value - coords[idx - 1]) < math.fabs(value - coords[idx])):
                return idx - 1
            else:
                return idx
        return np.nan

    @property
    def quantities(self) -> List[Quantity]:
        """Get a list of all quantities for which boundary data exists."""
        if self.has_boundary_data:
            qs = set()
            for subobst in self._all_subobstructions:
                for b in subobst._boundary_data.values():
                    qs.add(b.quantity)
            return list(qs)
        return []

    @property
    def meshes(self) -> List["Mesh"]:
        """Returns a list of all meshes this slice cuts through."""
        return [subobsts[0].mesh for subobsts in self._subobstructions.values()]

    def filter_by_orientation(self, orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> List[SubObstruction]:
        """Filter all SubObstructions by a specific orientation. All returned SubObstructions will contain boundary data
        in the specified orientation.
        """
        if self.has_boundary_data:
            return [subobst for subobst in self._all_subobstructions if orientation in subobst.orientations]
        return []

    def get_boundary_data(
        self, quantity: Union[Quantity, str], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0
    ) -> Dict[str, Boundary]:
        """Gets the boundary data for a specific quantity of all SubObstructions.

        :param quantity: The quantity to filter by.
        :param orientation: Optionally filter by a specific orientation as well (-3=-z, -2=-y, -1=-x, 1=x, 2=y, 3=z).
            A value of 0 indicates to no filter.
        """
        if isinstance(quantity, Quantity):
            quantity = quantity.name

        ret = {
            subobst.mesh.id: subobst.get_data(quantity)
            for subobst in self._all_subobstructions
            if subobst.has_boundary_data
        }
        if orientation == 0:
            return ret
        return {mesh: bndf for mesh, bndf in ret.items() if orientation in bndf.orientations}

    def get_nearest_timestep(self, time: float, visible_only: bool = False) -> int:
        """Calculates the nearest timestep for which data has been output for this obstruction."""
        if self.has_boundary_data:
            times = self.get_visible_times(self.times) if visible_only else self.times
            idx = np.searchsorted(times, time, side="left")
            if time > 0 and (idx == len(times) or math.fabs(time - times[idx - 1]) < math.fabs(time - times[idx])):
                return idx - 1
            else:
                return idx
        return np.nan

    def get_nearest_patch(self, x: float = None, y: float = None, z: float = None):
        """Gets the patch of the :class:`SubObstruction` that has the least distance to the given point.
        If there are multiple patches with the same distance, a random one will be selected.
        """
        if self.has_boundary_data:
            d_min = np.finfo(float).max
            patches_with_dist = list()

            for subobst in self._all_subobstructions:
                for patch in subobst.get_data(self.quantities[0])._patches:
                    dx = max(patch.extent.x_start - x, 0, x - patch.extent.x_end) if x is not None else 0
                    dy = max(patch.extent.y_start - y, 0, y - patch.extent.y_end) if y is not None else 0
                    dz = max(patch.extent.z_start - z, 0, z - patch.extent.z_end) if z is not None else 0
                    d = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if d < d_min:
                        d_min = d
                        patches_with_dist = [(patch, dx, dy, dz)]
                    elif math.isclose(d, d_min, rel_tol=1e-9, abs_tol=0.0):
                        patches_with_dist.append((patch, dx, dy, dz))

            patches_min = [p[0] for p in patches_with_dist]

            if x is not None and len(patches_with_dist) > 1:
                patches_with_dist.sort(key=lambda p: p[1])
                patches_min = [p[0] for p in patches_with_dist]
            elif y is not None and len(patches_with_dist) > 1:
                patches_with_dist.sort(key=lambda p: p[2])
                patches_min = [p[0] for p in patches_with_dist]
            elif z is not None and len(patches_with_dist) > 1:
                patches_with_dist.sort(key=lambda p: p[3])
                patches_min = [p[0] for p in patches_with_dist]

            if len(patches_min) > 0:
                return patches_min[0]
        return None

    @property
    def has_boundary_data(self):
        """Whether boundary data has been output in the simulation."""
        return any(subobst.has_boundary_data for subobst in self._all_subobstructions)

    def get_global_boundary_data_arrays(
        self, quantity: Union[str, Quantity]
    ) -> Dict[int, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        """Creates a global numpy ndarray from all subobstruction's boundary data for each orientation.

        Each array has the shape (n_t, a, b), where a and b are the two axes of the face in the order x, y, z (e.g. y
        and z for the orientations -1 and 1). The grid uses the finest cell size of the meshes the obstruction spans;
        values of coarser meshes are repeated, and points not covered by any patch are NaN. The grid is uniform, so
        on stretched meshes values are placed at the nearest grid point below their node. Frames a mesh did not write
        are NaN as well.

        If the face lies at different positions in different meshes, e.g. because the obstruction spans meshes with
        different cell sizes and FDS snaps its bounds to each grid, the orientation maps to a tuple with one array per
        position, ordered by position.

        :param quantity: The quantity's name or short name for which to load the global arrays.
        """
        if not self.has_boundary_data:
            return dict()
        return _global_boundary_arrays([subobst.get_data for subobst in self._all_subobstructions], quantity)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        for s in self._all_subobstructions:
            s.clear_cache()

    def vmin(
        self,
        quantity: Union[str, Quantity],
        orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0,
    ) -> float:
        """Minimum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        values = [s.vmin(quantity, orientation) for s in self._all_subobstructions if s.has_boundary_data]
        return min((v for v in values if not np.isnan(v)), default=np.nan)

    def vmax(
        self,
        quantity: Union[str, Quantity],
        orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0,
    ) -> float:
        """Maximum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        values = [s.vmax(quantity, orientation) for s in self._all_subobstructions if s.has_boundary_data]
        return max((v for v in values if not np.isnan(v)), default=np.nan)

    def get_subobstructions(self, mesh: "Mesh") -> List[SubObstruction]:
        """All parts of this obstruction in the given mesh; several OBST lines can share an ID."""
        return list(self._subobstructions.get(mesh.id, []))

    def __getitem__(self, key: Union[int, str, "Mesh"]):
        """Gets either the nth :class:`SubObstruction` or the (first) one with the given mesh-id."""

        if isinstance(key, int):
            return self._all_subobstructions[key]
        elif isinstance(key, str):
            return self._subobstructions[key][0]
        return self._subobstructions[key.id][0]

    def __eq__(self, other):
        if not isinstance(other, Obstruction):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self, *args, **kwargs):
        return (
            f"Obstruction(id={self.id}, Bounding-Box={self.bounding_box}, "
            f"SubObstructions={len(self._all_subobstructions)}"
            + (f", Quantities={[q.short_name for q in self.quantities]}" if self.has_boundary_data else "")
            + ")"
        )


def _find_boundary(boundary_data: Dict[int, Boundary], quantity: Union[str, Quantity]) -> Boundary:
    """The boundary data of a quantity, by its name or short name, among those of a mesh or subobstruction."""
    if isinstance(quantity, Quantity):
        quantity = quantity.name
    # Quantity == str matches the name or the short name, ignoring case
    b = next((b for b in boundary_data.values() if b.quantity == quantity), None)
    if b is None:
        raise KeyError(f"No boundary data found for quantity '{quantity}'")
    return b


def _global_boundary_arrays(
    lookups: List[Callable[[Union[str, Quantity]], Boundary]], quantity: Union[str, Quantity]
) -> Dict[int, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """Combines the patches of a quantity from several meshes or subobstructions into one array of shape (n_t, a, b)
    per orientation, or a tuple of them, one per position, if the faces of an orientation lie at several positions.
    See :meth:`Obstruction.get_global_boundary_data_arrays`.

    :param lookups: The boundary data lookup of each mesh or subobstruction, which raises KeyError if it does not have
        the quantity. Raises KeyError if none of them has it.
    """
    patches_by_orientation: Dict[int, List[Patch]] = dict()
    for lookup in lookups:
        try:
            boundary = lookup(quantity)
        except KeyError:
            continue
        for patch in boundary.patches:
            patches_by_orientation.setdefault(patch.orientation, list()).append(patch)
    if not patches_by_orientation:
        raise KeyError(f"No boundary data found for quantity '{quantity}'")

    result = dict()
    for orientation in sorted(patches_by_orientation):
        normal = ("x", "y", "z")[abs(orientation) - 1]
        levels: List[Tuple[float, List[Patch]]] = list()
        for patch in patches_by_orientation[orientation]:
            position = patch.extent[normal][0]
            tol = _tolerance(patch.mesh.coordinates[normal])
            level = next((lvl for lvl in levels if abs(lvl[0] - position) <= tol), None)
            if level is None:
                levels.append((position, [patch]))
            else:
                level[1].append(patch)
        levels.sort(key=lambda lvl: lvl[0])
        grids = tuple(_global_patch_grid(level_patches, normal) for _, level_patches in levels)
        result[orientation] = grids[0] if len(grids) == 1 else grids
    return result


def _tolerance(coordinates: np.ndarray) -> float:
    """Tolerance for comparing positions on a grid: a thousandth of its smallest cell."""
    return 1e-3 * float(np.min(np.diff(coordinates))) if coordinates.size > 1 else 1e-6


def _global_patch_grid(patches: List[Patch], normal: Literal["x", "y", "z"]) -> np.ndarray:
    """Combines patches lying in one plane into one array of shape (n_t, a, b) on the finest grid of the patches."""
    cell_centered = patches[0].cell_centered
    axes = [dim for dim in ("x", "y", "z") if dim != normal]
    nodes = {dim: [patch.get_coordinates(ignore_cell_centered=True)[dim] for patch in patches] for dim in axes}

    positions = dict()
    for dim in axes:
        lower = min(n[0] for n in nodes[dim])
        upper = max(n[-1] for n in nodes[dim])
        steps = [np.min(np.diff(n)) for n in nodes[dim] if n.size > 1]
        step = min(steps) if steps else 0.0
        n_nodes = int(round((upper - lower) / step)) + 1 if step > 0 else 1
        positions[dim] = lower + step * np.arange(n_nodes)
        if cell_centered:
            positions[dim] = positions[dim][:-1] + step / 2

    n_t = max(patch.data.shape[0] for patch in patches)
    grid = np.full((n_t,) + tuple(positions[dim].size for dim in axes), np.nan)
    # On a node shared by two patches, the patch further up in x, y and z wins, as each is written after those below
    order = sorted(range(len(patches)), key=lambda i: tuple(nodes[dim][i][0] for dim in axes))
    for i in order:
        patch = patches[i]
        global_indices = list()
        patch_indices = list()
        for dim in axes:
            patch_nodes = nodes[dim][i]
            tol = _tolerance(patch_nodes)
            pos = positions[dim]
            if cell_centered:
                inside = (pos > patch_nodes[0]) & (pos < patch_nodes[-1])
                n_values = patch_nodes.size - 1
            else:
                inside = (pos >= patch_nodes[0] - tol) & (pos <= patch_nodes[-1] + tol)
                n_values = patch_nodes.size
            # Each point of the global grid takes the value of the patch's cell it lies in, or of the patch's last
            # node at or below it, so values of coarser meshes are repeated
            idx = np.searchsorted(patch_nodes, pos[inside] + tol, side="right") - 1
            global_indices.append(np.flatnonzero(inside))
            patch_indices.append(np.clip(idx, 0, n_values - 1))
        data = patch.data
        grid[(slice(data.shape[0]),) + np.ix_(*global_indices)] = data[(slice(None),) + np.ix_(*patch_indices)]
    return grid
