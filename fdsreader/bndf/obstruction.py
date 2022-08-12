import math
import os
from typing import List, Dict, Tuple, Union, Sequence
from typing_extensions import Literal
import numpy as np

from fdsreader.utils import Extent, Quantity, Dimension
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings

# Unfortunately, this is necessary due to a cyclic reference. "Mesh" is only needed for static type hints anyway
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdsreader.fds_classes import Mesh


class Patch:
    """Container for the actual data which is stored as rectangular plane with specific orientation
        and extent.

    :ivar dimension: :class:`Dimension` object containing information about steps in each dimension.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar orientation: The direction the patch is facing (x={-1;1}, y={-2;2}, z={-3;3}).
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar n_t: Total number of time steps for which output data has been written.
    """

    def __init__(self, file_path: str, dimension: Dimension, extent: Extent, orientation: int, cell_centered: bool,
                 patch_offset: int, initial_offset: int, n_t: int, mesh: 'Mesh'):
        self.file_path = file_path
        self.dimension = dimension
        self.extent = extent
        self.orientation = orientation
        self.cell_centered = cell_centered
        self._patch_offset = patch_offset
        self._initial_offset = initial_offset
        self._time_offset = -1
        self.n_t = n_t
        self.mesh = mesh

    @property
    def shape(self) -> Tuple:
        """Convenience function to calculate the shape of the array containing data for this patch.
        """
        return self.dimension.shape(self.cell_centered)

    @property
    def size(self) -> int:
        """Convenience function to calculate the number of data points in the array for this patch.
        """
        return self.dimension.size(self.cell_centered)

    def _post_init(self, time_offset: int):
        """Fully initialize the patch as soon as the number of timesteps is known.
        """
        self._time_offset = time_offset

    def get_coordinates(self, ignore_cell_centered: bool = False) -> Dict[Literal['x', 'y', 'z'], np.ndarray]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
            For cell-centered boundary data, the coordinates can be adjusted to represent cell-centered coordinates.

            :param ignore_cell_centered: Whether to shift the coordinates when the bndf is cell_centered or not.
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
    def data(self):
        """Method to load the quantity data for a single patch.
        """
        if not hasattr(self, "_data"):
            dtype_data = fdtype.new((('f', self.dimension.size(cell_centered=False)),))

            if self.n_t == -1:
                self.n_t = (os.stat(self.file_path).st_size - self._initial_offset) // self._time_offset

            self._data = np.empty((self.n_t,) + self.shape)
            with open(self.file_path, 'rb') as infile:
                for t in range(self.n_t):
                    infile.seek(self._initial_offset + self._patch_offset + t * self._time_offset)
                    data = np.fromfile(infile, dtype_data, 1)[0][1].reshape(
                        self.dimension.shape(cell_centered=False), order='F')
                    if self.cell_centered:
                        self._data[t, :] = data[:-1, :-1]
                    else:
                        self._data[t, :] = data
        return self._data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
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

    def __init__(self, quantity: Quantity, cell_centered: bool, times: Sequence[float], n_t: int, patches: List[Patch],
                 lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self.quantity = quantity
        self.cell_centered = cell_centered
        self._patches = patches
        self.times = times
        self.n_t = n_t
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    @property
    def orientations(self):
        """Return all orientations for which there is data available.
        """
        return [p.orientation for p in self._patches]

    def get_nearest_timestep(self, time: float) -> int:
        """Calculates the nearest timestep for which data has been output for this obstruction.
        """
        idx = np.searchsorted(self.times, time, side="left")
        if time > 0 and (idx == len(self.times) or np.math.fabs(
                time - self.times[idx - 1]) < np.math.fabs(time - self.times[idx])):
            return idx - 1
        else:
            return idx

    @property
    def data(self) -> Dict[int, Patch]:
        """The :class:`Patch` in each direction (-3=-z, -2=-y, -1=-x, 1=x, 2=y, 3=y).
        """
        return {p.orientation: p for p in self._patches}

    def vmin(self, orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Minimum value of all patches at any time.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if orientation == 0:
            curr_min = np.min(self.lower_bounds)
            if curr_min == 0.0:
                return min(np.min(p.data) for p in self._patches)
            return curr_min
        else:
            return np.min(self.data[orientation].data)

    def vmax(self, orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Maximum value of all patches at any time.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if orientation == 0:
            curr_max = np.max(self.upper_bounds)
            if curr_max == np.float32(-1e33):
                return max(np.max(p.data) for p in self._patches)
            return curr_max
        else:
            return np.max(self.data[orientation].data)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
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

    def __init__(self, side_surfaces: Tuple, bound_indices: Tuple[int, int, int, int, int, int],
                 extent: Extent, mesh: 'Mesh'):
        self.extent = extent
        self.side_surfaces = side_surfaces
        self.bound_indices = {'x': (bound_indices[0], bound_indices[1]),
                              'y': (bound_indices[2], bound_indices[3]),
                              'z': (bound_indices[4], bound_indices[5])}
        self.mesh = mesh

        self._boundary_data: Dict[int, Boundary] = dict()

        self.hide_times = list()
        self.show_times = list()

    def _add_patches(self, bid: int, cell_centered: bool, quantity: str, short_name: str, unit: str,
                     patches: List[Patch], times: Sequence[float], n_t: int, lower_bounds: np.ndarray,
                     upper_bounds: np.ndarray):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = Boundary(Quantity(quantity, short_name, unit), cell_centered, times, n_t,
                                                patches, lower_bounds, upper_bounds)

        if not settings.LAZY_LOAD:
            _ = self._boundary_data[bid].data

    @property
    def orientations(self):
        """Return all orientations for which there is data available.
        """
        if self.has_boundary_data:
            return next(iter(self._boundary_data.values())).orientations
        return []

    def get_coordinates(self, ignore_cell_centered: bool = False) -> Dict[
        int, Dict[Literal['x', 'y', 'z'], np.ndarray]]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
            For cell-centered boundary data, the coordinates can be adjusted to represent cell-centered coordinates.

            :param ignore_cell_centered: Whether to shift the coordinates when the bndf is cell_centered or not.
        """
        if self.has_boundary_data:
            return {orientation: patch.get_coordinates(ignore_cell_centered) for orientation, patch in
                    next(iter(self._boundary_data.values())).data.items()}
        return {}

    def get_nearest_index(self, dimension: Literal['x', 'y', 'z'], orientation: int, value: float) -> int:
        """Get the nearest mesh coordinate index in a specific dimension for a specific orientation.
        """
        if self.has_boundary_data:
            coords = self.get_coordinates()[orientation][dimension]
            idx = np.searchsorted(coords, value, side="left")
            if idx > 0 and (idx == coords.size or np.math.fabs(value - coords[idx - 1]) < np.math.fabs(
                    value - coords[idx])):
                return idx - 1
            else:
                return idx
        return np.nan

    @property
    def has_boundary_data(self):
        return len(self._boundary_data) != 0

    def get_data(self, quantity: Union[str, Quantity]):
        if type(quantity) == Quantity:
            quantity = quantity.name
        return next(b for b in self._boundary_data.values() if
                    b.quantity.name.lower() == quantity.lower() or b.quantity.short_name.lower() == quantity.lower())

    def __getitem__(self, item):
        if type(item) == int:
            return self._boundary_data[item]
        return self.get_data(item)

    def _hide(self, time: float):
        self.hide_times.append(time)
        self.hide_times.sort()

    def _show(self, time: float):
        self.show_times.append(time)
        self.show_times.sort()

    @property
    def n_t(self):
        """Returns the number of timesteps for which boundary data is available.
        """
        if self.has_boundary_data:
            return next(iter(self._boundary_data.values())).n_t
        return np.nan

    @property
    def times(self):
        """Return all timesteps for which boundary data is available, if any.
        """
        if self.has_boundary_data:
            return next(iter(self._boundary_data.values())).times
        return np.array([])

    def get_visible_times(self, times: Sequence[float]) -> np.ndarray:
        """Returns an ndarray filtering all time steps when theSubObstruction is visible/not hidden.
        """
        ret = list()
        if self.has_boundary_data:
            hidden = False
            for time in times:
                if time in self.show_times:
                    hidden = False
                if time in self.hide_times:
                    hidden = True
                if not hidden:
                    ret.append(time)
        return np.array(ret)

    def vmin(self, quantity: Union[str, Quantity], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Minimum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if self.has_boundary_data:
            return self.get_data(quantity).vmin(orientation)
        return np.nan

    def vmax(self, quantity: Union[str, Quantity], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Maximum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if self.has_boundary_data:
            return self.get_data(quantity).vmax(orientation)
        return np.nan

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
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

    def __init__(self, oid: int, color_index: int, block_type: int, texture_origin: Tuple[float, float, float],
                 rgba: Union[Tuple[()], Tuple[float, float, float, float]] = ()):
        self.id = oid
        self.color_index = color_index
        self.block_type = block_type
        self.texture_origin = texture_origin
        if len(rgba) != 0:
            self.rgba = rgba

        self._subobstructions: Dict['Mesh', SubObstruction] = dict()

    @property
    def bounding_box(self) -> Extent:
        """:class:`Extent` object representing the bounding box around the Obstruction.
        """
        extents = [sub.extent for sub in self._subobstructions.values()]

        return Extent(min(extents, key=lambda e: e.x_start).x_start, max(extents, key=lambda e: e.x_end).x_end,
                      min(extents, key=lambda e: e.y_start).y_start, max(extents, key=lambda e: e.y_end).y_end,
                      min(extents, key=lambda e: e.z_start).z_start, max(extents, key=lambda e: e.z_end).z_end)

    @property
    def orientations(self):
        """Return all orientations for which there is data available.
        """
        if self.has_boundary_data:
            orientations = set()
            for subobst in self._subobstructions.values():
                orientations.update(subobst.orientations)
            return sorted(list(orientations))
        return []

    @property
    def n_t(self):
        """Returns the number of timesteps for which boundary data is available.
        """
        for subobst in self._subobstructions.values():
            if subobst.has_boundary_data:
                return subobst.n_t
        return np.nan

    @property
    def times(self):
        """Return all timesteps for which boundary data is available, if any.
        """
        for subobst in self._subobstructions.values():
            if subobst.has_boundary_data:
                return subobst.times
        return np.array([])

    def get_visible_times(self, times: Sequence[float]):
        """Returns an ndarray filtering all time steps when theSubObstruction is visible/not hidden.
        """
        for subobst in self._subobstructions.values():
            if subobst.has_boundary_data:
                return subobst.get_visible_times(times)
        return np.array([])

    def get_coordinates(self, ignore_cell_centered: bool = False) -> Dict[
        int, Dict[Literal['x', 'y', 'z'], np.ndarray]]:
        """Returns a dictionary containing a numpy ndarray with coordinates for each dimension.
            For cell-centered boundary data, the coordinates can be adjusted to represent cell-centered coordinates.

            :param ignore_cell_centered: Whether to shift the coordinates when the bndf is cell_centered or not.
        """
        if self.has_boundary_data:
            all_coords = dict()
            for orientation_int in self.orientations:
                orientation = ('x', 'y', 'z')[abs(orientation_int) - 1]
                coords = {'x': set(), 'y': set(), 'z': set()}
                for dim in ('x', 'y', 'z'):
                    if orientation == dim:
                        bounding_box_index = 0 if orientation_int < 0 else 1
                        coords[dim] = np.array([self.bounding_box[dim][bounding_box_index]])
                        continue
                    min_delta = math.inf
                    for subobst in self._subobstructions.values():
                        subobst_coords = subobst.get_coordinates(ignore_cell_centered)
                        if orientation_int not in subobst_coords:
                            continue
                        co = subobst_coords[orientation_int][dim]
                        min_delta = min(min_delta, co[1] - co[0])
                        coords[dim].update(co)

                    coords[dim] = np.array(sorted(list(coords[dim])))
                    to_keep = np.full(coords[dim].shape, True)
                    for i in range(len(coords[dim]) - 1):
                        if abs(coords[dim][i] - coords[dim][i + 1]) < min_delta / 2:
                            to_keep[i + 1] = False
                    coords[dim] = coords[dim][to_keep]

                    if len(coords[dim]) == 0:
                        bounding_box_index = 0 if orientation_int < 0 else 1
                        single_coordinate = self.bounding_box[dim][bounding_box_index]
                        nearest_coordinate = np.inf
                        for mesh in self._subobstructions.keys():
                            mesh_coords = mesh.coordinates[dim]
                            idx = np.searchsorted(mesh_coords, single_coordinate, side="left")
                            if idx > 0 and (idx == mesh_coords.size or np.math.fabs(
                                    single_coordinate - mesh_coords[idx - 1]) < np.math.fabs(
                                single_coordinate - mesh_coords[idx])):
                                idx = idx + 1
                            if mesh_coords[idx] - single_coordinate < nearest_coordinate - single_coordinate:
                                nearest_coordinate = mesh_coords[idx]
                        coords[dim] = np.array([nearest_coordinate])
                all_coords[orientation_int] = coords

            return all_coords
        return dict()

    def get_nearest_index(self, dimension: Literal['x', 'y', 'z'], orientation: int, value: float) -> int:
        """Get the nearest mesh coordinate index in a specific dimension for a specific orientation.
        """
        if self.has_boundary_data:
            coords = self.get_coordinates()[orientation][dimension]
            idx = np.searchsorted(coords, value, side="left")
            if idx > 0 and (idx == coords.size or np.math.fabs(value - coords[idx - 1]) < np.math.fabs(
                    value - coords[idx])):
                return idx - 1
            else:
                return idx
        return np.nan

    @property
    def quantities(self) -> List[Quantity]:
        """Get a list of all quantities for which boundary data exists.
        """
        if self.has_boundary_data:
            qs = set()
            for subobst in self._subobstructions.values():
                for b in subobst._boundary_data.values():
                    qs.add(b.quantity)
            return list(qs)
        return []

    @property
    def meshes(self) -> List['Mesh']:
        """Returns a list of all meshes this slice cuts through.
        """
        return list(self._subobstructions.keys())

    def filter_by_orientation(self, orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> List[SubObstruction]:
        """Filter all SubObstructions by a specific orientation. All returned SubObstructions will contain boundary data
            in the specified orientation.
        """
        if self.has_boundary_data:
            return [subobst for subobst in self._subobstructions.values() if
                    orientation in subobst.orientations]
        return []

    def get_boundary_data(self, quantity: Union[Quantity, str],
                          orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> Dict[str, Boundary]:
        """Gets the boundary data for a specific quantity of all SubObstructions.

        :param quantity: The quantity to filter by.
        :param orientation: Optionally filter by a specific orientation as well (-3=-z, -2=-y, -1=-x, 1=x, 2=y, 3=z).
            A value of 0 indicates to no filter.
        """
        if type(quantity) == Quantity:
            quantity = quantity.name

        ret = {subobst.mesh.id: subobst.get_data(quantity) for subobst in self._subobstructions.values() if
               subobst.has_boundary_data}
        if orientation == 0:
            return ret
        return {mesh: bndf for mesh, bndf in ret.items() if orientation in bndf.data.keys()}

    def get_nearest_timestep(self, time: float, visible_only: bool = False) -> int:
        """Calculates the nearest timestep for which data has been output for this obstruction.
        """
        if self.has_boundary_data:
            times = self.get_visible_times(self.times) if visible_only else self.times
            idx = np.searchsorted(times, time, side="left")
            if time > 0 and (idx == len(times) or np.math.fabs(
                    time - times[idx - 1]) < np.math.fabs(time - times[idx])):
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
            patches_min = list()

            for subobst in self._subobstructions.values():
                for patch in subobst.get_data(self.quantities[0])._patches:
                    dx = max(patch.extent.x_start - x, 0, x - patch.extent.x_end) if x is not None else 0
                    dy = max(patch.extent.y_start - y, 0, y - patch.extent.y_end) if y is not None else 0
                    dz = max(patch.extent.z_start - z, 0, z - patch.extent.z_end) if z is not None else 0
                    d = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if d <= d_min:
                        d_min = d
                        patches_min.append(patch)

            if x is not None:
                patches_min.sort(key=lambda patch: (patch.extent.x_end - patch.extent.x_start))
            if y is not None:
                patches_min.sort(key=lambda patch: (patch.extent.y_end - patch.extent.y_start))
            if z is not None:
                patches_min.sort(key=lambda patch: (patch.extent.z_end - patch.extent.z_start))

            if len(patches_min) > 0:
                return patches_min[0]
        return None

    @property
    def has_boundary_data(self):
        """Whether boundary data has been output in the simulation.
        """
        return any(subobst.has_boundary_data for subobst in self._subobstructions.values())

    def get_global_boundary_data_arrays(self, quantity: Union[str, Quantity]) -> Dict[
        int, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Creates a global numpy ndarray from all subobstruction's boundary data for each orientation.

            :param quantity: The quantity's name or short name for which to load the global arrays.
        """
        if not self.has_boundary_data:
            return dict()
        for subobst in self._subobstructions.values():
            if subobst.has_boundary_data:
                cell_centered = next(iter(subobst._boundary_data.values())).cell_centered

        result = dict()
        for orientation_int in self.orientations:
            subobst_sets = [list(), list()]
            dim = ['x', 'y', 'z'][abs(orientation_int) - 1]
            random_subobst = next(
                subobst for subobst in self._subobstructions.values() if orientation_int in subobst.get_coordinates())
            base_coord = random_subobst.get_coordinates(ignore_cell_centered=False)[orientation_int][dim][0]

            for subobst in self._subobstructions.values():
                if not subobst.has_boundary_data:
                    continue
                coords = subobst.get_coordinates(ignore_cell_centered=False)
                if orientation_int not in coords:
                    continue
                if coords[orientation_int][dim][0] == base_coord:
                    subobst_sets[0].append(subobst)
                else:
                    subobst_sets[1].append(subobst)
            if len(subobst_sets[1]) == 0:
                subobst_sets.pop(1)

            first_result_grid = None
            for subobsts in subobst_sets:
                coord_min = {'x': math.inf, 'y': math.inf, 'z': math.inf}
                coord_max = {'x': -math.inf, 'y': -math.inf, 'z': -math.inf}
                for dim in ('x', 'y', 'z'):
                    for subobst in subobsts:
                        patch_extent = subobst.get_data(quantity).data[orientation_int].extent
                        co = subobst.get_coordinates(ignore_cell_centered=True)[orientation_int][dim]
                        co = co[np.where(np.logical_and(co >= patch_extent[dim][0], co <= patch_extent[dim][1]))]
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
                    for subobst in subobsts:
                        subobst_coords = subobst.get_coordinates(ignore_cell_centered=True)[orientation_int]
                        if len(subobst_coords[dim]) <= 1:
                            step_size = 0
                        else:
                            step_size = subobst_coords[dim][1] - subobst_coords[dim][0]
                        step_sizes_min[dim] = min(step_size, step_sizes_min[dim])
                        step_sizes_max[dim] = max(step_size, step_sizes_max[dim])
                        global_max[dim] = max(subobst_coords[dim][-1], global_max[dim])

                for dim in ('x', 'y', 'z'):
                    if step_sizes_min[dim] == 0:
                        step_sizes_min[dim] = math.inf
                        steps[dim] = 1
                    else:
                        steps[dim] = max(int(round((coord_max[dim] - coord_min[dim]) / step_sizes_min[dim])), 1) + (
                            0 if cell_centered else 1)

                grid = np.full((self.n_t, steps['x'], steps['y'], steps['z']), np.nan)

                start_coordinates = {'x': coord_min['x'], 'y': coord_min['y'], 'z': coord_min['z']}
                start_idx = dict()
                end_idx = dict()
                for subobst in subobsts:
                    patch_data = np.expand_dims(subobst.get_data(quantity).data[orientation_int].data,
                                                axis=abs(orientation_int))
                    subobst_coords = subobst.get_coordinates(ignore_cell_centered=True)[orientation_int]
                    for axis in (0, 1, 2):
                        dim = ('x', 'y', 'z')[axis]
                        if axis == abs(orientation_int) - 1:
                            start_idx[dim] = 0
                            end_idx[dim] = 1
                            continue
                        n_repeat = max(
                            int(round((subobst_coords[dim][1] - subobst_coords[dim][0]) / step_sizes_min[dim])), 1)

                        start_idx[dim] = int(
                            round((subobst_coords[dim][0] - start_coordinates[dim]) / step_sizes_min[dim]))
                        end_idx[dim] = int(
                            round((subobst_coords[dim][-1] - start_coordinates[dim]) / step_sizes_min[dim]))

                        # We ignore border points unless they are actually on the border of the simulation space as all
                        # other border points actually appear twice, as the subobstructions overlap. This only
                        # applies for face_centered data, as cell_centered data will not overlap.
                        if not cell_centered:
                            if axis != abs(orientation_int) - 1:
                                reduced_shape = list(patch_data.shape)
                                reduced_shape[axis + 1] -= 1
                                reduced_data_slices = tuple(slice(s) for s in reduced_shape)
                                patch_data = patch_data[reduced_data_slices]

                            # Temporarily save border points to add them back to the array again later
                            if subobst_coords[dim][-1] == global_max[dim]:
                                end_idx[dim] += 1
                                temp_data_slices = [slice(s) for s in patch_data.shape]
                                temp_data_slices[axis + 1] = slice(patch_data.shape[axis + 1] - 1, None)
                                temp_data = patch_data[tuple(temp_data_slices)]

                        if n_repeat > 1:
                            patch_data = np.repeat(patch_data, n_repeat, axis=axis + 1)

                        # Add border points back again if needed
                        if not cell_centered and subobst_coords[dim][-1] == global_max[dim]:
                            patch_data = np.concatenate((patch_data, temp_data), axis=axis + 1)

                    grid[:, start_idx['x']: end_idx['x'], start_idx['y']: end_idx['y'],
                    start_idx['z']: end_idx['z']] = patch_data.reshape(
                        (self.n_t, end_idx['x'] - start_idx['x'], end_idx['y'] - start_idx['y'],
                         end_idx['z'] - start_idx['z']))

                # Remove empty dimensions, but make sure to note remove the time dimension if there is only a single timestep
                grid = np.squeeze(grid)
                if len(grid.shape) == 2:
                    grid = grid[np.newaxis, :, :]
                if first_result_grid is not None:
                    result[orientation_int] = (first_result_grid, grid)
                else:
                    result[orientation_int] = grid
        return result

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for s in self._subobstructions.values():
            s.clear_cache()

    def vmin(self, quantity: Union[str, Quantity], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Minimum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if self.has_boundary_data:
            return min(s.vmin(quantity, orientation) for s in self._subobstructions.values())
        else:
            return np.nan

    def vmax(self, quantity: Union[str, Quantity], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Maximum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if self.has_boundary_data:
            return max(s.vmax(quantity, orientation) for s in self._subobstructions.values())
        return np.nan

    def __getitem__(self, key: Union[int, str, 'Mesh']):
        """Gets either the nth :class:`SubObstruction` or the one with the given mesh-id.
        """

        if type(key) == int:
            return tuple(self._subobstructions.values())[key]
        elif type(key) == str:
            key = tuple(mesh for mesh in self.meshes if mesh.id == key)[0]
        return self._subobstructions[key]

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self, *args, **kwargs):
        return f"Obstruction(id={self.id}, Bounding-Box={self.bounding_box}, SubObstructions={len(self._subobstructions)}" + (
            f", Quantities={[q.short_name for q in self.quantities]}" if self.has_boundary_data else "") + ")"
