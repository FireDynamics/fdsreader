import os
from typing import List, Dict, Tuple, Union, Sequence
from typing_extensions import Literal
import numpy as np

from fdsreader.utils import Surface, Extent, Quantity, Dimension
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings


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
                 patch_offset: int, initial_offset: int, n_t: int):
        self.file_path = file_path
        self.dimension = dimension
        self.extent = extent
        self.orientation = orientation
        self.cell_centered = cell_centered
        self._patch_offset = patch_offset
        self._initial_offset = initial_offset
        self._time_offset = -1
        self.n_t = n_t

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
        :class:`Obstruction` with the corresponding label and unit.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar lower_bounds: Dictionary with lower bounds for each timestep with meshes as keys.
    :ivar upper_bounds: Dictionary with upper bounds for each timestep with meshes as keys.
    :ivar n_t: Total number of time steps for which output data has been written.
    """

    def __init__(self, quantity: Quantity, cell_centered: bool, times: np.ndarray, n_t: int, patches: List[Patch],
                 lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self.quantity = quantity
        self.cell_centered = cell_centered
        self.patches = patches
        self.times = times
        self.n_t = n_t
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    @property
    def data(self) -> Dict[int, Patch]:
        """The :class:`Patch` in each direction (-3=-z, -2=-y, -1=-x, 1=x, 2=y, 3=y).
        """
        return {p.orientation: p for p in self.patches}

    def vmin(self, orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Minimum value of all patches at any time.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        if orientation == 0:
            curr_min = np.min(self.lower_bounds)
            if curr_min == 0.0:
                return min(np.min(p.data) for p in self.patches)
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
                return max(np.max(p.data) for p in self.patches)
            return curr_max
        else:
            return np.max(self.data[orientation].data)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for p in self.patches:
            p.clear_cache()

    def __repr__(self):
        return f"Boundary(Quantity={self.quantity}, Patches={len(self.patches)})"


class SubObstruction:
    """An :class:`Obstruction` consists of 1 or more SubObstructions which can be hidden at specific points in time.

    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar bound_indices: Indices used to define obstruction bounds in terms of mesh locations.
    :ivar side_surfaces: Tuple of six surfaces for each side of the cuboid.
    :ivar hide_times: List with points in time from when on the SubObstruction will be hidden.
    :ivar show_times: List with points in time from when on the SubObstruction will be shown.
    """

    def __init__(self, side_surfaces: Tuple[Surface, ...], bound_indices: Tuple[int, int, int, int, int, int],
                 extent: Extent):
        self.extent = extent
        self.side_surfaces = side_surfaces
        self.bound_indices = {'x': (bound_indices[0], bound_indices[1]),
                              'y': (bound_indices[2], bound_indices[3]),
                              'z': (bound_indices[4], bound_indices[5])}

        self._boundary_data: Dict[int, Boundary] = dict()

        self.hide_times = list()
        self.show_times = list()

    def _add_patches(self, bid: int, cell_centered: bool, quantity: str, label: str, unit: str, patches: List[Patch],
                     times: np.ndarray, n_t: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = Boundary(Quantity(quantity, label, unit), cell_centered, times, n_t, patches,
                                                lower_bounds, upper_bounds)

        if not settings.LAZY_LOAD:
            _ = self._boundary_data[bid].data

    def get_data(self, quantity: Union[str, Quantity]):
        if type(quantity) != str:
            quantity = quantity.quantity
        return next(b for b in self._boundary_data.values() if
                    b.quantity.quantity.lower() == quantity.lower() or b.quantity.label.lower() == quantity.lower())

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

    def visible_times(self, times: Sequence[float]) -> np.ndarray:
        """Returns an ndarray containing all time steps when there is data available for the SubObstruction. Will return an
            empty list when no data is output at all.

        :param times: All timesteps of the simulation.
        """
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

    def vmin(self, quantity: Union[str, Quantity], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Minimum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        return self.get_data(quantity).vmin(orientation)

    def vmax(self, quantity: Union[str, Quantity], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Maximum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        return self.get_data(quantity).vmax(orientation)

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
     \n-1 - default color
     \n-2 - invisible
     \n-3 - use red, green, blue and alpha (rgba attribute)
     \nn>0 - use n’th color table entry
    :ivar block_type: Defines how the obstruction is drawn.
     \n-1 - use surface to obtain blocktype
     \n0 - regular block
     \n2 - outline
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

        self._subobstructions: List[SubObstruction] = list()

    @property
    def bounding_box(self) -> Extent:
        """:class:`Extent` object representing the bounding box around the Obstruction.
        """
        extents = [sub.extent for sub in self._subobstructions]

        return Extent(min(extents, key=lambda e: e.x_start).x_start, max(extents, key=lambda e: e.x_end).x_end,
                      min(extents, key=lambda e: e.y_start).y_start, max(extents, key=lambda e: e.y_end).y_end,
                      min(extents, key=lambda e: e.z_start).z_start, max(extents, key=lambda e: e.z_end).z_end)

    @property
    def quantities(self) -> List[Quantity]:
        """Get a list of all quantities for which boundary data exists.
        """
        return [b.quantity for b in self._subobstructions[0]._boundary_data.values()]

    def filter_by_orientation(self, orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> List[SubObstruction]:
        """Filter all SubObstructions by a specific orientation. All returned SubObstructions will contain boundary data
            in the specified orientation.
        """
        return [subobst for subobst in self._subobstructions if
                orientation in next(iter(subobst._boundary_data.values())).data.keys()]

    def get_boundary_data(self, quantity: Union[Quantity, str],
                          orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> List[Boundary]:
        """Gets the boundary data for a specific quantity of all SubObstructions.

        :param quantity: The quantity to filter by.
        :param orientation: Optionally filter by a specific orientation as well (-3=-z, -2=-y, -1=-x, 1=x, 2=y, 3=z).
            A value of 0 indicates to no filter.
        """
        if type(quantity) != str:
            quantity = quantity.quantity

        ret = [subobst.get_data(quantity) for subobst in self._subobstructions]
        if orientation == 0:
            return ret
        return [bndf for bndf in ret if orientation in bndf.data.keys()]

    @property
    def has_boundary_data(self):
        """Whether boundary data has been output in the simulation.
        """
        return len(self._subobstructions[0]._boundary_data) != 0

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for s in self._subobstructions:
            s.clear_cache()

    def vmin(self, quantity: Union[str, Quantity], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Minimum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        return min(s.vmin(quantity, orientation) for s in self._subobstructions)

    def vmax(self, quantity: Union[str, Quantity], orientation: Literal[-3, -2, -1, 0, 1, 2, 3] = 0) -> float:
        """Maximum value of all patches at any time for a specific quantity.

        :param orientation: Optionally filter by patches with a specific orientation.
        """
        return max(s.vmax(quantity, orientation) for s in self._subobstructions)

    def __getitem__(self, index):
        """Gets the nth :class:`SubObstruction`.
        """
        return self._subobstructions[index]

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self, *args, **kwargs):
        return f"Obstruction(id={self.id}, Bounding-Box={self.bounding_box}, SubObstructions={len(self._subobstructions)}" + (
            f", Quantities={[q.label for q in self.quantities]}" if self.has_boundary_data else "") + ")"
