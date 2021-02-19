from typing import List, Dict, Tuple, Union
from typing_extensions import Literal
import numpy as np

from fdsreader.utils import Surface, Mesh, Extent, Quantity, Dimension
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

    def __init__(self, file_path: str, dimension: Dimension, extent: Extent, orientation: int,
                 cell_centered: bool, _initial_offset: int, n_t: int):
        self.file_path = file_path
        self.dimension = dimension
        self.extent = extent
        self.orientation = orientation
        self.cell_centered = cell_centered
        self._initial_offset = _initial_offset
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
            self._data = np.empty((self.n_t,) + self.shape)
            dtype_data = fdtype.new((('f', self.dimension.size(cell_centered=False)),))
            with open(self.file_path, 'rb') as infile:
                for t in range(self.n_t):
                    infile.seek(self._initial_offset + t * self._time_offset)
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
    """Container for the actual data which is stored as rectangular plane with specific orientation
        and extent.

    :ivar quantity: Quantity object containing information about the quantity calculated for this
        :class:`Obstruction` with the corresponding label and unit.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar lower_bounds: Dictionary with lower bounds for each timestep with meshes as keys.
    :ivar upper_bounds: Dictionary with upper bounds for each timestep with meshes as keys.
    :ivar n_t: Total number of time steps for which output data has been written.
    """

    def __init__(self, cell_centered: bool, quantity: Quantity, times: np.ndarray, n_t: int,
                 lower_bounds: np.ndarray, upper_bounds: np.ndarray, initial_mesh: Mesh):
        self.cell_centered = cell_centered
        self.quantity = quantity
        self.times = times
        self.lower_bounds = {initial_mesh: lower_bounds}
        self.upper_bounds = {initial_mesh: upper_bounds}
        self.n_t = n_t
        self.extent = None

        self._patches: Dict[Mesh, List[Patch]] = dict()

    def _add_patches(self, mesh: Mesh, patches: List[Patch], lower_bounds: np.ndarray,
                     upper_bounds: np.ndarray, ):
        self._patches[mesh] = patches
        self.lower_bounds[mesh] = lower_bounds
        self.upper_bounds[mesh] = upper_bounds

    @staticmethod
    def sort_patches_cartesian(patches_in: List[Patch]):
        """Returns all patches (of same orientation!) sorted in cartesian coordinates.
        """
        patches = patches_in.copy()
        if len(patches) != 0:
            patches_cart = [[patches[0]]]
            orientation = abs(patches[0].orientation)
            if orientation == 1:  # x
                patches.sort(key=lambda p: (p.extent.y_start, p.extent.z_start))
            elif orientation == 2:  # y
                patches.sort(key=lambda p: (p.extent.x_start, p.extent.z_start))
            elif orientation == 3:  # z
                patches.sort(key=lambda p: (p.extent.x_start, p.extent.y_start))

            if orientation == 1:
                for patch in patches[1:]:
                    if patch.extent.y_start == patches_cart[-1][-1].extent.y_start:
                        patches_cart[-1].append(patch)
                    else:
                        patches_cart.append([patch])
            else:
                for patch in patches[1:]:
                    if patch.extent.x_start == patches_cart[-1][-1].extent.x_start:
                        patches_cart[-1].append(patch)
                    else:
                        patches_cart.append([patch])
            return patches_cart
        return patches

    def get_patches_in_mesh(self, mesh: Mesh) -> List[Patch]:
        """Gets all patches in a specific mesh.
        """
        if not hasattr(self._patches[mesh][0], "_data"):
            for patch in self._patches[mesh]:
                _ = patch.data
        return self._patches[mesh]

    @property
    def faces(self) -> Dict[Literal[-3, -2, -1, 1, 2, 3], np.ndarray]:
        """Global/combines faces for all 6 orientations.
        """
        if not hasattr(self, "_faces"):
            self._prepare_faces()
        return self._faces

    @property
    def vmin(self) -> float:
        """Minimum value of all patches in any time step.
        """
        curr_min = min(np.min(b) for b in self.lower_bounds.values())
        if curr_min == 0.0:
            return min(min(np.min(p.data) for p in ps) for ps in self._patches.values())
        return curr_min

    @property
    def vmax(self) -> float:
        """Maximum value of all patches in any time step.
        """
        curr_max = max(np.max(b) for b in self.upper_bounds.values())
        if curr_max == np.float32(-1e33):
            return max(max(np.max(p.data) for p in ps) for ps in self._patches.values())
        return curr_max

    def _prepare_faces(self):
        patches_for_face = {-3: list(), -2: list(), -1: list(), 1: list(), 2: list(), 3: list()}
        for patches in self._patches.values():
            for patch in patches:
                patches_for_face[patch.orientation].append(patch)

        self._faces: Dict[Literal[-3, -2, -1, 1, 2, 3], np.ndarray] = dict()
        for face in (-3, -2, -1, 1, 2, 3):
            patches = self.sort_patches_cartesian(patches_for_face[face])
            if len(patches) == 0:
                continue

            shape_dim1 = sum([patch_row[0].shape[0] for patch_row in patches])
            shape_dim2 = sum([patch.shape[1] for patch in patches[0]])
            self._faces[face] = np.ndarray(shape=(self.n_t, shape_dim1, shape_dim2))
            dim1_pos = 0
            dim2_pos = 0
            for patch_row in patches:
                d1 = patch_row[0].shape[0]
                for patch in patch_row:
                    d2 = patch.shape[1]
                    self._faces[face][:, dim1_pos:dim1_pos + d1,
                    dim2_pos:dim2_pos + d2] = patch.data
                    dim2_pos += d2
                dim1_pos += d1
                dim2_pos = 0

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for patches in self._patches.values():
            for patch in patches:
                patch.clear_cache()

    def __getitem__(self, item):
        """Either gets all patches in a mesh [type(item)==Mesh] or all faces with a specific
            orientation [type(item)==int].
        """
        if type(item) == Mesh:
            return self.get_patches_in_mesh(item)
        else:
            return self.faces[item]

    def __repr__(self, *args, **kwargs):
        return f"Boundary(quantity={self.quantity}, cell_centered={self.cell_centered})"


class Obstruction:
    """A box-shaped obstruction with specific surfaces (materials) on each side.

    :ivar id: ID of the obstruction.
    :ivar side_surfaces: Tuple of six surfaces for each side of the cuboid.
    :ivar bound_indices: Indices used to define obstruction bounds in terms of mesh locations.
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
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    """

    def __init__(self, oid: int,
                 side_surfaces: Tuple[Surface, Surface, Surface, Surface, Surface, Surface],
                 bound_indices: Tuple[int, int, int, int, int, int], color_index: int,
                 block_type: int, texture_origin: Tuple[float, float, float],
                 rgba: Union[Tuple[()], Tuple[float, float, float, float]] = ()):
        self.id = oid
        self.side_surfaces = side_surfaces
        self.bound_indices = {'x': (bound_indices[0], bound_indices[1]),
                              'y': (bound_indices[2], bound_indices[3]),
                              'z': (bound_indices[4], bound_indices[5])}
        self.color_index = color_index
        self.block_type = block_type
        self.texture_origin = texture_origin
        if len(rgba) != 0:
            self.rgba = rgba

        self._extents: Dict[Mesh, Extent] = dict()
        self.extent: Extent = tuple()

        self._boundary_data: Dict[int, Boundary] = dict()

    def _post_init(self):
        vals = self._extents.values()
        self.extent = Extent(
            min(vals, key=lambda e: e.x_start).x_start, max(vals, key=lambda e: e.x_end).x_end,
            min(vals, key=lambda e: e.y_start).y_start, max(vals, key=lambda e: e.y_end).y_end,
            min(vals, key=lambda e: e.z_start).z_start, max(vals, key=lambda e: e.z_end).z_end)
        for boundary in self._boundary_data.values():
            boundary.extent = self.extent

    def _add_patches(self, bid: int, cell_centered: bool, quantity: str, label: str, unit: str,
                     mesh: Mesh, patches: List[Patch], times: np.ndarray, n_t: int,
                     lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = Boundary(cell_centered, Quantity(quantity, label, unit),
                                                times, n_t, lower_bounds, upper_bounds, mesh)
        self._boundary_data[bid]._add_patches(mesh, patches, lower_bounds, upper_bounds)

        if not settings.LAZY_LOAD:
            self._boundary_data[bid].get_patches_in_mesh(mesh)

    @property
    def quantities(self) -> List[Quantity]:
        """Get a list of all quantities for which boundary data exists.
        """
        return [b.quantity for b in self._boundary_data.values()]

    def get_boundary_data(self, quantity: Union[Quantity, str]):
        """Gets the boundary data for a specific quantity.
        """
        if type(quantity) != str:
            quantity = quantity.quantity
        return next((x for x in self._boundary_data.values() if
                     x.quantity.quantity.lower() == quantity.lower() or
                     x.quantity.label.lower() == quantity.lower()), None)

    @property
    def has_boundary_data(self):
        """Whether boundary data has been output in the simulation.
        """
        return len(self._boundary_data) != 0

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for bndf in self._boundary_data.values():
            bndf.clear_cache()

    def __getitem__(self, item):
        if type(item) == Quantity or type(item) == str:
            return self.get_boundary_data(item)
        return self._boundary_data[item]

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self, *args, **kwargs):
        return f"Obstruction(id={self.id}, extent={self.extent}" + \
               (f", Quantities={self.quantities}" if self.has_boundary_data else "") + ")"
