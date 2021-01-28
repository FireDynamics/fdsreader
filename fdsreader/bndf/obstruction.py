from operator import add
from typing import List, Dict, Tuple, Union, Literal
import numpy as np

from fdsreader.utils import Surface, Mesh, Extent, Quantity, Dimension
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings


class Patch:
    """Container for the actual data which is stored as rectangular plane with specific orientation
        and extent.

    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar orientation: The direction the patch is facing (x={-1;1}, y={-2;2}, z={-3;3}).
    :ivar data: Numpy ndarray with the actual data.
    :ivar t_n: Total number of time steps for which output data has been written.
    """

    def __init__(self, file_path: str, dimension: Dimension, extent: Extent, orientation: int,
                 cell_centered: bool, initial_offset: int, t_n: int):
        self.file_path = file_path
        self.dimension = dimension
        self.extent = extent
        self.orientation = orientation
        self.cell_centered = cell_centered
        self.initial_offset = initial_offset
        self.time_offset = -1
        self.t_n = t_n

    @property
    def shape(self) -> Tuple:
        """Convenience function to calculate the shape of the array containing data for this patch.
        """
        return self.dimension.shape(self.cell_centered)

    def _post_init(self, time_offset: int):
        """Fully initialize the patch as soon as the number of timesteps is known.
        """
        self.time_offset = time_offset

    @property
    def data(self):
        """Method to load the quantity data for a single patch for a single timestep.
        """
        if not hasattr(self, "_data"):
            self._data = np.empty((self.t_n,) + self.shape)
            dtype_data = fdtype.new((('f', str(self.shape)),))
            with open(self.file_path, 'rb') as infile:
                for t in range(self.t_n):
                    infile.seek(self.initial_offset + t * self.time_offset)
                    self._data[t, :] = np.fromfile(infile, dtype_data, 1)[0][1].reshape(self.shape,
                                                                                        order='F')
        return self._data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "_data"):
            del self._data

    def __repr__(self, *args, **kwargs):
        return f"Patch(shape={self.shape}, orientation={self.orientation}, extent={self.extent})"


class Boundary:
    def __init__(self, cell_centered: bool, quantity: Quantity, times: np.ndarray, t_n: int,
                 lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self.cell_centered = cell_centered
        self.quantity = quantity
        self.times = times
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.t_n = t_n
        self.extent = None

        self._patches: Dict[Mesh, List[Patch]] = dict()

    def _add_patches(self, mesh: Mesh, patches: List[Patch]):
        self._patches[mesh] = patches

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

    def get_patches_in_mesh(self, mesh: Mesh):
        """
        """
        if not hasattr(self._patches[mesh][0], "_data"):
            for patch in self._patches[mesh]:
                _ = patch.data
        return self._patches[mesh]

    @property
    def faces(self) -> Dict[Literal[-3, -2, -1, 1, 2, 3], np.ndarray]:
        """
        """
        if not hasattr(self, "_faces"):
            self._prepare_faces()
        return self._faces

    def _prepare_faces(self):
        patches_for_face = {-3: list(), -2: list(), -1: list(), 1: list(), 2: list(), 3: list()}
        for patches in self._patches.values():
            for patch in patches:
                patches_for_face[patch.orientation].append(patch)

        self._faces: Dict[Literal[-3, -2, -1, 1, 2, 3], np.ndarray] = dict()
        # for face in (-3, -2, -1, 1, 2, 3):
        for face in (3,):
            patches = self.sort_patches_cartesian(patches_for_face[face])
            if len(patches) == 0:
                continue

            shape_dim1 = sum([patch_row[0].shape[0] for patch_row in patches])
            shape_dim2 = sum([patch.shape[1] for patch in patches[0]])
            self._faces[face] = np.ndarray(shape=(self.t_n, shape_dim1, shape_dim2))
            # print(patches)
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

    def __getitem__(self, item):
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
    """

    def __init__(self, oid: int,
                 side_surfaces: Tuple[Surface, Surface, Surface, Surface, Surface, Surface],
                 bound_indices: Tuple[int, int, int, int, int, int], color_index: int,
                 block_type: int, texture_origin: Tuple[float, float, float],
                 rgba: Union[Tuple[()], Tuple[float, float, float, float]] = ()):
        self.id = oid
        self.side_surfaces = side_surfaces
        self.bound_indices = bound_indices
        self.color_index = color_index
        self.block_type = block_type
        self.texture_origin = texture_origin
        if len(rgba) != 0:
            self.rgba = rgba

        self._extents: Dict[Mesh, Extent] = dict()
        self.extent = tuple()

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
                     mesh: Mesh, patches: List[Patch], times: np.ndarray, t_n: int,
                     lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = Boundary(cell_centered, Quantity(quantity, label, unit),
                                                times, t_n, lower_bounds, upper_bounds)
        self._boundary_data[bid]._add_patches(mesh, patches)

        if not settings.LAZY_LOAD:
            self._boundary_data[bid].get_patches_in_mesh(mesh)

    @property
    def quantities(self) -> List[Quantity]:
        """Get a list of all quantities for which boundary data exists.
        """
        return [b.quantity for b in self._boundary_data.values()]

    def get_boundary_data(self, quantity: Union[Quantity, str]):
        if type(quantity) == str:
            return next(x for x in self._boundary_data.values() if
                        x.quantity.quantity.lower() == quantity.lower())
        return next((x for x in self._boundary_data.values() if x.quantity == quantity), None)

    @property
    def has_boundary_data(self):
        return len(self._boundary_data) != 0

    def __getitem__(self, item):
        if type(item) == Quantity or type(item) == str:
            return self.get_boundary_data(item)
        return self._boundary_data[item]

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self, *args, **kwargs):
        return f"Obstruction(id={self.id}, extent={self.extent}" + \
               (f", Quantities={self.quantities}" if self.has_boundary_data else "") + ")"
