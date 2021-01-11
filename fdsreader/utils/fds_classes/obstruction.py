from operator import add
from typing import List, Dict, Tuple, Union
import numpy as np

from fdsreader.utils import Surface, Mesh, Extent, Quantity, Dimension
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings


class Patch:
    """Container for the actual data which is stored as rectangular plane with specific orientation
        and extent.

    :ivar extent: Extent object containing 3-dimensional extent information.
    :ivar orientation: The direction the patch is facing (x={-1;1}, y={-2;2}, z={-3;3}).
    :ivar data: Numpy ndarray with the actual data.
    :ivar t_n: Total number of time steps for which output data has been written.
    """

    def __init__(self, file_path: str, extent: Dimension, orientation: int, cell_centered: bool,
                 initial_offset: int):
        self.file_path = file_path
        self.extent = extent
        self.orientation = orientation
        self.cell_centered = cell_centered
        self.initial_offset = initial_offset
        self.t_n = -1
        self.time_offset = -1

    @property
    def shape(self) -> Tuple:
        """Convenience function to calculate the shape of the array containing data for this patch.
        """
        if abs(self.orientation) == 1:
            dim = tuple(map(add, self.extent.shape(self.cell_centered), (0, 2, 2)))
        elif abs(self.orientation) == 2:
            dim = tuple(map(add, self.extent.shape(self.cell_centered), (2, 0, 2)))
        else:
            dim = tuple(map(add, self.extent.shape(self.cell_centered), (2, 2, 0)))
        return dim

    def _post_init(self, t_n: int, time_offset: int):
        """Fully initialize the patch as soon as the number of timesteps is known.
        """
        self.time_offset = time_offset
        self.t_n = t_n

    @property
    def data(self):
        """Method to load the quantity data for a single patch for a single timestep.
        """
        if not hasattr(self, "_data"):
            self._data = np.empty((self.t_n,) + self.shape)
            dtype_data = fdtype.new((('f', str(self.shape)),))
            with open(self.file_path, 'rb') as infile:
                for t in range(self.t_n):
                    self._data[t, :] = fdtype.read(infile, dtype_data, 1)
        return self._data

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "_data"):
            del self._data

    def __repr__(self, *args, **kwargs):
        return f"Patch(shape={self.shape}, orientation={self.orientation})"


class Boundary:
    def __init__(self, cell_centered: bool, quantity: Quantity, times: np.ndarray, t_n: int):
        self.cell_centered = cell_centered
        self.quantity = quantity
        self.times = times
        self.t_n = t_n

        self._patches: Dict[Mesh, List[Patch]] = dict()

    def _add_patches(self, mesh: Mesh, patches: List[Patch]):
        self._patches[mesh] = patches

    def get_patches(self, mesh: Mesh):
        if not hasattr(self._patches[mesh][0], "_data"):
            for patch in self._patches[mesh]:
                _ = patch.data
        return self._patches[mesh]


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

        self._extent = tuple()
        self._extents: Dict[Mesh, Extent] = dict()

        self._boundary_data: Dict[int, Boundary] = dict()

    @property
    def extent(self):
        if len(self._extent) == 0:
            self._extent = (min(self._extents, key=lambda e: e.x_start),
                            min(self._extents, key=lambda e: e.y_start),
                            min(self._extents, key=lambda e: e.z_start),
                            max(self._extents, key=lambda e: e.x_end),
                            max(self._extents, key=lambda e: e.y_end),
                            max(self._extents, key=lambda e: e.z_end))
        return self._extent

    def _add_patches(self, bid: int, cell_centered: bool, quantity: str, label: str, unit: str,
                     mesh: Mesh, patches: List[Patch], times: np.ndarray, t_n: int):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = Boundary(cell_centered, Quantity(quantity, label, unit),
                                                times, t_n)
        self._boundary_data[bid]._add_patches(mesh, patches)

        if not settings.LAZY_LOAD:
            self._boundary_data[bid].get_patches(mesh)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self, *args, **kwargs):
        return f"Obstruction(id={self.id})"
