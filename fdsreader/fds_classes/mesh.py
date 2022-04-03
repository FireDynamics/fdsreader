from typing import Dict, Tuple, Sequence, List, Union
from typing_extensions import Literal
import numpy as np

from fdsreader import settings
from fdsreader.utils import Dimension, Extent, Quantity
from fdsreader.bndf import Boundary, Patch


class Mesh:
    """3-dimensional Mesh of fixed, defined size.

    :ivar coordinates: Coordinate values for each of the 3 dimension.
    :ivar dimensions: :class:`Dimension` describing the size of the 3 dimensions regarding indices.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar n: Number of elements for each of the 3 dimensions.
    :ivar n_size: Total number of blocks in this mesh.
    :var id: Mesh id/short_name assigned to this mesh.
    """
    id = None  # Needed for hash to work

    def __init__(self, coordinates: Dict[Literal['x', 'y', 'z'], np.ndarray],
                 extents: Dict[Literal['x', 'y', 'z'], Tuple[float, float]], mesh_id: str):
        """
        :param coordinates: Coordinate values of the three axes.
        :param extents: Extent of the mesh in each dimension.
        :param mesh_id: ID of this mesh.
        """
        self.id = mesh_id
        self.coordinates = coordinates
        self.dimension = Dimension(
            coordinates['x'].size if coordinates['x'].size > 0 else 1,
            coordinates['y'].size if coordinates['y'].size > 0 else 1,
            coordinates['z'].size if coordinates['z'].size > 0 else 1)

        self.n_size = self.dimension.size()
        self.extent = Extent(extents['x'][0], extents['x'][1], extents['y'][0], extents['y'][1],
                             extents['z'][0], extents['z'][1])

        self.obstructions = list()

        self._boundary_data: Dict[int, Boundary] = dict()

    def get_obstruction_mask(self, times: Sequence[float], cell_centered=False) -> np.ndarray:
        """Marks all cells which are blocked by an obstruction.

        :param times: All timesteps of the simulation.
        :returns: A 4-dimensional array with time as first and x,y,z as last dimensions. The array depends on time as
            as obstructions may be hidden as specific points in time.
        """
        shape = self.dimension.shape(cell_centered=cell_centered)
        mask = np.ones((len(times), shape[0], shape[1], shape[2]), dtype=bool)
        c = 1 if cell_centered else 0

        for obst in self.obstructions:
            for subobst in obst:
                x1, x2 = subobst.bound_indices['x']
                y1, y2 = subobst.bound_indices['y']
                z1, z2 = subobst.bound_indices['z']
                for t, _ in enumerate(subobst.visible_times(times)):
                    mask[t, x1:max(x2 + c, x1 + 1), y1:max(y2 + c, y1 + 1), z1:max(z2 + c, z1 + 1)] = False
        return mask

    def get_obstruction_mask_slice(self, subslice):
        """Marks all cells of a single slice which are blocked by an obstruction.

        :returns: A 4-dimensional array with time as first and x,y,z as last dimensions. The array depends on time as
            obstructions may be hidden as specific points in time.
        """
        orientation = subslice.orientation
        value = subslice.extent[orientation][0]
        cell_centered = subslice.cell_centered

        slc_index = self.coordinate_to_index((value,), dimension=(orientation,), cell_centered=cell_centered)[0]

        mask_indices = [slice(None)] * 4
        mask_indices[orientation] = slice(slc_index, slc_index + 1, 1)
        mask_indices = tuple(mask_indices)

        return np.squeeze(self.get_obstruction_mask(subslice.times, cell_centered=cell_centered)[mask_indices])

    def coordinate_to_index(self, coordinate: Tuple[float, ...],
                            dimension: Tuple[Literal[1, 2, 3, 'x', 'y', 'z'], ...] = ('x', 'y', 'z'),
                            cell_centered=False) -> Tuple[int, ...]:
        """Finds the nearest point in the mesh's grid and returns its indices.

        :param coordinate: Tuple of 3 floats. If the dimension parameter is supplied, up to 2
            dimensions can be left out from the tuple.
        :param dimension: The dimensions in which to return the indices (1=x, 2=y, 3=z).
        :param cell_centered: Instead of finding the nearest point on the mesh, find the center of the nearest cell.
        """
        # Convert possible integer input to chars
        dimension = tuple(('x', 'y', 'z')[dim - 1] if type(dim) == int else dim for dim in dimension)

        ret = list()
        for i, dim in enumerate(dimension):
            co = coordinate[i]
            coords = self.coordinates[dim]
            if cell_centered:
                coords = coords[:-1] + (coords[1] - coords[0]) / 2
            idx = np.searchsorted(coords, co, side="left")
            if co > 0 and (idx == len(coords) or np.math.fabs(co - coords[idx - 1]) < np.math.fabs(
                    co - coords[idx])):
                ret.append(idx - 1)
            else:
                ret.append(idx)
        return tuple(ret)

    def get_nearest_coordinate(self, coordinate: Tuple[float, ...],
                            dimension: Tuple[Literal[1, 2, 3, 'x', 'y', 'z'], ...] = ('x', 'y', 'z'),
                            cell_centered=False) -> Tuple[float, ...]:
        """Finds the nearest point in the mesh's grid.

        :param coordinate: Tuple of 3 floats. If the dimension parameter is supplied, up to 2
            dimensions can be left out from the tuple.
        :param dimension: The dimensions in which to return the indices (1=x, 2=y, 3=z).
        :param cell_centered: Instead of finding the nearest point on the mesh, find the center of the nearest cell.
        """
        indices = self.coordinate_to_index(coordinate, dimension, cell_centered)
        ret = list()
        for i, dim in enumerate(dimension):
            coords = self.coordinates[dim]
            if cell_centered:
                coords = coords[:-1] + (coords[1] - coords[0]) / 2
            ret.append(coords[indices[i]])
        return tuple(ret)

    def _add_patches(self, bid: int, cell_centered: bool, quantity: str, short_name: str, unit: str,
                     patches: List[Patch], times: np.ndarray, n_t: int, lower_bounds: np.ndarray,
                     upper_bounds: np.ndarray):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = Boundary(Quantity(quantity, short_name, unit), cell_centered, times, n_t,
                                                patches, lower_bounds, upper_bounds)

        if not settings.LAZY_LOAD:
            _ = self._boundary_data[bid].data

    def get_boundary_data(self, quantity: Union[str, Quantity]):
        if type(quantity) == Quantity:
            quantity = quantity.name
        return next(b for b in self._boundary_data.values() if
                    b.quantity.name.lower() == quantity.lower() or b.quantity.short_name.lower() == quantity.lower())


    def __getitem__(self, dimension: Literal[0, 1, 2, 'x', 'y', 'z']) -> np.ndarray:
        """Get all values in given dimension.

        :param dimension: The dimension in which to return all grid values (0=x, 1=y, 2=z).
        """
        # Convert possible integer input to chars
        if type(dimension) == int:
            dimension = ('x', 'y', 'z')[dimension]
        return self.coordinates[dimension]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self, *args, **kwargs):
        return f'Mesh(id="{self.id}", extent={str(self.extent)}, dimension={str(self.dimension)})'
