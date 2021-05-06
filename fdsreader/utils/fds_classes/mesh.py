from typing import Dict, Tuple, Sequence
from typing_extensions import Literal
import numpy as np

from fdsreader.utils import Dimension, Extent


class Mesh:
    """3-dimensional Mesh of fixed, defined size.

    :ivar coordinates: Coordinate values for each of the 3 dimension.
    :ivar dimensions: :class:`Dimension` describing the size of the 3 dimensions regarding indices.
    :ivar extent: :class:`Extent` object containing 3-dimensional extent information.
    :ivar n: Number of elements for each of the 3 dimensions.
    :ivar n_size: Total number of blocks in this mesh.
    :var id: Mesh id/label assigned to this mesh.
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
            as obstructions may be hidden as specific points in time.
        """
        orientation = subslice.orientation
        value = subslice.extent[orientation - 1][0]
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
        """
        # Convert possible integer input to chars
        dimension = tuple(('x', 'y', 'z')[dim-1] if type(dim) == int else dim for dim in dimension)

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
