from typing import Dict, Tuple
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
    :ivar id: Mesh id/label assigned to this mesh.
    """

    def __init__(self, coordinates: Dict[str, np.ndarray], extents: Dict[str, Tuple[float, float]],
                 mesh_id: str):
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

    def __getitem__(self, dimension: Literal[0, 1, 2, 'x', 'y', 'z']) -> np.ndarray:
        """Get all values in given dimension.

        :param dimension: The dimension in which to return all grid values (0=x, 1=y, 2=z).
        """
        if type(dimension) == int:
            dimension = ('x', 'y', 'z')[dimension]
        return self.coordinates[dimension]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self, *args, **kwargs):
        return f'Mesh(id="{self.id}", extent={str(self.extent)}, dimension={str(self.dimension)})'
