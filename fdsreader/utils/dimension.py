from functools import reduce
from operator import mul
from typing import Tuple

from typing_extensions import Literal


class Dimension:
    """Three-dimensional index-based extent with support for a missing dimension (2D).

    :ivar x: Number of data points in x-direction (end is exclusive).
    :ivar y: Number of data points in y-direction (end is exclusive).
    :ivar z: Number of data points in z-direction (end is exclusive).
    """

    def __init__(self, *args, skip_dimension: Literal['x', 1, 'y', 2, 'z', 3, ''] = ''):
        dimensions = list(args)

        if skip_dimension in ('x', 1):
            dimensions.insert(0, 0)
        elif skip_dimension in ('y', 2):
            dimensions.insert(1, 0)
        elif skip_dimension in ('z', 3):
            dimensions.append(0)

        self.x = dimensions[0]
        self.y = dimensions[1]
        self.z = dimensions[2]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self, *args, **kwargs):
        return f"Dimension(shape={self.shape})"

    def __getitem__(self, dimension: Literal[0, 1, 2, 'x', 'y', 'z']):
        if type(dimension) == int:
            dimension = ('x', 'y', 'z')[dimension]
        return self.__dict__[dimension]

    def size(self, cell_centered=False):
        return reduce(mul, self.shape(cell_centered))

    def shape(self, cell_centered=False) -> Tuple[int, int, int]:
        c = 0 if cell_centered else 1
        x = self.x + c if self.x != 0 else 1
        y = self.y + c if self.y != 0 else 1
        z = self.z + c if self.z != 0 else 1
        return x, y, z
