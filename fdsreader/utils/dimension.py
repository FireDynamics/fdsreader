from functools import reduce
from operator import mul
from typing import Tuple, List

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
            dimensions.insert(0, 1)
        elif skip_dimension in ('y', 2):
            dimensions.insert(1, 1)
        elif skip_dimension in ('z', 3):
            dimensions.append(1)

        self.x = dimensions[0]
        self.y = dimensions[1]
        self.z = dimensions[2]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self, *args, **kwargs):
        return f"Dimension({self.x}, {self.y}, {self.z})"

    def __getitem__(self, dimension: Literal[0, 1, 2, 'x', 'y', 'z']):
        if type(dimension) == int:
            dimension = ('x', 'y', 'z')[dimension]
        return self.__dict__[dimension]

    def size(self, cell_centered=False):
        return reduce(mul, self.shape(cell_centered))

    def shape(self, cell_centered=False) -> Tuple:
        """Method to get the actual number of data points per dimension.
        """
        s = list()
        c = -1 if cell_centered else 0
        if self.x != 1:
            s.append(self.x + c)
        if self.y != 1:
            s.append(self.y + c)
        if self.z != 1:
            s.append(self.z + c)
        return tuple(s)

    def as_tuple(self, cell_centered=False, reduced=True) -> Tuple:
        """Gives the dimensions in tuple notation (optionally without empty dimensions).

        :param reduced: Whether to leave out empty dimensions (size of 1) or not.
        """
        c = -1 if cell_centered else 0
        if reduced:
            if self.x == 1:
                return self.y + c, self.z + c
            elif self.y == 1:
                return self.x + c, self.z + c
            elif self.z == 1:
                return self.x + c, self.y + c
        return self.x + c, self.y + c, self.z + c

    def as_list(self, cell_centered=False, reduced=True) -> List:
        """Gives the dimension in list notation (without empty extents).

        :param reduced: Whether to leave out empty dimensions (size of 1) or not.
        """
        return list(self.as_tuple(cell_centered, reduced))
