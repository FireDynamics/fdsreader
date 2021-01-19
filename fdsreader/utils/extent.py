from functools import reduce
from operator import mul
from typing import Tuple

from typing_extensions import Literal


class Extent:
    """Three-dimensional value-based extent with support for a missing dimension (2D).
    """

    def __init__(self, *args, skip_dimension: Literal['x', 1, 'y', 2, 'z', 3, ''] = ''):
        self._extents = list()
        self._step_sizes = list()

        if len(args) % 3 != 0:
            ValueError("An invalid number of arguments were passed to the constructor.")
        for i in range(0, len(args), 3):
            self._extents.append((float(args[i]), float(args[i + 1])))
            self._step_sizes.append(float(args[i + 2]))

        if skip_dimension in ('x', 1):
            self._extents.insert(0, (0, 0))
            self._step_sizes.insert(0, 0)
        elif skip_dimension in ('y', 2):
            self._extents.insert(1, (0, 0))
            self._step_sizes.insert(1, 0)
        elif skip_dimension in ('z', 3):
            self._extents.append((0, 0))
            self._step_sizes.append(0)

    def __eq__(self, other):
        return self._extents == other._extents and self._step_sizes == other._step_sizes

    def __repr__(self, *args, **kwargs):
        return "Extent([{:.2f}, {:.2f}] x [{:.2f}, {:.2f}] x [{:.2f}, {:.2f}])".format(self.x_start, self.x_end,
                                                              self.y_start, self.y_end,
                                                              self.z_start, self.z_end)

    def __getitem__(self, dimension: Literal[0, 1, 2, 'x', 'y', 'z']):
        if type(dimension) == int:
            dimension = ('x', 'y', 'z')[dimension]
        return self.__dict__[dimension]

    @property
    def x(self):
        """Gives the number of data points in x-direction (end is inclusive).
        """
        return round((self._extents[0][1] - self._extents[0][0]) / self._step_sizes[0]) + 1

    @property
    def y(self):
        """Gives the number of data points in y-direction (end is inclusive).
        """
        return round((self._extents[1][1] - self._extents[1][0]) / self._step_sizes[0]) + 1

    @property
    def z(self):
        """Gives the number of data points in z-direction (end is inclusive).
        """
        return round((self._extents[2][1] - self._extents[2][0]) / self._step_sizes[0]) + 1

    @property
    def x_start(self):
        """Gives the absolute extent in x-direction.
        """
        return self._extents[0][0]

    @property
    def y_start(self):
        """Gives the absolute extent in y-direction.
        """
        return self._extents[1][0]

    @property
    def z_start(self):
        """Gives the absolute extent in z-direction.
        """
        return self._extents[2][0]

    @property
    def x_end(self):
        """Gives the absolute extent in x-direction.
        """
        return self._extents[0][1]

    @property
    def y_end(self):
        """Gives the absolute extent in y-direction.
        """
        return self._extents[1][1]

    @property
    def z_end(self):
        """Gives the absolute extent in z-direction.
        """
        return self._extents[2][1]
