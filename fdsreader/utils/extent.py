from typing import Tuple, List

from typing_extensions import Literal


class Extent:
    """Three-dimensional value-based extent with support for a missing dimension (2D).
    """

    def __init__(self, *args, skip_dimension: Literal['x', 1, 'y', 2, 'z', 3, ''] = ''):
        self._extents = list()

        if len(args) % 2 != 0:
            ValueError("An invalid number of arguments were passed to the constructor.")
        for i in range(0, len(args), 2):
            self._extents.append((float(args[i]), float(args[i + 1])))

        if skip_dimension in ('x', 1):
            self._extents.insert(0, (0, 0))
        elif skip_dimension in ('y', 2):
            self._extents.insert(1, (0, 0))
        elif skip_dimension in ('z', 3):
            self._extents.append((0, 0))

    def __eq__(self, other):
        return self._extents == other._extents

    def __repr__(self, *args, **kwargs):
        return "Extent([{:.2f}, {:.2f}] x [{:.2f}, {:.2f}] x [{:.2f}, {:.2f}])".format(self.x_start,
                                                                                       self.x_end,
                                                                                       self.y_start,
                                                                                       self.y_end,
                                                                                       self.z_start,
                                                                                       self.z_end)

    def __getitem__(self, item):
        return self._extents[item]

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

    def as_tuple(self, reduced=True) -> Tuple:
        """Gives the extent in tuple notation (without empty extents).

        :param reduced: Whether to leave out empty extents or not.
        """
        if reduced:
            if self.x_start == self.x_end:
                return self.y_start, self.y_end, self.z_start, self.z_end
            elif self.y_start == self.y_end:
                return self.x_start, self.x_end, self.z_start, self.z_end
            elif self.z_start == self.z_end:
                return self.x_start, self.x_end, self.y_start, self.y_end
        return self.x_start, self.x_end, self.y_start, self.y_end, self.z_start, self.z_end

    def as_list(self, reduced=True) -> List:
        """Gives the extent in list notation (without empty extents).

        :param reduced: Whether to leave out empty extents or not.
        """
        return list(self.as_tuple(reduced))
