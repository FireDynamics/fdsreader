from typing_extensions import Literal


class Extent:
    """
    Three-dimensional extent with support for a missing dimension (size of 1).
    """

    def __init__(self, *args, skip_dimension: Literal['x', 'y', 'z', ''] = ''):
        self._extents = list()

        if len(args) % 2 == 1:
            ValueError("An uneven number of ranges were passed to the constructor.")
        for i in range(0, len(args), 2):
            self._extents.append((int(args[i]), int(args[i + 1])))

        if skip_dimension == 'x':
            self._extents.insert(0, (0, 0))
        elif skip_dimension == 'y':
            self._extents.insert(1, (0, 0))
        elif skip_dimension == 'z':
            self._extents.append((0, 0))

    def __eq__(self, other):
        return self._extents == other._extents

    def __str__(self):
        return "[{}, {}] x [{}, {}] x [{}, {}]".format(self.x_start, self.x_end, self.y_start,
                                                       self.y_end, self.z_start, self.z_end)

    def size(self, cell_centered=False):
        if cell_centered:
            x = self.x-1 if self.x != 1 else 1
            y = self.y-1 if self.y != 1 else 1
            z = self.z-1 if self.z != 1 else 1
            return x * y * z
        else:
            return self.x * self.y * self.z

    @property
    def x(self):
        """
        Gives the number of data points in x-direction (end is inclusive).
        """
        return self._extents[0][1] - self._extents[0][0] + 1

    @property
    def y(self):
        """
        Gives the number of data points in y-direction (end is inclusive).
        """
        return self._extents[1][1] - self._extents[1][0] + 1

    @property
    def z(self):
        """
        Gives the number of data points in z-direction (end is inclusive).
        """
        return self._extents[2][1] - self._extents[2][0] + 1

    @property
    def x_start(self):
        """
        Gives the absolute extent in x-direction.
        """
        return self._extents[0][0]

    @property
    def y_start(self):
        """
        Gives the absolute extent in y-direction.
        """
        return self._extents[1][0]

    @property
    def z_start(self):
        """
        Gives the absolute extent in z-direction.
        """
        return self._extents[2][0]

    @property
    def x_end(self):
        """
        Gives the absolute extent in x-direction.
        """
        return self._extents[0][1]

    @property
    def y_end(self):
        """
        Gives the absolute extent in y-direction.
        """
        return self._extents[1][1]

    @property
    def z_end(self):
        """
        Gives the absolute extent in z-direction.
        """
        return self._extents[2][1]
