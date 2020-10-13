import numpy as np

from utils import Extent


class Mesh:
    """
    3-dimensional Mesh of fixed, defined size.
    :ivar coordinates: Coordinate values for each of the 3 dimension.
    :ivar extent: Tuple with three tuples containing minimum and maximum coordinate value on the
                  corresponding dimension.
    :ivar mesh:
    :ivar n: Number of elements for each of the 3 dimensions.
    :ivar n_size: Total number of blocks in this mesh.
    :ivar label: Label associated with this mesh.
    """

    def __init__(self, x_coordinates, y_coordinates, z_coordinates, label):
        """
        :param x_coordinates: Coordinate values of x-axis.
        :param y_coordinates: Coordinate values of y-axis.
        :param z_coordinates: Coordinate values of z-axis.
        :param label: Label associated with this mesh.
        """
        self.coordinates = [x_coordinates, y_coordinates, z_coordinates]
        self.extent = Extent(x_coordinates[0], x_coordinates[-1], y_coordinates[0],
                             y_coordinates[-1], z_coordinates[0], z_coordinates[-1])
        # Todo: Does this really do what it is supposed to do? What is it even supposed to do?
        # Todo: Numpy: Deprecated
        self.mesh = np.meshgrid(self.coordinates)

        self.n = [x_coordinates.size, y_coordinates.size, z_coordinates.size]
        self.n_size = self.n[0] * self.n[1] * self.n[2]

        self.label = label

    def __str__(self, *args, **kwargs):
        return "{}, {} x {} x {}, ".format(self.label, self.n[0], self.n[1], self.n[2]) + str(
            self.extent)
