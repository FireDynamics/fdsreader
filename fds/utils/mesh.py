import logging
import mmap
from typing import Iterator

import numpy as np

from fds.utils import FDS_DATA_TYPE_FLOAT


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
        self.extent = ((x_coordinates[0], x_coordinates[-1]),
                       (y_coordinates[0], y_coordinates[-1]),
                       (z_coordinates[0], z_coordinates[-1]))
        # Todo: Does this really do what it is supposed to do? What is it even supposed to do?
        self.mesh = np.meshgrid(self.coordinates)

        self.n = [x_coordinates.size, y_coordinates.size, z_coordinates.size]
        self.n_size = self.n[0] * self.n[1] * self.n[2]

        self.label = label

    def __str__(self, *args, **kwargs):
        return "{}, {} x {} x {}, [{}, {}] x [{}, {}] x [{}, {}]".format(self.label,
                                                                         self.n[0], self.n[1],
                                                                         self.n[2],
                                                                         self.extent[0][0],
                                                                         self.extent[0][1],
                                                                         self.extent[1][0],
                                                                         self.extent[1][1],
                                                                         self.extent[2][0],
                                                                         self.extent[2][1])


class MeshCollection:
    """
    Creates a collection of meshes by collecting all mesh (GRID) information in a given .smv file.
    """

    def __init__(self, file_path: str):
        """
        :param file_path: Path to the .smv file containing information about the meshes.
        """
        self._meshes = list()

        logging.debug("scanning smv file for meshes: %s", file_path)

        with open(file_path, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                       access=mmap.ACCESS_READ) as smv_file:
            float_dtype = np.dtype(FDS_DATA_TYPE_FLOAT)
            pos = smv_file.find(b'GRID')

            while pos > 0:
                smv_file.seek(pos)

                label = smv_file.readline().split()[1].decode()
                logging.debug("found MESH with label: %s", label)

                grid_numbers = smv_file.readline().split()
                nx = int(grid_numbers[0]) + 1
                # # correct for 2D cases
                # if nx == 2: nx = 1
                ny = int(grid_numbers[1]) + 1
                # if ny == 2: ny = 1
                nz = int(grid_numbers[2]) + 1
                # if nz == 2: nz = 1

                logging.debug("number of cells: %i x %i x %i", nx, ny, nz)

                pos = smv_file.find(b'TRNX', pos + 1)
                smv_file.seek(pos)
                smv_file.readline()
                smv_file.readline()
                x_coordinates = np.empty(nx, dtype=float_dtype)
                for ix in range(nx):
                    x_coordinates[ix] = smv_file.readline().split()[1]

                pos = smv_file.find(b'TRNY', pos + 1)
                smv_file.seek(pos)
                smv_file.readline()
                smv_file.readline()
                y_coordinates = np.empty(ny, dtype=float_dtype)
                for iy in range(ny):
                    y_coordinates[iy] = smv_file.readline().split()[1]

                pos = smv_file.find(b'TRNZ', pos + 1)
                smv_file.seek(pos)
                smv_file.readline()
                smv_file.readline()
                z_coordinates = np.empty(nz, dtype=float_dtype)
                for iz in range(nz):
                    z_coordinates[iz] = smv_file.readline().split()[1]

                self._meshes.append(Mesh(x_coordinates, y_coordinates, z_coordinates, label))

                pos = smv_file.find(b'GRID', pos + 1)

    def __str__(self, *args, **kwargs):
        return '\n'.join((str(m) for m in self))

    def __getitem__(self, item: int):
        return self._meshes[item]

    def __iter__(self) -> Iterator[Mesh]:
        return iter(self._meshes)
