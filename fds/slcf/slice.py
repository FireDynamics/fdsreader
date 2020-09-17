"""
Defines classes and methods to load slice-files and work with sliced data.
"""

import os
import mmap
import logging
from typing import List, Tuple, Union, Iterator
from typing_extensions import Literal
import numpy as np

# As the binary representation of raw data is compiler dependent, this information must be provided
# by the user
FDS_DATA_TYPE_INTEGER = "i4"  # i4 -> 32 bit integer (native endiannes, probably little-endian)
FDS_DATA_TYPE_FLOAT = "f4"  # f4 -> 32 bit floating point (native endiannes, probably little-endian)
FDS_DATA_TYPE_CHAR = "a"  # a -> 8 bit character
FDS_FORTRAN_BACKWARD = True  # sets weather the blocks are ended with the size of the block


def get_slice_type(name: Literal['header', 'index', 'time', 'data'],
                   n_size=0) -> np.dtype:
    """
    Convenience function to get the numpy datatype for a given type name.
    :param name: The data type name.
    :param n_size: Required argument for the 'data' type that defines the total size of the slice.
    :return: Numpy datatype for given name.
    """
    type_slice_str = FDS_DATA_TYPE_INTEGER + ", "
    if name == 'header':
        type_slice_str += "30" + FDS_DATA_TYPE_CHAR
    elif name == 'index':
        type_slice_str += "6" + FDS_DATA_TYPE_INTEGER
    elif name == 'time':
        type_slice_str += FDS_DATA_TYPE_FLOAT
    elif name == 'data':
        type_slice_str += "(%i)%s" % (n_size, FDS_DATA_TYPE_FLOAT)
    else:
        raise ValueError("Parameter name (" + str(
            name) + ") has to be one of ('header', 'index', 'time', 'data')!")
    if FDS_FORTRAN_BACKWARD:
        type_slice_str += ", " + FDS_DATA_TYPE_INTEGER
    return np.dtype(type_slice_str)


class SliceMesh:
    """

    """

    def __init__(self, x1, x2, normal_direction: Literal['x', 'y', 'z', 0, 1, 2],
                 norm_offset: float):
        """

        :param x1:
        :param x2:
        :param normal_direction:
        :param norm_offset:
        """

        if normal_direction in ('x', 0):
            self.directions = [1, 2]
        elif normal_direction in ('y', 1):
            self.directions = [0, 2]
        elif normal_direction in ('z', 2):
            self.directions = [0, 1]
        else:
            raise ValueError("Parameter normal_direction  (" + str(
                normal_direction) + ") has to be one of (x,y,z,0,1,2)!")

        self.normal_direction = normal_direction
        self.norm_offset = norm_offset

        self.axes = [x1, x2]
        self.mesh = np.meshgrid(x1, x2)

        self.extent = (np.min(x1), np.max(x1), np.min(x2), np.max(x2))

        self.n = [x1.size, x2.size]
        self.n_size = self.n[0] * self.n[1]


class Mesh:
    """

    """

    def __init__(self, x1, x2, x3, label):
        """

        :param x1:
        :param x2:
        :param x3:
        :param label:
        """
        self.axes = [x1, x2, x3]
        self.ranges = [[x1[0], x1[-1]], [x2[0], x2[-1]], [x3[0], x3[-1]]]
        self.mesh = np.meshgrid(self.axes)

        self.n = [x1.size, x2.size, x3.size]
        self.n_size = self.n[0] * self.n[1] * self.n[2]

        self.label = label

    def __str__(self, *args, **kwargs):
        return "{}, {} x {} x {}, [{}, {}] x [{}, {}] x [{}, {}]".format(self.label,
                                                                         self.n[0], self.n[1],
                                                                         self.n[2],
                                                                         self.ranges[0][0],
                                                                         self.ranges[0][1],
                                                                         self.ranges[1][0],
                                                                         self.ranges[1][1],
                                                                         self.ranges[2][0],
                                                                         self.ranges[2][1])

    def extract_slice_mesh(self,
                           normal_direction: Literal['x', 'y', 'z', 0, 1, 2],
                           norm_index: int) -> SliceMesh:
        """

        :param normal_direction :
        :param norm_index:
        :return:
        """
        if normal_direction in ('x', 0):
            offset = self.axes[0][norm_index]
            return SliceMesh(self.axes[1], self.axes[2], normal_direction, offset)
        if normal_direction in ('y', 1):
            offset = self.axes[1][norm_index]
            return SliceMesh(self.axes[0], self.axes[2], normal_direction, offset)
        if normal_direction in ('z', 2):
            offset = self.axes[2][norm_index]
            return SliceMesh(self.axes[0], self.axes[1], normal_direction, offset)
        raise ValueError("Parameter normal_direction  (" + str(
            normal_direction) + ") has to be one of (x,y,z,0,1,2)!")


class MeshCollection:
    """
    Creates a collection of slices by collecting all mesh (GRID) information in a given .smv file.
    :param file_path: Path to the .smv file containing information about the meshes.
    """

    def __init__(self, file_path: str):
        self._meshes = list()

        logging.debug("scanning smv file for meshes: %s", file_path)

        with open(file_path, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                       access=mmap.ACCESS_READ) as smv_file:
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
                x_coordinate = np.empty(nx, dtype=np.dtype(FDS_DATA_TYPE_FLOAT))
                for ix in range(nx):
                    x_coordinate[ix] = float(smv_file.readline().split()[1])

                pos = smv_file.find(b'TRNY', pos + 1)
                smv_file.seek(pos)
                smv_file.readline()
                smv_file.readline()
                y_coordinate = np.empty(ny, dtype=np.dtype(FDS_DATA_TYPE_FLOAT))
                for iy in range(ny):
                    y_coordinate[iy] = float(smv_file.readline().split()[1])

                pos = smv_file.find(b'TRNZ', pos + 1)
                smv_file.seek(pos)
                smv_file.readline()
                smv_file.readline()
                z_coordinate = np.empty(nz, dtype=np.dtype(FDS_DATA_TYPE_FLOAT))
                for iz in range(nz):
                    z_coordinate[iz] = float(smv_file.readline().split()[1])

                self._meshes.append(Mesh(x_coordinate, y_coordinate, z_coordinate, label))

                pos = smv_file.find(b'GRID', pos + 1)

    def __str__(self, *args, **kwargs):
        return '\n'.join((str(m) for m in self))

    def __getitem__(self, item: int):
        return self._meshes[item]

    def __iter__(self) -> Iterator[Mesh]:
        return iter(self._meshes)


class Slice:
    """

    """
    offset = 3 * get_slice_type('header').itemsize + get_slice_type('index').itemsize

    def __init__(self, quantity: str, label: str, units: str, filename: str, mesh_id: int,
                 index_ranges: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                 centered: bool, root: str = '.'):
        self.root = root
        self.quantity = quantity
        self.label = label
        self.units = units
        self.filename = filename
        self.mesh_id = mesh_id
        self.index_ranges = index_ranges
        self.centered = centered

        self.read_size = (index_ranges[0][1] - index_ranges[0][0] + 1) * \
                         (index_ranges[1][1] - index_ranges[1][0] + 1) * \
                         (index_ranges[2][1] - index_ranges[2][0] + 1)

        self.stride = get_slice_type('time').itemsize + get_slice_type('data',
                                                                       self.read_size).itemsize
        # Numpy 2D-array with time on the first and all data values on the second axis
        self.data_raw = None
        # Numpy array with all times for which data has been loaded
        self.times = None
        # Numpy array with all times in this slice file
        self._all_times = None
        # The SliceMesh
        self.slice_mesh = None
        #
        self.sd = None

        infile = open(os.path.join(self.root, self.filename), 'r')
        infile.seek(0, 0)

        self.header = np.fromfile(infile, dtype=get_slice_type('header'), count=3)
        logging.debug("slice header: %s", self.header)
        self.index = np.fromfile(infile, dtype=get_slice_type('index'), count=1)[0]
        logging.debug("slice index: %i", str(self.index))

        if self.centered:
            self.n_size = 1
            if (index_ranges[0][1] - index_ranges[0][0]) > 0:
                self.n_size *= index_ranges[0][1] - index_ranges[0][0]
            if (index_ranges[1][1] - index_ranges[1][0]) > 0:
                self.n_size *= index_ranges[1][1] - index_ranges[1][0]
            if (index_ranges[2][1] - index_ranges[2][0]) > 0:
                self.n_size *= index_ranges[2][1] - index_ranges[2][0]
        else:
            self.n_size = self.read_size

        if index_ranges[0][0] == index_ranges[0][1]:
            self.normal_direction = 0
            self.norm_offset = index_ranges[0][0]
        if index_ranges[1][0] == index_ranges[1][1]:
            self.normal_direction = 1
            self.norm_offset = index_ranges[1][0]
        if index_ranges[2][0] == index_ranges[2][1]:
            self.normal_direction = 2
            self.norm_offset = index_ranges[2][0]

    @property
    def all_times(self):
        if self._all_times is None:
            self._initialize_times()
        return self._all_times

    def _initialize_times(self):
        """
        Reads in all time information.
        :return:
        """
        count = 0
        times = list()
        with open(os.path.join(self.root, self.filename), 'r') as slcf:
            while True:
                slcf.seek(Slice.offset + count * self.stride)
                slice_time_raw = np.fromfile(slcf, dtype=get_slice_type('time'), count=1)

                if len(slice_time_raw) == 0:
                    break

                slice_time = slice_time_raw[0][1]

                times.append(slice_time)
                count += 1

        self._all_times = np.array(times)

    def read_data(self, dt=-1, average_dt=-1):
        """
        Reading in the slice's data, optionally only for a specific selection.
        :param dt:
        :param average_dt:
        :return:
        """
        type_time = get_slice_type('time')
        type_data = get_slice_type('data', self.read_size)
        stride = type_time.itemsize + type_data.itemsize

        if dt != -1:
            n_slices = int(self.all_times[-1] / dt)
        else:
            n_slices = self.all_times.size

        self.times = np.zeros(n_slices)

        self.data_raw = np.zeros((n_slices, self.read_size))

        with open(os.path.join(self.root, self.filename), 'r') as infile:
            for t in range(n_slices):
                if dt != -1:
                    central_time_index = np.where(self.all_times > t * dt)[0][0]

                    logging.debug("central index, time: %i, %i", central_time_index,
                                  self.all_times[central_time_index])

                    self.times[t] = self.all_times[central_time_index]
                    time_indices = central_time_index

                    if average_dt != -1:
                        time_indices = np.where((self.all_times > t * dt - average_dt / 2.0) & (
                                self.all_times < t * dt + average_dt / 2.0))[0]
                else:
                    time_indices = (t,)
                logging.debug("using time indices: %s", time_indices)

                for st in time_indices:
                    infile.seek(Slice.offset + st * stride)
                    slice_data = np.fromfile(infile, dtype=type_data, count=1)
                    self.data_raw[t, :] += slice_data[0][1]

                self.data_raw[t, :] /= len(time_indices)

    def map_data(self, mesh_col: MeshCollection):
        """

        :param mesh_col:
        :return:
        """
        mesh = mesh_col[self.mesh_id]
        self.slice_mesh = mesh.extract_slice_mesh(self.normal_direction, self.norm_offset)

        n1 = self.slice_mesh.n[0]
        n2 = self.slice_mesh.n[1]
        if self.centered:
            n1 -= 1
            n2 -= 1

        self.sd = np.zeros((self.times.size, n2, n1))
        for i in range(self.times.size):
            if self.centered:
                self.sd[i] = np.reshape(self.data_raw[i], (n2, n1))[1:, 1:]
            else:
                self.sd[i] = np.reshape(self.data_raw[i], (n2, n1))

    def __str__(self, *args, **kwargs):
        ret_str = "%s, %s, %s, mesh_id: %i, [%i, %i] x [%i, %i] x [%i, %i]" % (
            self.label, self.quantity, self.units, self.mesh_id, self.index_ranges[0][0],
            self.index_ranges[0][1], self.index_ranges[1][0], self.index_ranges[1][1],
            self.index_ranges[2][0], self.index_ranges[2][1])

        if self.times is not None:
            ret_str += ", times: {}, [{}, {}]".format(self.times.size, self.times[0],
                                                       self.times[-1])

        if self.data_raw is not None:
            # Todo: Implement
            ret_str += "TO BE IMPLEMENTED"

        return ret_str


class SliceCollection:
    """
    Creates a collection of slices by collecting all slice (SLC) information in a given .smv file.
    :param file_path: Path to the .smv file containing information about the slices.
    """

    def __init__(self, file_path: str):
        self._slices = list()

        logging.debug("scanning smv file for slcf: %s", file_path)

        with open(file_path, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                       access=mmap.ACCESS_READ) as smv_file:
            pos = smv_file.find(b'SLC')
            while pos > 0:
                smv_file.seek(pos)
                line = smv_file.readline()
                centered = False
                if line.find(b'SLCC') != -1:
                    centered = True

                logging.debug("slice centered: %s", centered)

                mesh_id = int(line.split(b'&')[0].split()[1].decode()) - 1

                index_range = line.split(b'&')[1].split()
                x1 = int(index_range[0])
                x2 = int(index_range[1])
                y1 = int(index_range[2])
                y2 = int(index_range[3])
                z1 = int(index_range[4])
                z2 = int(index_range[5])

                fn = smv_file.readline().decode().strip()
                q = smv_file.readline().decode().strip()
                l = smv_file.readline().decode().strip()
                u = smv_file.readline().decode().strip()

                self._slices.append(
                    Slice(q, l, u, fn, mesh_id, ((x1, x2), (y1, y2), (z1, z2)), centered,
                          root=os.path.dirname(file_path)))

                logging.debug("slice info: %i %s", mesh_id, [[x1, x2], [y1, y2], [z1, z2]])

                pos = smv_file.find(b'SLC', pos + 1)

    def combine_slices(self, slices: Union[List[Slice], Tuple[Slice]]):
        """

        :param slice_indices: Indices of slices to combine.
        :return:
        """
        base_extent = slices[0].slice_mesh.extent
        min_x1 = base_extent[0]
        max_x1 = base_extent[1]
        min_x2 = base_extent[2]
        max_x2 = base_extent[3]

        for slc in slices:
            min_x1 = min(slc.slice_mesh.extent[0], min_x1)
            max_x1 = max(slc.slice_mesh.extent[1], max_x1)
            min_x2 = min(slc.slice_mesh.extent[2], min_x2)
            max_x2 = max(slc.slice_mesh.extent[3], max_x2)

        dx1 = (base_extent[1] - base_extent[0]) / (slices[0].slice_mesh.n[0] - 1)
        dx2 = (base_extent[3] - base_extent[2]) / (slices[0].slice_mesh.n[1] - 1)

        n1 = int((max_x1 - min_x1) / dx1)
        n2 = int((max_x2 - min_x2) / dx2)

        if not slices[0].centered:
            n1 += 1
            n2 += 1

        x1 = np.linspace(min_x1, max_x1, n1)
        x2 = np.linspace(min_x2, max_x2, n2)

        mesh = np.meshgrid(x2, x1)
        data = np.ones((slices[0].times.size, n2, n1))
        mask = np.zeros((n2, n1))
        mask[:] = True

        for slc in slices:

            off1 = int((slc.slice_mesh.extent[0] - min_x1) / dx1)
            off2 = int((slc.slice_mesh.extent[2] - min_x2) / dx2)

            cn1 = slc.slice_mesh.n[0]
            cn2 = slc.slice_mesh.n[1]

            if slc.centered:
                cn1 -= 1
                cn2 -= 1

            # TODO: fix index order?
            data[:, off2:off2 + cn2, off1:off1 + cn1] = slc.sd
            mask[off2:off2 + cn2, off1:off1 + cn1] = False

        return mesh, [min_x1, max_x1, min_x2, max_x2], data, mask

    def find_slice_by_label(self, label: str) -> Slice:
        """

        :param label:
        :return:
        """
        for slc in self:
            if slc.label == label:
                return slc
        raise ValueError("no slice matching label: " + label)

    def find_slices(self, quantity: str, normal_direction: Literal['x', 'y', 'z', 0, 1, 2],
                    offset: float):
        """

        :param quantity:
        :param normal_direction:
        :param offset:
        :return:
        """
        slices = list()

        for slc in self:
            if slc.quantity == quantity and slc.normal_direction == normal_direction and np.abs(
                    slc.slice_mesh.norm_offset - offset) < 0.01:
                slices.append(slc)

        return slices

    def __str__(self, *args, **kwargs):
        lines = list()
        for s in range(len(self._slices)):
            lines.append("index %d-03: %s" % (s, self._slices[s]))
        return '\n'.join(lines)

    def __getitem__(self, item: int):
        return self._slices[item]

    def __iter__(self) -> Iterator[Slice]:
        return iter(self._slices)
