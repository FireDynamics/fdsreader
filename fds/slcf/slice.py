"""
Defines classes and methods to load slice-files and work with sliced data.
"""

import os
import mmap
import logging
from typing import List, Tuple, Union, Iterator
from typing_extensions import Literal
import numpy as np

from fds.utils import FDS_DATA_TYPE_INTEGER, FDS_DATA_TYPE_FLOAT, FDS_DATA_TYPE_CHAR, \
    FDS_FORTRAN_BACKWARD
from fds.mesh import MeshCollection


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
            name) + ") has to be one of ['header', 'index', 'time', 'data']!")
    if FDS_FORTRAN_BACKWARD:
        type_slice_str += ", " + FDS_DATA_TYPE_INTEGER
    return np.dtype(type_slice_str)


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
        """
        Numpy array with all times in this slice file
        """
        if not hasattr(self, "_all_times"):
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

        # Numpy array with all times for which data has been loaded
        self.times = np.zeros(n_slices)

        # Numpy 2D-array with time on the first and all data values on the second axis
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

        if hasattr(self, "times"):
            ret_str += ", times: {}, [{}, {}]".format(self.times.size, self.times[0],
                                                      self.times[-1])

        if hasattr(self, "data_raw"):
            # Todo: Implement
            ret_str += "TO BE IMPLEMENTED"

        return ret_str

    def equals(self, quantity: str, normal_direction: Literal[0, 1, 2], normal_offset: float):
        return quantity == self.quantity and normal_direction == self.normal_direction \
               and np.abs(normal_offset - self.slice_mesh.normal_offset) < 0.01

    def __eq__(self, other):
        return other.quantity == self.quantity and other.normal_direction == self.normal_direction \
               and np.abs(other.slice_mesh.normal_offset - self.slice_mesh.normal_offset) < 0.01


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

                # Read in index ranges for x, y and z
                index_range = line.split(b'&')[1].split()
                x_start = int(index_range[0])
                x_end = int(index_range[1])
                y_start = int(index_range[2])
                y_end = int(index_range[3])
                z_start = int(index_range[4])
                z_end = int(index_range[5])

                filename = smv_file.readline().decode().strip()
                quantity = smv_file.readline().decode().strip()
                label = smv_file.readline().decode().strip()
                units = smv_file.readline().decode().strip()

                self._slices.append(Slice(quantity, label, units, filename, mesh_id,
                                          ((x_start, x_end), (y_start, y_end), (z_start, z_end)),
                                          centered, root=os.path.dirname(file_path)))

                logging.debug("slice info: %i %s", mesh_id,
                              [[x_start, x_end], [y_start, y_end], [z_start, z_end]])

                pos = smv_file.find(b'SLC', pos + 1)

    @staticmethod
    def combine_slices(slices: Union[List[Slice], Tuple[Slice]]):
        """

        :param slices: Slices to combine.
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

    def find_slices(self, quantity: str, normal_direction: Literal[0, 1, 2], normal_offset: float):
        """

        :param quantity:
        :param normal_direction: Direction of the normal (parallel to one axis).
        :param normal_offset:
        :return:
        """
        slices = list()

        for slc in self:
            if slc.equals(quantity, normal_direction, normal_offset):
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


class SliceMesh:
    """
    slice of a mesh in a given direction.
    :ivar normal_direction: Direction of the normal, defines which axes the slice mesh has.
    :ivar normal_offset: Offset value in normal direction.
    :ivar directions: Labels for axes directions.
    :ivar axes: Coordinate values for the two axes the slice mesh expands on.
    :ivar mesh: Numpy meshgrid along both axes.
    :ivar n: Number of elements for both of the 2 dimensions.
    :ivar n_size: Total number of blocks in this mesh slice.
    """

    def __init__(self, mesh, normal_direction: Literal[0, 1, 2], normal_offset_index: float):
        """
        :param mesh: The mesh to extract a slice from.
        :param normal_direction: Direction of the normal (parallel to one axis).
        :param normal_offset_index: Offset index in normal direction.
        """
        # Todo: Centered/Nicht-centered
        if normal_direction in ('x', 0):
            self.normal_direction = 0
            self.directions = ['y', 'z']
            self.normal_offset = mesh.coordinates[0][normal_offset_index]
            self.axes = [mesh.coordinates[1], mesh.coordinates[2]]
        if normal_direction in ('y', 1):
            self.normal_direction = 1
            self.directions = ['x', 'z']
            self.normal_offset = mesh.coordinates[1][normal_offset_index]
            self.axes = [mesh.coordinates[0], mesh.coordinates[2]]
        if normal_direction in ('z', 2):
            self.normal_direction = 2
            self.directions = ['x', 'y']
            self.normal_offset = mesh.coordinates[2][normal_offset_index]
            self.axes = [mesh.coordinates[0], mesh.coordinates[1]]
        else:
            raise ValueError("Parameter normal_direction  (" + str(
                normal_direction) + ") has to be one of [x,y,z,0,1,2]!")

        self.mesh = np.meshgrid(self.axes[0], self.axes[1])

        self.extent = (
            np.min(self.axes[0]), np.max(self.axes[0]), np.min(self.axes[1]), np.max(self.axes[1]))

        self.n = [self.axes[0].size, self.axes[1].size]
        self.n_size = self.n[0] * self.n[1]
