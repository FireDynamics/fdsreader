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
from fds.utils.mesh import MeshCollection


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
    offset = 3 * get_slice_type('header').itemsize + get_slice_type('index').itemsize

    def __init__(self, quantity: str, label: str, units: str, filename: str, mesh_id: int,
                 extent: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                 centered: bool, root_path: str = '.'):
        """
        :param quantity: Quantity string as defined in .fds file.
        :param label: Label assigned by fds in .smv file.
        :param units: Units string assigned by fds in .smv file.
        :param filename: Name of the corresponding slice file (.sf).
        :param mesh_id: ID of the mesh containing this slice.
        :param extent: Tuple with three tuples containing minimum and maximum coordinate value on
                       the corresponding dimension.
        :param centered: Indicates whether centered positioning for data is used.
        :param root_path: Path containing the corresponding slice file (.sf).
        """
        self.root = root_path
        self.quantity = quantity
        self.label = label
        self.units = units
        self.filename = filename
        self.mesh_id = mesh_id
        self.extent = extent
        self.centered = centered

        self.read_size = (extent[0][1] - extent[0][0] + 1) * \
                         (extent[1][1] - extent[1][0] + 1) * \
                         (extent[2][1] - extent[2][0] + 1)

        self.stride = get_slice_type('time').itemsize + get_slice_type('data',
                                                                       self.read_size).itemsize

        infile = open(os.path.join(self.root, self.filename), 'r')
        infile.seek(0, 0)

        self.header = np.fromfile(infile, dtype=get_slice_type('header'), count=3)
        logging.debug("slice header: %s", self.header)
        self.index = np.fromfile(infile, dtype=get_slice_type('index'), count=1)[0]
        logging.debug("slice index: %i", str(self.index))

        logging.debug("scanning smv file for slcf: %s", file_path)

        with open(file_path, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                       access=mmap.ACCESS_READ) as smv_file:
            pos = smv_file.find(b'SLC')
            while pos > 0:
                smv_file.seek(pos)
                line = smv_file.readline()

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

    def __str__(self, *args, **kwargs):
        ret_str = "%s, %s, %s, mesh_id: %i, [%i, %i] x [%i, %i] x [%i, %i]" % (
            self.label, self.quantity, self.units, self.mesh_id, self.extent[0][0],
            self.extent[0][1], self.extent[1][0], self.extent[1][1],
            self.extent[2][0], self.extent[2][1])

        if hasattr(self, "times"):
            ret_str += ", times: {}, [{}, {}]".format(self.times.size, self.times[0],
                                                      self.times[-1])

        if hasattr(self, "data_raw"):
            ret_str += ", " + str(self.data_raw[0, :5].extend(self.data_raw[-1, -5:]))

        return ret_str

    def __eq__(self, other):
        return True


