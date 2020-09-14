"""

"""

import os
import sys
import typing
import mmap
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import fds.utils

# as the binary representation of raw data is compiler dependent, this
# information must be provided by the user
FDS_DATA_TYPE_INTEGER = "i4"  # i4 -> 32 bit integer (native endiannes, probably little-endian)
FDS_DATA_TYPE_FLOAT = "f4"  # f4 -> 32 bit floating point (native endiannes, probably little-endian)
FDS_DATA_TYPE_CHAR = "a"  # a -> 8 bit character
FDS_FORTRAN_BACKWARD = True  # sets weather the blocks are ended with the size of the block


def get_slice_type(name: str, n_size=0):
    """

    :param name:
    :param n_size:
    :return:
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
        return None
    if FDS_FORTRAN_BACKWARD:
        type_slice_str += ", " + FDS_DATA_TYPE_INTEGER
    return np.dtype(type_slice_str)


class SliceMesh:
    """

    """

    def __init__(self, x1, x2, norm_dir: typing.Union[str, int], norm_offset: float):
        """

        :param x1:
        :param x2:
        :param norm_dir:
        :param norm_offset:
        """

        if norm_dir in ('x', 0):
            self.directions = [1, 2]
        elif norm_dir in ('y', 1):
            self.directions = [0, 2]
        elif norm_dir in ('z', 2):
            self.directions = [0, 1]

        self.norm_direction = norm_dir
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

    def extract_slice_mesh(self, norm_dir: typing.Union[str, int], norm_index: int) -> SliceMesh:
        """

        :param norm_dir:
        :param norm_index:
        :return:
        """
        if norm_dir in ('x', 0):
            offset = self.axes[0][norm_index]
            return SliceMesh(self.axes[1], self.axes[2], norm_dir, offset)
        if norm_dir in ('y', 1):
            offset = self.axes[1][norm_index]
            return SliceMesh(self.axes[0], self.axes[2], norm_dir, offset)
        if norm_dir in ('z', 2):
            offset = self.axes[2][norm_index]
            return SliceMesh(self.axes[0], self.axes[1], norm_dir, offset)
        raise ValueError(
            "Parameter norm_dir (" + str(norm_dir) + ") has to be one of (x,y,z,0,1,2)!")


class MeshCollection:
    """

    """

    def __init__(self, filename: str):
        """

        :param filename:
        """
        self.meshes = list()

        logging.debug("scanning smv file for meshes: %s", filename)

        with open(filename, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                      access=mmap.ACCESS_READ) as s:
            pos = s.find(b'GRID')

            while pos > 0:
                s.seek(pos)

                label = s.readline().split()[1].decode()
                logging.debug("found MESH with label: %s", label)

                grid_numbers = s.readline().split()
                nx = int(grid_numbers[0]) + 1
                # # correct for 2D cases
                # if nx == 2: nx = 1
                ny = int(grid_numbers[1]) + 1
                # if ny == 2: ny = 1
                nz = int(grid_numbers[2]) + 1
                # if nz == 2: nz = 1

                logging.debug("number of cells: %i x %i x %i", nx, ny, nz)

                pos = s.find(b'TRNX', pos + 1)
                s.seek(pos)
                s.readline()
                s.readline()
                x_coordinate = np.zeros(nx)
                for ix in range(nx):
                    x_coordinate[ix] = float(s.readline().split()[1])

                pos = s.find(b'TRNY', pos + 1)
                s.seek(pos)
                s.readline()
                s.readline()
                y_coordinate = np.zeros(ny)
                for iy in range(ny):
                    y_coordinate[iy] = float(s.readline().split()[1])

                pos = s.find(b'TRNZ', pos + 1)
                s.seek(pos)
                s.readline()
                s.readline()
                z_coordinate = np.zeros(nz)
                for iz in range(nz):
                    z_coordinate[iz] = float(s.readline().split()[1])

                self.meshes.append(Mesh(x_coordinate, y_coordinate, z_coordinate, label))

                pos = s.find(b'GRID', pos + 1)

    def __str__(self, *args, **kwargs):
        return '\n'.join((str(m) for m in self.meshes))

    def __getitem__(self, item: int):
        return self.meshes[item]


class Slice:
    """

    """
    offset = 3 * get_slice_type('header').itemsize + get_slice_type('index').itemsize

    def __init__(self, quantity: str, label: str, units: str, filename: str, mesh_id: int,
                 index_ranges: typing.Union[
                     typing.List[typing.Union[typing.List[int], typing.Tuple[int]]], typing.Tuple[
                         typing.Union[typing.List[int], typing.Tuple[int]]]],
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
            self.norm_direction = 0
            self.norm_offset = index_ranges[0][0]
        if index_ranges[1][0] == index_ranges[1][1]:
            self.norm_direction = 1
            self.norm_offset = index_ranges[1][0]
        if index_ranges[2][0] == index_ranges[2][1]:
            self.norm_direction = 2
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
        mesh = mesh_col.meshes[self.mesh_id]
        self.slice_mesh = mesh.extract_slice_mesh(self.norm_direction, self.norm_offset)

        n1 = self.slice_mesh.n[0]
        n2 = self.slice_mesh.n[1]
        if self.centered:
            n1 -= 1
            n2 -= 1

        self.sd = np.zeros((self.times.size, n2, n1))
        for i in range(self.times.size):
            if self.centered:
                self.sd[i] = np.reshape(self.data_raw[i],
                                        (self.slice_mesh.n[1], self.slice_mesh.n[0]))[1:, 1:]
            else:
                self.sd[i] = np.reshape(self.data_raw[i],
                                        (self.slice_mesh.n[1], self.slice_mesh.n[0]))

    def __str__(self, *args, **kwargs):
        str_general = "%s, %s, %s, mesh_id: %i, [%i, %i] x [%i, %i] x [%i, %i]" % (
            self.label, self.quantity, self.units, self.mesh_id, self.index_ranges[0][0],
            self.index_ranges[0][1], self.index_ranges[1][0], self.index_ranges[1][1],
            self.index_ranges[2][0], self.index_ranges[2][1])

        str_times = ''
        if self.times is not None:
            str_times = ", times: {}, [{}, {}]".format(self.times.size, self.times[0],
                                                       self.times[-1])

        str_data = ''
        if self.data_raw is not None:
            # Todo: Implement
            str_data = "TO BE IMPLEMENTED"

        return str_general + str_times + str_data


class SliceCollection:
    """

    """

    def __init__(self, filename: str, meshes: MeshCollection = None):
        """

        :param filename:
        :param meshes:
        """
        if meshes is None:
            self.meshes = MeshCollection(filename)
        else:
            self.meshes = meshes
        self.slices = list()

        logging.debug("scanning smv file for slcf: %s", filename)

        with open(filename, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                      access=mmap.ACCESS_READ) as s:
            pos = s.find(b'SLC')
            while pos > 0:
                s.seek(pos)
                line = s.readline()
                centered = False
                if line.find(b'SLCC') >= 0:
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

                fn = s.readline().decode().strip()
                q = s.readline().decode().strip()
                l = s.readline().decode().strip()
                u = s.readline().decode().strip()

                self.slices.append(
                    Slice(q, l, u, fn, mesh_id, [[x1, x2], [y1, y2], [z1, z2]], centered, root=os.path.dirname(filename)))

                logging.debug("slice info: %i %s", mesh_id, [[x1, x2], [y1, y2], [z1, z2]])

                pos = s.find(b'SLC', pos + 1)

    def combine_slices(self, slices: typing.Iterable[Slice]):
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
        logging.warning("no slice matching label: " + label)

    def find_slices(self, quantity: str, normal_direction: int, offset: float):
        """

        :param quantity:
        :param normal_direction:
        :param offset:
        :return:
        """
        slices = list()

        for slc in self:
            if slc.quantity == quantity and slc.norm_direction == normal_direction and np.abs(
                    slc.slice_mesh.norm_offset - offset) < 0.01:
                slices.append(slc)

        return slices

    def __str__(self, *args, **kwargs):
        lines = list()
        for s in range(len(self.slices)):
            lines.append("index %d-03: %s" % (s, self.slices[s]))
        return '\n'.join(lines)

    def __getitem__(self, item: int):
        return self.slices[item]


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    root_dir = sys.argv[1]
    smv_fn, _ = fds.utils.scan_directory_smv(root_dir)[0]

    meshes = MeshCollection(smv_fn)

    logging.debug(meshes)

    slc_col = SliceCollection(smv_fn)

    slice1 = slc_col.slices[0]
    slice2 = slc_col.slices[6]

    slice1.read_data()
    slice2.read_data()

    slice1.map_data(meshes)
    slice2.map_data(meshes)

    logging.debug(slice1.slice_mesh.extent + " " + slice2.slice_mesh.extent)

    fig, axis = plt.subplots()

    cmin = min(np.amin(slice1.sd[-1]), np.amin(slice2.sd[-1]))
    cmax = max(np.amax(slice1.sd[-1]), np.amax(slice2.sd[-1]))

    im1 = axis.imshow(slice1.sd[-1], extent=slice1.slice_mesh.extent, origin='lower', vmax=cmax,
                      vmin=cmin,
                      animated=True)
    im2 = axis.imshow(slice2.sd[-1], extent=slice2.slice_mesh.extent, origin='lower', vmax=cmax,
                      vmin=cmin,
                      animated=True)

    axis.autoscale()

    plt.xlabel(slice1.slice_mesh.directions[0])
    plt.ylabel(slice1.slice_mesh.directions[1])
    plt.title(slice1)

    plt.colorbar(im1)

    iteration = 0

    def updatefig(frame, *args):
        nonlocal iteration
        iteration += 1
        if iteration >= slice1.times.size:
            iteration = 0
        im1.set_array(slice1.sd[iteration])
        im2.set_array(slice2.sd[iteration])
        return im1, im2

    _ = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)

    plt.show()

    # f = open(os.path.join(root_dir, smv_fn), 'r')
    # list_slice_summary, list_meshes, list_slcf = readGeometry(f)
    # f.close()

    # for slc in list_slice_summary:
    #    direction = 'x='
    #    if slc['dir'] == 1: direction = 'y='
    #    if slc['dir'] == 2: direction = 'z='
    #    print("available slice: ", slc['q'], 'at ', direction, slc['coord'])

    # times = readSliceTimes(os.path.join(root_dir, list_slcf[0]['fn']),
    # list_slcf[0]['n_size'])

    ## print(times)

    # time = 44.9
    # time_step = np.where(times > time)[0][0]
    # print('read in time :', time, 'at step: ', time_step)
    # data = readSliceData(os.path.join(root_dir, slice_fns[0]), time_step)


if __name__ == "__main__":
    main()
