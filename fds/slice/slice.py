import os
import sys
import re
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import typing
import matplotlib.animation as animation

import logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

# as the binary representation of raw data is compiler dependent, this
# information must be provided by the user
fds_data_type_integer = "i4" # i4 -> 32 bit integer (native endiannes, probably little-endian)
fds_data_type_float = "f4" # f4 -> 32 bit floating point (native endiannes, probably little-endian)
fds_data_type_char = "a"  # a -> 8 bit character
fds_fortran_backward = True # sets weather the blocks are ended with the size of the block
def getSliceType(name: str, n_size=0):
    type_slice_str = fds_data_type_integer + ", "
    if name == 'header':
        type_slice_str += "30" + fds_data_type_char
    elif name == 'index':
        type_slice_str += "6" + fds_data_type_integer
    elif name == 'time':
        type_slice_str += fds_data_type_float
    elif name == 'data':
        type_slice_str += "({1}){2}".format(n_size, fds_data_type_float)
    else:
        return None
    if fds_fortran_backward:
        type_slice_data_str += ", " + fds_data_type_integer
    return np.dtype(type_slice_data_str)

class SliceMesh:
    def __init__(self, x1, x2, norm_dir:str, norm_offset:float):

        if norm_dir == 'x' or norm_dir == 0:
            self.directions = [1,2]
        elif norm_dir == 'y' or norm_dir == 1:
            self.directions = [0,2]
        elif norm_dir == 'z' or norm_dir == 2:
            self.directions = [0,1]

        self.norm_direction = norm_dir
        self.norm_offset = norm_offset

        self.axes = [x1, x2]
        self.mesh = np.meshgrid(x1, x2)

        self.extent = (np.min(x1), np.max(x1), np.min(x2), np.max(x2))

        self.n = [x1.size, x2.size]
        self.nSize = self.n[0] * self.n[1]


class SliceMeshCollection:
    def __init__(self):
        self.meshes = []

    def read(self, filename):
        pass

class Mesh:
    def __init__(self, x1, x2, x3, label):
        self.axes = [x1, x2, x3]
        self.ranges = [[x1[0], x1[-1]] , [x2[0], x2[-1]] , [x3[0], x3[-1]]]
        self.mesh = np.meshgrid(self.axes)

        self.n = [x1.size, x2.size, x3.size]
        self.nSize = self.n[0] * self.n[1] * self.n[2]

        self.label = label

    def infoString(self):
        return "{}, {} x {} x {}, [{}, {}] x [{}, {}] x [{}, {}]".format(self.label,
            self.n[0], self.n[1], self.n[2],
            self.ranges[0][0], self.ranges[0][1], self.ranges[1][0], self.ranges[1][1],
            self.ranges[2][0], self.ranges[2][1])

    def extractSliceMesh(self, norm_dir:str, norm_index:int) -> SliceMesh:
        if norm_dir == 'x' or norm_dir == 0:
            offset = self.axes[0][norm_index]
            return SliceMesh(self.axes[1], self.axes[2], norm_dir, offset)
        if norm_dir == 'y' or norm_dir == 1:
            offset = self.axes[1][norm_index]
            return SliceMesh(self.axes[0], self.axes[2], norm_dir, offset)
        if norm_dir == 'z' or norm_dir == 2:
            offset = self.axes[2][norm_index]
            return SliceMesh(self.axes[0], self.axes[1], norm_dir, offset)

        return None

class MeshCollection:
    def __init__(self):
        self.meshes = []

    def print(self):
        for m in self.meshes:
            print(m.infoString())

def readMeshes(filename):
    import mmap

    mc = MeshCollection()

    logging.debug("scanning smv file for meshes: {}".format(filename))

    infile = open(filename, 'r')
    with mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as s:

        cpos = s.find(b'GRID')

        while cpos > 0:
            s.seek(cpos)

            label = s.readline().split()[1].decode()
            logging.debug("found MESH with label: {}".format(label))

            grid_numbers = s.readline().split()
            nx = int(grid_numbers[0]) + 1
            # # correct for 2D cases
            # if nx == 2: nx = 1
            ny = int(grid_numbers[1]) + 1
            # if ny == 2: ny = 1
            nz = int(grid_numbers[2]) + 1
            # if nz == 2: nz = 1

            logging.debug("number of cells: {} x {} x {}".format(nx, ny, nz))

            cpos = s.find(b'TRNX', cpos + 1)
            s.seek(cpos)
            s.readline()
            s.readline()
            x_coor = np.zeros(nx)
            for ix in range(nx):
                x_coor[ix] = float(s.readline().split()[1])

            cpos = s.find(b'TRNY', cpos + 1)
            s.seek(cpos)
            s.readline()
            s.readline()
            y_coor = np.zeros(ny)
            for iy in range(ny):
                y_coor[iy] = float(s.readline().split()[1])

            cpos = s.find(b'TRNZ', cpos + 1)
            s.seek(cpos)
            s.readline()
            s.readline()
            z_coor = np.zeros(nz)
            for iz in range(nz):
                z_coor[iz] = float(s.readline().split()[1])

            mc.meshes.append(Mesh(x_coor, y_coor, z_coor, label))

            cpos = s.find(b'GRID', cpos + 1)

    infile.close()

    return mc




class SliceCollection:
    def __init__(self, mc=None):
        self.meshes = mc

        self.slices = []

    def setMeshes(self, mc):
        self.meshes = mc

    def print(self):
        for s in range(len(self.slices)):
            print("index {:03d}: {:s}".format(s, self.slices[s].infoString()))

    def __getitem__(self, item):
        return self.slices[item]

class Slice:
    def __init__(self, quantity, label, units, filename, mesh_id, index_ranges, centered):
        self.quantity = quantity
        self.label = label
        self.units = units
        self.filename = filename
        self.mesh_id = mesh_id
        self.index_ranges = index_ranges
        self.centered = centered

        self.readSize = (index_ranges[0][1] - index_ranges[0][0] + 1) * \
                        (index_ranges[1][1] - index_ranges[1][0] + 1) * \
                        (index_ranges[2][1] - index_ranges[2][0] + 1)

        if self.centered:
            self.nSize = 1
            if (index_ranges[0][1] - index_ranges[0][0]) > 0:
                self.nSize *= index_ranges[0][1] - index_ranges[0][0]
            if (index_ranges[1][1] - index_ranges[1][0]) > 0:
                self.nSize *= index_ranges[1][1] - index_ranges[1][0]
            if (index_ranges[2][1] - index_ranges[2][0]) > 0:
                self.nSize *= index_ranges[2][1] - index_ranges[2][0]
        else:
            self.nSize = self.readSize

        if index_ranges[0][0] == index_ranges[0][1]:
            self.norm_direction = 0
            self.norm_offset = index_ranges[0][0]
        if index_ranges[1][0] == index_ranges[1][1]:
            self.norm_direction = 1
            self.norm_offset = index_ranges[1][0]
        if index_ranges[2][0] == index_ranges[2][1]:
            self.norm_direction = 2
            self.norm_offset = index_ranges[2][0]

        self.data_raw = None
        self.times = None
        self.all_times = None

    def readAllTimes(self, root='.'):
        slcf = open(os.path.join(root, self.filename), 'r')

        offset = 3 * getSliceType('header').itemsize + getSliceType('index').itemsize

        type_time = getSliceType('time')
        type_data = getSliceType('data', self.readSize)

        stride = type_time.itemsize + type_data.itemsize

        count = 0
        times = []
        while True:
            slcf.seek(offset + count * stride)
            slice_time_raw = np.fromfile(slcf, dtype=type_time, count=1)

            if len(slice_time_raw) == 0:
                break

            slice_time = slice_time_raw[0][1]
            # print("found time: ", slice_time)

            times.append(slice_time)
            count += 1

        self.all_times = np.array(times)

    def readTimeSelection(self, root='.', dt=None, average_dt=None):

        if self.all_times is None:
            logging.error("read in slice times first")
            return

        if dt is None:
            logging.error("provide a dt for slice selection")
            return

        infile = open(os.path.join(root, self.filename), 'r')
        infile.seek(0, 0)

        slice_header = np.fromfile(infile, dtype=getSliceType('header'),
                                   count=3)
        print("slice header: ", slice_header)
        slice_index = np.fromfile(infile, dtype=getSliceType('index'),
                                  count=1)[0]
        print("slice index: ", slice_index)

        type_time = getSliceType('time')
        type_data = getSliceType('data', self.readSize)
        stride = type_time.itemsize + type_data.itemsize

        offset = 3 * getSliceType('header').itemsize + getSliceType('index').itemsize

        nSlices = int(self.all_times[-1] / dt)

        self.times = np.zeros(nSlices)

        self.data_raw = np.zeros((nSlices, self.readSize))

        for t in range(nSlices):

            central_time_index = np.where(self.all_times > t * dt)[0][0]

            print("central index, time: {}, {}".format(central_time_index,
                                                       self.all_times[central_time_index]))

            self.times[t] = self.all_times[central_time_index]
            time_indices = central_time_index

            if average_dt:
                time_indices = np.where((self.all_times > t * dt - average_dt / 2.0) & (self.all_times < t * dt + average_dt / 2.0))[0]

            print("using time indices: {}".format(time_indices))

            for st in time_indices:
                infile.seek(offset + st * stride)
                slice_time = np.fromfile(infile, dtype=type_time, count=1)
                slice_data = np.fromfile(infile, dtype=type_data, count=1)
                self.data_raw[t, :] += slice_data[0][1]

            self.data_raw[t, :] /= len(time_indices)

    def readData(self, root='.'):

        if self.all_times is None:
            logging.error("read in slice times first")
            return

        infile = open(os.path.join(root, self.filename), 'r')
        infile.seek(0, 0)

        slice_header = np.fromfile(infile, dtype=getSliceType('header'), count=3)
        print("slice header: ", slice_header)
        slice_index = np.fromfile(infile, dtype=getSliceType('index'), count=1)[0]
        print("slice index: ", slice_index)

        type_time = getSliceType('time')
        type_data = getSliceType('data', self.readSize)
        stride = type_time.itemsize + type_data.itemsize

        offset = 3 * getSliceType('header').itemsize + getSliceType('index').itemsize

        self.data_raw = np.zeros((self.all_times.size, self.readSize))
        # print("time size: ", self.all_times.size)
        # print("self size: ", self.nSize)
        # print("read size: ", self.readSize)

        for t in range(self.all_times.size):
            infile.seek(offset + t * stride)
            slice_time = np.fromfile(infile, dtype=type_time, count=1)
            # print("slice time: ", slice_time)
            slice_data = np.fromfile(infile, dtype=type_data, count=1)
            # print(slice_data)
            # print("slice data: ", slice_data[0][0], slice_data[0][1][0:2],
            # slice_data[0][2])
            self.data_raw[t,:] = slice_data[0][1]


        infile.close()

        self.times = np.copy(self.all_times)

    def mapData(self, meshes: MeshCollection):

        cm = meshes.meshes[self.mesh_id]
        self.sm = cm.extractSliceMesh(self.norm_direction, self.norm_offset)

        # print(self.norm_direction, self.norm_offset)


        n1 = self.sm.n[0]
        n2 = self.sm.n[1]
        if self.centered:
            n1 -= 1
            n2 -= 1

        self.sd = np.zeros((self.times.size, n2, n1))
        for i in range(self.times.size):
            if self.centered:
                self.sd[i] = np.reshape(self.data_raw[i], (self.sm.n[1], self.sm.n[0]))[1:,1:]
            else:
                self.sd[i] = np.reshape(self.data_raw[i], (self.sm.n[1], self.sm.n[0]))


    def infoString(self):

        str_general = "{}, {}, {}, mesh_id: {}, [{}, {}] x [{}, {}] x [{}, {}]".format(self.label, self.quantity, self.units, self.mesh_id,
            self.index_ranges[0][0], self.index_ranges[0][1], self.index_ranges[1][0], self.index_ranges[1][1],
            self.index_ranges[2][0], self.index_ranges[2][1])

        str_times = ''
        if self.times is not None:
            str_times = ", times: {}, [{}, {}]".format(self.times.size, self.times[0], self.times[-1])

        str_data = ''
        if self.data_raw is not None:
            str_data = "TO BE IMPLEMENTED"

        return str_general + str_times + str_data

def findSlices(slices, quantity, normal_direction, offset):
    res = []

    for s in slices:
        if s.quantity == quantity and s.norm_direction == normal_direction and np.abs(s.sm.norm_offset - offset) < 0.01:
            res.append(s)

    return res


def combineSlices(slices):

    min_x1 = +1e18
    max_x1 = -1e18
    min_x2 = +1e18
    max_x2 = -1e18

    for s in slices:
        min_x1 = min(s.sm.extent[0], min_x1)
        max_x1 = max(s.sm.extent[1], max_x1)
        min_x2 = min(s.sm.extent[2], min_x2)
        max_x2 = max(s.sm.extent[3], max_x2)

    dx1 = (slices[0].sm.extent[1] - slices[0].sm.extent[0]) / (slices[0].sm.n[0] - 1)
    dx2 = (slices[0].sm.extent[3] - slices[0].sm.extent[2]) / (slices[0].sm.n[1] - 1)

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

    for s in slices:

        off1 = int((s.sm.extent[0] - min_x1) / dx1)
        off2 = int((s.sm.extent[2] - min_x2) / dx2)

        cn1 = s.sm.n[0]
        cn2 = s.sm.n[1]

        if s.centered:
            cn1 -= 1
            cn2 -= 1

        # TODO: fix index order?
        data[:,off2:off2 + cn2, off1:off1 + cn1] = s.sd
        mask[off2:off2 + cn2, off1:off1 + cn1] = False

    return mesh, [min_x1, max_x1, min_x2, max_x2], data, mask

def readSliceInfos(filename):
    import mmap

    sc = SliceCollection()

    logging.debug("scanning smv file for slices: {}".format(filename))

    infile = open(filename, 'r')
    with mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as s:
        cpos = s.find(b'SLC')
        while cpos > 0:
            s.seek(cpos)
            line = s.readline()
            centered = False
            if line.find(b'SLCC') >= 0: centered = True

            logging.debug("slice centered: {}".format(centered))

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

            sc.slices.append(Slice(q, l, u, fn, mesh_id, [[x1, x2], [y1, y2], [z1, z2]], centered))

            logging.debug("slice info: {} {}".format(mesh_id, [[x1, x2], [y1, y2], [z1, z2]]))

            cpos = s.find(b'SLC', cpos + 1)

    infile.close()
    return sc



def scanDirectory(directory: str):
    import glob
    import os.path

    list_fn_smv_abs = glob.glob(directory + "/*.smv")

    if len(list_fn_smv_abs) == 0:
        return None

    if len(list_fn_smv_abs) > 1:
        logging.warning("multiple smv files found, choosing an arbitrary file")

    return os.path.basename(list_fn_smv_abs[0])


# MAIN

#
#
# root_dir = sys.argv[1]
# smv_fn, slice_fns = scanDirectory(root_dir)
#
# meshes = readMeshes(smv_fn)
#
# meshes.print()
#
# slices = readSliceInfos(smv_fn)
#
# for s in slices.slices: s.readTimes()
#
# slices.print()
#
# s1 = slices.slices[0]
# s2 = slices.slices[6]
#
# s1.readData()
# s2.readData()
#
# s1.mapData(meshes)
# s2.mapData(meshes)
#
# print(s1.sm.extent, s2.sm.extent)
#
# fig, ax = plt.subplots()
#
# cmin = min(np.amin(s1.sd[-1]), np.amin(s1.sd[-1]))
# cmax = max(np.amax(s1.sd[-1]), np.amax(s1.sd[-1]))
#
# im1 = ax.imshow(s1.sd[-1], extent=s1.sm.extent, origin='lower', vmax=cmax, vmin=cmin, animated=True)
# im2 = ax.imshow(s2.sd[-1], extent=s2.sm.extent, origin='lower', vmax=cmax, vmin=cmin, animated=True)
#
# ax.autoscale()
#
# plt.xlabel(s1.sm.directions[0])
# plt.ylabel(s1.sm.directions[1])
# plt.title(s1.infoString())
#
# plt.colorbar(im1)
#
# it = 0
#
# def updatefig(*args):
#     global s1, s2, it, im1, im2
#     it += 1
#     if it >= s1.times.size: it = 0
#     im1.set_array(s1.sd[it])
#     im2.set_array(s2.sd[it])
#     return im1, im2
#
# ani1 = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
#
# plt.show()
#
# #
# # f = open(os.path.join(root_dir, smv_fn), 'r')
# # list_slice_summary, list_meshes, list_slcf = readGeometry(f)
# # f.close()
# #
# # for slc in list_slice_summary:
# #     dir = 'x='
# #     if slc['dir'] == 1: dir = 'y='
# #     if slc['dir'] == 2: dir = 'z='
# #     print("available slice: ", slc['q'], 'at ', dir, slc['coord'])
# #
# # times = readSliceTimes(os.path.join(root_dir, list_slcf[0]['fn']), list_slcf[0]['n_size'])
# #
# # # print(times)
# #
# # time = 44.9
# # time_step = np.where(times > time)[0][0]
# # print('read in time :', time, 'at step: ', time_step)
# # data = readSliceData(os.path.join(root_dir, slice_fns[0]), time_step)

