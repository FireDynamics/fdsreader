import mmap
import os

import numpy as np
import logging

from utils import FDS_DATA_TYPE_FLOAT, Mesh, Extent
from slcf import Slice
from isof import Isosurface


class Simulation:
    def __init__(self, smv_file_path):
        self.smv_file_path = smv_file_path
        self.root_path = os.path.dirname(self.smv_file_path)

        with open(smv_file_path, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                           access=mmap.ACCESS_READ) as smv_file:
            smv_file.seek(smv_file.find(b'FDSVERSION'))
            smv_file.readline()
            self.fds_version = smv_file.readline().decode().strip()

            smv_file.seek(smv_file.find(b'CHID'))
            smv_file.readline()
            self.chid = smv_file.readline().decode().strip()

            self.out_file_path = os.path.join(self.root_path, self.chid + ".out")

            self.meshes = self._load_meshes(smv_file)
            self.obstacles = self._load_obstacles(smv_file)

    def _load_meshes(self, smv_file):
        meshes = list()

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

            meshes.append(Mesh(x_coordinates, y_coordinates, z_coordinates, label))

            pos = smv_file.find(b'GRID', pos + 1)

        return meshes

    def _load_obstacles(self, smv_file):
        obstacles = dict()
        pos = smv_file.find(b'OBST')
        while pos > 0:
            smv_file.seek(pos)
            smv_file.readline()
            n = int(smv_file.readline().decode().strip())
            for i in range(n):
                obst = smv_file.readline().decode().strip().split()
                obst_id = int(obst[6])
                if obst_id not in obstacles:
                    obstacles[obst_id] = (
                        (int(obst[0]), int(obst[1])), (int(obst[2]), int(obst[3])),
                        (int(obst[4]), int(obst[5]))
                    )
            pos = smv_file.find(b'OBST')

        return list(obstacles.values())

    @property
    def slices(self):
        """
        Lazy loads all slices for the simulation.
        :return: All slices.
        """
        if not hasattr(self, "_slices"):
            slices = dict()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'SLC')
                while pos > 0:
                    smv_file.seek(pos)
                    line = smv_file.readline()
                    if line.find(b'SLCC') != -1:
                        centered = True
                    else:
                        centered = False

                    slice_id = int(line.split(b'!')[1].strip().split()[0].decode())

                    mesh_id = int(line.split(b'&')[0].strip().split()[1].decode()) - 1

                    # Read in index ranges for x, y and z
                    index_ranges = [int(i.strip()) for i in
                                    line.split(b'&')[1].split(b'!')[0].strip().split()]

                    filename = smv_file.readline().decode().strip()
                    quantity = smv_file.readline().decode().strip()
                    label = smv_file.readline().decode().strip()
                    unit = smv_file.readline().decode().strip()

                    if slice_id not in slices:
                        slices[slice_id] = Slice(self.root_path, centered)
                    slices[slice_id]._add_subslice(filename, quantity, label, unit,
                                                   Extent(*index_ranges), mesh_id)

                    pos = smv_file.find(b'SLC', pos + 1)
            self._slices = list(slices.values())
        return self._slices

    @property
    def data_3d(self):
        """
        Lazy loads all plot3d data for the simulation.
        :return: All plot3d data.
        """
        # Only load slices once initially and then reuse the loaded information
        if not hasattr(self, "_3d_data"):
            slices = dict()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'SLC')
                while pos > 0:
                    smv_file.seek(pos)
                    line = smv_file.readline()
                    if line.find(b'SLCC') != -1:
                        centered = True
                    else:
                        centered = False

                    slice_id = int(line.split(b'!')[1].strip().split()[0].decode())

                    mesh_id = int(line.split(b'&')[0].strip().split()[1].decode()) - 1

                    # Read in index ranges for x, y and z
                    index_ranges = [int(i.strip()) for i in
                                    line.split(b'&')[1].split(b'!')[0].strip().split()]

                    filename = smv_file.readline().decode().strip()
                    quantity = smv_file.readline().decode().strip()
                    label = smv_file.readline().decode().strip()
                    unit = smv_file.readline().decode().strip()

                    if slice_id not in slices:
                        slices[slice_id] = Slice(os.path.dirname(self.smv_file_path), centered)
                    slices[slice_id]._add_subslice(filename, quantity, label, unit,
                                                   Extent(*index_ranges), mesh_id)

                    pos = smv_file.find(b'SLC', pos + 1)
            if len(slices) > 0:
                self._slices = list(slices.values())
            else:
                raise IOError("This simulation did not output any slices.")
        return self._slices

    @property
    def isosurfaces(self):
        """
        Lazy loads all isosurfaces for the simulation.
        :return: All isof data.
        """
        # Dictionary mapping isosurface index to all its computed levels
        # iso_levels = dict()
        # # Read information about isosurfaces in .out-file
        # with open(self.out_file_path, 'r') as infile, \
        #         mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as out_file:
        #     pos = out_file.find(b'Isosurface File Information')
        #     out_file.seek(pos)
        #     for i in range(4): out_file.readline()
        #
        #     line = out_file.readline().decode().strip()
        #     while "Quantity:" in line:
        #         index, levels = line.split("Quantity:")
        #         levels = levels.split(":")[1].strip().split(" ")
        #         iso_levels[int(index)] = [float(level) for level in levels]
        #
        #         line = out_file.readline().decode().strip()

        # Only load isosurfaces once initially and then reuse the loaded information
        if not hasattr(self, "_isosurfaces"):
            self._isosurfaces = list()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'ISOG')
                while pos > 0:
                    smv_file.seek(pos - 1)
                    if smv_file.read(1).decode() == "T":
                        double_quantity = True
                    else:
                        double_quantity = False
                    smv_file.readline()

                    iso_filename = smv_file.readline().decode().strip()
                    if double_quantity:
                        viso_filename = smv_file.readline().decode().strip()
                    quantity = smv_file.readline().decode().strip()
                    label = smv_file.readline().decode().strip()
                    unit = smv_file.readline().decode().strip()
                    if double_quantity:
                        v_quantity = smv_file.readline().decode().strip()
                        v_label = smv_file.readline().decode().strip()
                        v_unit = smv_file.readline().decode().strip()

                    if double_quantity:
                        self._isosurfaces.append(
                            Isosurface(os.path.dirname(self.smv_file_path), double_quantity,
                                       iso_filename, quantity, label, unit,
                                       viso_filename=viso_filename, v_quantity=v_quantity,
                                       v_label=v_label, v_unit=v_unit))
                    else:
                        self._isosurfaces.append(
                            Isosurface(os.path.dirname(self.smv_file_path), double_quantity,
                                       iso_filename, quantity, label, unit))

                    pos = smv_file.find(b'ISOG', pos + 1)
        return self._isosurfaces
