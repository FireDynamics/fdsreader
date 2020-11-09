import mmap
import os
from typing import List, Dict, Literal

import numpy as np
import logging

from utils import Mesh, Extent, Surface
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

            smv_file.seek(smv_file.find(b'HRRPUVCUT'))
            smv_file.readline()
            smv_file.readline()
            self.hrrpuv_cutoff = float(smv_file.readline().decode().strip())

            smv_file.seek(smv_file.find(b'TOFFSET'))
            smv_file.readline()
            self.default_texture_origin = tuple(smv_file.readline().decode().strip().split())

            self.out_file_path = os.path.join(self.root_path, self.chid + ".out")

            self.surfaces = self._load_surfaces(smv_file)
            self.meshes = self._load_meshes(smv_file)

    def _load_meshes(self, smv_file: mmap.mmap) -> Dict[str, Mesh]:
        meshes = dict()

        def read_dimension(dim: Literal[b'X', b'Y', b'Z'], n: int, startpos: int):
            pos = smv_file.find(b'TRN' + dim, startpos)
            smv_file.seek(pos)
            smv_file.readline()
            noc = int(smv_file.readline().decode().strip())
            for _ in range(noc):
                smv_file.readline()
            coordinates = np.empty(n, dtype=np.float32)
            for i in range(n):
                coordinates[i] = smv_file.readline().split()[1]
            return coordinates

        pos = smv_file.find(b'GRID')
        while pos > 0:
            smv_file.seek(pos)

            mesh_id = smv_file.readline().split()[1].decode()
            logging.debug("found MESH with id: %s", mesh_id)

            grid_numbers = smv_file.readline().split()
            nx = int(grid_numbers[0]) + 1
            ny = int(grid_numbers[1]) + 1
            nz = int(grid_numbers[2]) + 1
            # # correct for 2D cases
            # if nx == 2: nx = 1
            # if ny == 2: ny = 1
            # if nz == 2: nz = 1

            logging.debug("number of cells: %i x %i x %i", nx, ny, nz)

            meshes[mesh_id] = Mesh(read_dimension(b'X', nx, pos + 1), read_dimension(b'Y', ny, pos + 1),
                                   read_dimension(b'Z', nz, pos + 1), mesh_id, smv_file, pos, self.surfaces,
                                   self.default_texture_origin)

            pos = smv_file.find(b'GRID', pos + 1)

        return meshes

    def _load_surfaces(self, smv_file: mmap.mmap) -> List[Surface]:
        surfaces = list()
        pos = smv_file.find(b'SURFACE')
        while pos > 0:
            smv_file.seek(pos)
            smv_file.readline()

            surface_id = smv_file.readline().decode().strip()

            line = smv_file.readline().decode().strip().split()
            tmpm, material_emissivity = float(line[0]), float(line[1])

            line = smv_file.readline().decode().strip().split()

            surface_type, texture_width, texture_height, rgb, transparency = int(line[0]), float(line[1]), float(
                line[2]), (float(line[3]), float(line[4]), float(line[5])), float(line[6])

            texture_map = smv_file.readline().decode().strip()
            texture_map = None if texture_map == "null" else os.path.join(self.root_path, texture_map)

            surfaces.append(
                Surface(surface_id, tmpm, material_emissivity, surface_type, texture_width, texture_height, texture_map,
                        rgb, transparency))

            pos = smv_file.find(b'SURFACE')
        return surfaces

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
        # Only load isosurfaces once initially and then reuse the loaded information
        if not hasattr(self, "_isosurfaces"):
            self._isosurfaces = list()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'ISOG')
                while pos > 0:
                    smv_file.seek(pos - 1)
                    double_quantity = smv_file.read(1) == b'T'
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
