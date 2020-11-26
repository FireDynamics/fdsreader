import mmap
import os
from typing import List, Literal

import numpy as np
import logging

from fdsreader.bndf import Boundary
from fdsreader.plot3d import Plot3D
from fdsreader.utils import Mesh, Extent, Surface, Quantity
from fdsreader.slcf import Slice
from fdsreader.isof import Isosurface


class Simulation:
    """
    Master class managing all data for a given simulation.
    :ivar smv_file_path: Path to the smv-file for the simulation.
    :ivar root_path: Path to the root directory of the simulation.
    :ivar fds_version: Version of FDS the simulation was performed with.
    :ivar chid: Name (ID) of the simulation.
    :ivar hrrpuv_cutoff: The hrrpuv_cutoff value.
    :ivar default_texture_origin: The default origin used for textures with no explicit origin.
    :ivar out_file_path: Path to the .out file of the simulation.
    :ivar surfaces: List containing all surfaces defined in this simulation.
    :ivar meshes: List containing all meshes (grids) defined in this simulation.
    """

    def __init__(self, smv_file_path):
        self.smv_file_path = smv_file_path
        self.root_path = os.path.dirname(self.smv_file_path)

        with open(smv_file_path, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                           access=mmap.ACCESS_READ) as smv_file:
            smv_file.seek(smv_file.find(b'FDSVERSION', 0))
            smv_file.readline()
            self.fds_version = smv_file.readline().decode().strip()

            smv_file.seek(smv_file.find(b'CHID', 0))
            smv_file.readline()
            self.chid = smv_file.readline().decode().strip()

            smv_file.seek(smv_file.find(b'HRRPUVCUT', 0))
            smv_file.readline()
            smv_file.readline()
            self.hrrpuv_cutoff = float(smv_file.readline().decode().strip())

            smv_file.seek(smv_file.find(b'TOFFSET', 0))
            smv_file.readline()
            self.default_texture_origin = tuple(smv_file.readline().decode().strip().split())

            self.out_file_path = os.path.join(self.root_path, self.chid + ".out")

            self.surfaces = self._load_surfaces(smv_file)
            self.meshes = self._load_meshes(smv_file)

    def _load_meshes(self, smv_file: mmap.mmap) -> List[Mesh]:
        """
        Method to load the mesh information from the smv file.
        """
        meshes: List[Mesh] = list()

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

        pos = smv_file.find(b'GRID', 0)
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

            meshes.append(Mesh(read_dimension(b'X', nx, pos + 1), read_dimension(b'Y', ny, pos + 1),
                               read_dimension(b'Z', nz, pos + 1), mesh_id, len(meshes) - 1,
                               smv_file, pos, self.surfaces,
                               self.default_texture_origin))

            pos = smv_file.find(b'GRID', pos + 1)

        return meshes

    def _load_surfaces(self, smv_file: mmap.mmap) -> List[Surface]:
        """
        Method to load the surface information from the smv file.
        """
        surfaces = list()
        pos = smv_file.find(b'SURFACE', 0)
        while pos > 0:
            smv_file.seek(pos)
            smv_file.readline()

            surface_id = smv_file.readline().decode().strip()

            line = smv_file.readline().decode().strip().split()
            tmpm, material_emissivity = float(line[0]), float(line[1])

            line = smv_file.readline().decode().strip().split()

            surface_type, texture_width, texture_height, rgb, transparency = int(line[0]), float(
                line[1]), float(
                line[2]), (float(line[3]), float(line[4]), float(line[5])), float(line[6])

            texture_map = smv_file.readline().decode().strip()
            texture_map = None if texture_map == "null" else os.path.join(self.root_path,
                                                                          texture_map)

            surfaces.append(
                Surface(surface_id, tmpm, material_emissivity, surface_type, texture_width,
                        texture_height, texture_map,
                        rgb, transparency))

            pos = smv_file.find(b'SURFACE')
        return surfaces

    @property
    def slices(self) -> List[Slice]:
        """
        Lazy loads all slices for the simulation.
        :returns: All slices.
        """
        # Only load slices once initially and then reuse the loaded information
        if not hasattr(self, "_slices"):
            slices = dict()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'SLC', 0)
                while pos > 0:
                    smv_file.seek(pos)
                    line = smv_file.readline()
                    if line.find(b'SLCC') != -1:
                        cell_centered = True
                    else:
                        cell_centered = False

                    slice_id = int(line.split(b'!')[1].strip().split()[0].decode())

                    mesh_index = int(line.split(b'&')[0].strip().split()[1].decode()) - 1

                    # Read in index ranges for x, y and z
                    index_ranges = [int(i.strip()) for i in
                                    line.split(b'&')[1].split(b'!')[0].strip().split()]

                    filename = smv_file.readline().decode().strip()
                    quantity = smv_file.readline().decode().strip()
                    label = smv_file.readline().decode().strip()
                    unit = smv_file.readline().decode().strip()

                    if slice_id not in slices:
                        slices[slice_id] = Slice(self.root_path, cell_centered)
                    slices[slice_id]._add_subslice(filename, quantity, label, unit,
                                                   Extent(*index_ranges), self.meshes[mesh_index])

                    pos = smv_file.find(b'SLC', pos + 1)
            if len(slices) > 0:
                self._slices = list(slices.values())
            else:
                raise IOError("This simulation did not output any slices.")
        return self._slices

    @property
    def boundaries(self) -> List[Boundary]:
        """
        Lazy loads all boundary data for the simulation.
        :returns: All boundary data.
        """
        if not hasattr(self, "_boundaries"):
            boundaries = dict()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'BND', 0)
                while pos > 0:
                    smv_file.seek(pos)
                    line = smv_file.readline().decode().strip().split()
                    if line[0] == 'BNDC':
                        cell_centered = True
                    else:
                        cell_centered = False
                    mesh_index = int(line[1]) - 1

                    filename = smv_file.readline().decode().strip()
                    quantity = smv_file.readline().decode().strip()
                    label = smv_file.readline().decode().strip()
                    unit = smv_file.readline().decode().strip()

                    bid = int(filename.split('_')[-1][:-3])

                    if bid not in boundaries:
                        boundaries[bid] = Boundary(bid, self.root_path, cell_centered, quantity,
                                                   label, unit)
                    boundaries[bid]._add_subboundary(filename, self.meshes[mesh_index])

                    pos = smv_file.find(b'BND', pos + 1)
            if len(boundaries) > 0:
                self._boundaries = list(boundaries.values())
            else:
                raise IOError("This simulation did not output any plot3d data.")
        return self._boundaries

    @property
    def data_3d(self) -> List[Plot3D]:
        """
        Lazy loads all plot3d data for the simulation.
        :returns: All plot3d data.
        """
        # Todo: Also read SMOKG3D data?
        # Only load plot3d data once initially and then reuse the loaded information
        if not hasattr(self, "_3d_data"):
            plot3ds = dict()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'PL3D', 0)
                while pos > 0:
                    smv_file.seek(pos)
                    line = smv_file.readline().decode().strip().split()

                    time = float(line[1])

                    mesh_index = int(line[2]) - 1

                    filename = smv_file.readline().decode().strip()
                    quantities = list()
                    for _ in range(5):
                        quantity = smv_file.readline().decode().strip()
                        label = smv_file.readline().decode().strip()
                        unit = smv_file.readline().decode().strip()
                        quantities.append(Quantity(quantity, label, unit))

                    if time not in plot3ds:
                        plot3ds[time] = Plot3D(self.root_path, time, quantities)
                    plot3ds[time]._add_subplot(filename, self.meshes[mesh_index])

                    pos = smv_file.find(b'PL3D', pos + 1)
            if len(plot3ds) > 0:
                self._3d_data = list(plot3ds.values())
            else:
                raise IOError("This simulation did not output any plot3d data.")
        return self._3d_data

    @property
    def isosurfaces(self):
        """
        Lazy loads all isosurfaces for the simulation.
        :returns: All isof data.
        """
        # Todo: Check for multimesh
        # Only load isosurfaces once initially and then reuse the loaded information
        if not hasattr(self, "_isosurfaces"):
            self._isosurfaces = list()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'ISOG', 0)
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
            if len(self._isosurfaces) == 0:
                raise IOError("This simulation did not output any isosurfaces.")
        return self._isosurfaces
