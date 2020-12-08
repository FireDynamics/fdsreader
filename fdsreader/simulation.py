import glob
import mmap
import os
from typing import List, Literal

import numpy as np
import logging
import pickle

from fdsreader.bndf import Boundary
from fdsreader.bndf.BoundaryCollection import BoundaryCollection
from fdsreader.isof.IsosurfaceCollection import IsosurfaceCollection
from fdsreader.plot3d import Plot3D
from fdsreader.plot3d.Plot3dCollection import Plot3DCollection
from fdsreader.slcf.SliceCollection import SliceCollection
from fdsreader.utils import Mesh, Extent, Surface, Quantity
from fdsreader.slcf import Slice
from fdsreader.isof import Isosurface
from fdsreader.utils.data import create_hash, get_smv_file


class Simulation:
    """Master class managing all data for a given simulation.

    :ivar smv_file_path: Path to the .smv file of the simulation.
    :ivar root_path: Path to the root directory of the simulation.
    :ivar fds_version: Version of FDS the simulation was performed with.
    :ivar chid: Name (ID) of the simulation.
    :ivar hrrpuv_cutoff: The hrrpuv_cutoff value.
    :ivar default_texture_origin: The default origin used for textures with no explicit origin.
    :ivar out_file_path: Path to the .out file of the simulation.
    :ivar surfaces: List containing all surfaces defined in this simulation.
    :ivar meshes: List containing all meshes (grids) defined in this simulation.
    """

    def __new__(cls, path: str):
        root_path = os.path.dirname(path)
        smv_file_path = get_smv_file(path)

        with open(smv_file_path, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                           access=mmap.ACCESS_READ) as smv_file:
            smv_file.seek(smv_file.find(b'CHID', 0))
            smv_file.readline()
            chid = smv_file.readline().decode().strip()

        pickle_file_path = Simulation._get_pickle_filename(root_path, chid)

        if path and os.path.isfile(path):
            with open(pickle_file_path) as f:
                sim = pickle.load(f)
            if not isinstance(sim, cls):
                os.remove(pickle_file_path)
            else:
                if sim._hash == create_hash(smv_file_path):
                    return sim

        return super(Simulation, cls).__new__(cls)

    def __init__(self, path: str):
        """
        :param path: Either the path to the directory containing the simulation data or direct path
            to the .smv file for the simulation in case that multiple simulation output was written to
            the same directory.
        """
        # Check if the file has already been instantiated via a cached pickle file
        if not hasattr(self, "_hash"):
            self.smv_file_path = get_smv_file(path)

            self.root_path = os.path.dirname(self.smv_file_path)

            with open(self.smv_file_path, 'r') as infile, mmap.mmap(infile.fileno(), 0,
                                                                    access=mmap.ACCESS_READ) as smv_file:
                smv_file.seek(smv_file.find(b'VERSION', 0))
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

                # Hash will be saved to simulation pickle file and compared to new hash when loading
                # the pickled simulation again in the next run of the program.
                self._hash = create_hash(self.smv_file_path)
                pickle.dump(self,
                            open(Simulation._get_pickle_filename(self.root_path, self.chid), 'wb'))

    @classmethod
    def _get_pickle_filename(cls, root_path: str, chid: str):
        """Get the filename used to save the pickled simulation.
        """
        return root_path + chid + ".pickle"

    def _load_meshes(self, smv_file: mmap.mmap) -> List[Mesh]:
        """Method to load the mesh information from the smv file.
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
                               read_dimension(b'Z', nz, pos + 1), mesh_id, len(meshes),
                               smv_file, pos, self.surfaces,
                               self.default_texture_origin))

            pos = smv_file.find(b'GRID', pos + 1)

        return meshes

    def _load_surfaces(self, smv_file: mmap.mmap) -> List[Surface]:
        """Method to load the surface information from the smv file.
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
    def slices(self) -> SliceCollection:
        """Lazy loads all slices for the simulation.

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
                        slices[slice_id] = list()
                    slices[slice_id].append(
                        {"extent": Extent(*index_ranges), "mesh": self.meshes[mesh_index],
                         "filename": filename, "quantity": quantity, "label": label, "unit": unit})

                    pos = smv_file.find(b'SLC', pos + 1)

            if len(slices) == 0:
                raise IOError("This simulation did not output any slices.")

            self._slices = SliceCollection(
                Slice(self.root_path, cell_centered, slice_data) for slice_data in slices.values())
        return self._slices

    @property
    def boundaries(self) -> BoundaryCollection:
        """Lazy loads all boundary data for the simulation.
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
                self._boundaries = BoundaryCollection(boundaries.values())
            else:
                raise IOError("This simulation did not output any plot3d data.")
        return self._boundaries

    @property
    def data_3d(self) -> Plot3DCollection:
        """Lazy loads all plot3d data for the simulation.
        :returns: All plot3d data.
        """
        # Todo: Also read SMOKG3D data?
        # Only load plot3d data once initially and then reuse the loaded information
        if not hasattr(self, "_plot3ds"):
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
                self._plot3ds = Plot3DCollection(plot3ds.values())
            else:
                raise IOError("This simulation did not output any plot3d data.")
        return self._plot3ds

    @property
    def isosurfaces(self) -> IsosurfaceCollection:
        """Lazy loads all isosurfaces for the simulation.
        :returns: All isof data.
        """
        # Only load isosurfaces once initially and then reuse the loaded information
        if not hasattr(self, "_isosurfaces"):
            isosurfaces = dict()
            with open(self.smv_file_path, 'r') as infile, \
                    mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as smv_file:
                pos = smv_file.find(b'ISOG', 0)
                while pos > 0:
                    smv_file.seek(pos - 1)
                    double_quantity = smv_file.read(1) == b'T'

                    mesh_index = int(smv_file.readline().decode().strip().split()[1]) - 1

                    iso_filename = smv_file.readline().decode().strip()
                    iso_id = int(iso_filename.split('_')[-1][:-4])

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
                        if iso_id not in isosurfaces:
                            isosurfaces[iso_id] = Isosurface(iso_id,
                                                             os.path.dirname(self.smv_file_path),
                                                             double_quantity, quantity, label, unit,
                                                             v_quantity=v_quantity, v_label=v_label,
                                                             v_unit=v_unit)
                        isosurfaces[iso_id]._add_subsurface(self.meshes[mesh_index], iso_filename,
                                                            viso_filename=viso_filename)
                    else:
                        if iso_id not in isosurfaces:
                            isosurfaces[iso_id] = Isosurface(iso_id,
                                                             os.path.dirname(self.smv_file_path),
                                                             double_quantity, quantity, label, unit)
                        isosurfaces[iso_id]._add_subsurface(self.meshes[mesh_index], iso_filename)

                    pos = smv_file.find(b'ISOG', pos + 1)
            if len(isosurfaces) > 0:
                self._isosurfaces = IsosurfaceCollection(isosurfaces.values())
            else:
                raise IOError("This simulation did not output any isosurfaces.")
        return self._isosurfaces

    def clear_cache(self, clear_persistent_cache=False):
        """Remove all data from the internal cache that has been loaded so far to free memory.

        :param clear_persistent_cache: Whether to clear the persistent simulation cache as well.
        """
        if hasattr(self, "_slices"):
            self._slices.clear_cache()
        if hasattr(self, "_boundaries"):
            self._boundaries.clear_cache()
        if hasattr(self, "_plot3ds"):
            self._plot3ds.clear_cache()
        if hasattr(self, "_isosurfaces"):
            self._isosurfaces.clear_cache()

        if clear_persistent_cache:
            os.remove(Simulation._get_pickle_filename(self.root_path, self.chid))
