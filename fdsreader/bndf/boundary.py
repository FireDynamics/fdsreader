import os
from operator import add
from typing import List, BinaryIO, Dict, Tuple
import numpy as np

from fdsreader.utils import Quantity, Mesh, Extent, settings, Obstruction
import fdsreader.utils.fortran_data as fdtype


class Patch:
    """Container for the actual data which is stored as rectangular plane with specific orientation
        and extent.

    :ivar extent: Extent object containing 3-dimensional extent information.
    :ivar orientation: The direction the patch is facing (x={-1;1}, y={-2;2}, z={-3;3}).
    :ivar data: Numpy ndarray with the actual data.
    :ivar t_n: Total number of time steps for which output data has been written.
    """

    def __init__(self, extent: Extent, orientation: int, cell_centered: bool):
        self.extent = extent
        self.orientation = orientation
        self.t_n = -1
        self.cell_centered = cell_centered

    @property
    def shape(self) -> Tuple:
        """Convenience function to calculate the shape of the array containing data for this patch.
        """
        if abs(self.orientation) == 1:
            dim = tuple(map(add, self.extent.shape(self.cell_centered), (0, 2, 2)))
        elif abs(self.orientation) == 2:
            dim = tuple(map(add, self.extent.shape(self.cell_centered), (2, 0, 2)))
        else:
            dim = tuple(map(add, self.extent.shape(self.cell_centered), (2, 2, 0)))
        return dim

    def _post_init(self, t_n: int):
        """Fully initialize the patch as soon as the number of timesteps is known.
        """
        self.t_n = t_n
        self.data = np.empty((self.t_n,) + self.shape)

    def _load_data(self, infile: BinaryIO, t: int):
        """Method to load the quantity data for a single patch for a single timestep.
        """
        dtype_data = fdtype.new((('f', str(self.shape)),))
        self.data[t, :] = fdtype.read(infile, dtype_data, 1)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "data"):
            del self.data

    def __repr__(self, *args, **kwargs):
        return f"Patch(shape={self.shape}, orientation={self.orientation})"


class SubBoundary:
    """Contains all boundary data for a single mesh subdivided into patches.

    :ivar file_path: The path to the file containing data for this specific :class:`SubBoundary`.
    :ivar mesh: The mesh containing all boundary data in this :class:`SubBoundary`.
    """

    def __init__(self, file_path: str, mesh: Mesh, cell_centered: bool):
        self.file_path = file_path
        self.mesh = mesh
        self.cell_centered = cell_centered

        self._patches: Dict[Obstruction, List[Patch]] = dict()
        with open(self.file_path, 'rb') as infile:
            # Offset of the binary file to the end of the file header.
            self._offset = 3 * fdtype.new((('c', 30),)).itemsize
            infile.seek(self._offset)

            n_patches = fdtype.read(infile, fdtype.INT, 1)[0][0][0]
            dtype_patches = fdtype.new((('i', 9),))
            patch_infos = fdtype.read(infile, dtype_patches, n_patches)[0]

            self._offset += fdtype.INT.itemsize + dtype_patches.itemsize * n_patches

            for patch_info in patch_infos:
                co = self.mesh.coordinates
                obst = mesh.obstructions[patch_info[7] + 1]
                extent = Extent(co[0][patch_info[0]], co[0][patch_info[1]], co[1][patch_info[2]],
                                co[1][patch_info[3]], co[2][patch_info[4]], co[2][patch_info[5]])
                orientation = patch_info[6]
                if obst not in self._patches:
                    self._patches[obst] = list()
                self._patches[obst].append(Patch(extent, orientation, self.cell_centered))

    @property
    def obstruction_data(self) -> Dict[Obstruction, List[Patch]]:
        """Method to lazy load the boundary data for all patches in a single mesh.

        :returns: The actual data in form of a dictionary that maps an obstruction to a list with
            all six sides of the cuboid obstruction, which are here called patches
            (objects containing numpy ndarrays).
        """
        if not hasattr(next(iter(self._patches.values()))[0], "data"):
            total_dim_size = 0
            for patches in self._patches.values():
                for patch in patches:
                    total_dim_size += fdtype.new((('f', str(patch.shape)),)).itemsize
            t_n = (os.stat(self.file_path).st_size - self._offset) // (
                    fdtype.FLOAT.itemsize + total_dim_size)
            for patches in self._patches.values():
                for patch in patches:
                    patch._post_init(t_n)
            self._times = np.empty(shape=(t_n,))

            with open(self.file_path, 'rb') as infile:
                infile.seek(self._offset)
                for t in range(self._times.shape[0]):
                    time = fdtype.read(infile, fdtype.FLOAT, 1)
                    self._times[t] = time
                    for patches in self._patches.values():
                        for patch in patches:
                            patch._load_data(infile, t)
        return self._patches

    @property
    def times(self):
        if not hasattr(self, "_times"):
            _ = self.obstruction_data
        return self._times

    def __getitem__(self, obst: Obstruction) -> List[Patch]:
        return self.obstruction_data[obst]

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for patches in self._patches.values():
            for patch in patches:
                patch.clear_cache()


class Boundary:
    """Boundary file data container including metadata. Consists of multiple subboundaries.

    :ivar id: The ID of this boundary.
    :ivar root_path: Path to the directory containing all boundary files.
    :ivar quantities: List of :class:`Quantity` objects containing information about the quantities
        calculated for this boundary with the corresponding label and unit.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar sub_boundaries: List of :class:`SubBoundary` objects containing all boundary data in a
        single mesh.
    """

    def __init__(self, boundary_id: int, root_path: str, cell_centered: bool, quantity: str,
                 label: str, unit: str):
        self.id = boundary_id
        self.root_path = root_path
        self.cell_centered = cell_centered
        self.quantity = Quantity(quantity, label, unit)

        self._subboundaries: Dict[Mesh, SubBoundary] = dict()

        self._times = None

    def _add_subboundary(self, filename: str, mesh: Mesh) -> SubBoundary:
        """Created a :class:`SubBoundary` object and adds it to the list of sub_boundaries.
        """
        subboundary = SubBoundary(os.path.join(self.root_path, filename), mesh, self.cell_centered)
        self._subboundaries[mesh] = subboundary

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            # Implicitly load the data for one subboundary
            _ = subboundary.obstruction_data

        return subboundary

    def __getitem__(self, mesh: Mesh):
        """Returns the :class:`SubBoundary` that contains data for the given mesh.
        """
        return self._subboundaries[mesh]

    @property
    def times(self):
        return next(iter(self._subboundaries.values())).times

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for subboundary in self._subboundaries.values():
            subboundary.clear_cache()
