import os
from typing import List, BinaryIO, Tuple
import numpy as np

from utils import Quantity, Mesh, Extent
import utils.fortran_data as fdtype


class Boundary:
    def __init__(self, root_path: str, cell_centered: bool, quantity: str, label: str, unit: str):
        self.root_path = root_path
        self.cell_centered = cell_centered
        self.quantity = Quantity(quantity, label, unit)

        self._subplots: List[_SubBoundary] = list()

        self.times = None

    def _add_subboundary(self, filename: str, mesh: Mesh):
        file_path = os.path.join(self.root_path, filename)
        self._subplots.append(_SubBoundary(file_path, mesh, self))


class Patch:
    def __init__(self, extent: Extent, orientation: int, obst_index: int):
        self.extent = extent
        self.orientation = orientation
        self.obst_index = obst_index

    def _init(self, t_n: int):
        self.data = np.empty((t_n,) + self._get_dimension())

    def _get_dimension(self):
        if self.orientation == 1:
            dim = (1, {self.extent.y + 2}, {self.extent.z + 2})
        elif self.orientation == 2:
            dim = ({self.extent.x + 2}, 1, {self.extent.z + 2})
        else:
            dim = ({self.extent.x + 2}, {self.extent.y + 2}, 1)
        return dim

    def read_data(self, infile: BinaryIO, offset: int, t: int) -> Tuple[float, int]:
        """
        Method to load the quantity data for a single patch.
        """
        time = fdtype.read(infile, fdtype.FLOAT, 1, offset)
        offset += fdtype.FLOAT.itemsize

        dtype_data = fdtype.new((('f', str(self._get_dimension())),))
        self.data[t, :] = fdtype.read(infile, dtype_data, 1, offset)
        offset += dtype_data.itemsize

        return time, offset


class _SubBoundary:
    """

    """

    def __init__(self, file_path: str, mesh: Mesh, parent_boundary: Boundary):
        self.file_path = file_path
        self.mesh = mesh

        self._patches = list()
        with open(self.file_path, 'rb') as infile:
            self._offset = 3 * fdtype.new((('c', 30),)).itemsize
            n_patches = fdtype.read(infile, fdtype.INT, 1, self._offset)
            self._offset += fdtype.INT.itemsize

            dtype_patches = fdtype.new((('i', 9),))
            patch_infos = fdtype.read(infile, dtype_patches, n_patches, self._offset)
            self._offset += dtype_patches.itemsize * n_patches

            for patch in patch_infos:
                co = self.mesh.coordinates
                self._patches.append(
                    Patch(Extent(co[0][patch[0]], co[0][patch[1]], co[1][patch[0]],
                                 co[1][patch[1]], co[2][patch[0]], co[2][patch[1]]),
                          patch[6], patch[7]))

        t_n = (os.stat(file_path).st_size - self._offset) // (fdtype.FLOAT.itemsize + fdtype.new(
            (('f', str(self._patches[0]._get_dimension())),)).itemsize)
        for patch in self._patches:
            patch._init(t_n)

        parent_boundary.times = np.empty(shape=(t_n,))
        self._times = parent_boundary.times

    def get_data(self) -> List[Patch]:
        """
        Method to lazy load the boundary data for all patches in a single mesh.
        """
        if not hasattr(self._patches[0], "data"):
            with open(self.file_path, 'rb') as infile:
                offset = self._offset
                for t in range(self._times.shape[0]):
                    for patch in self._patches:
                        time, offset = patch.read_data(infile, offset, t)
                        self._times[t] = time

        return self._patches
