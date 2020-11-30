import os
from typing import List, BinaryIO, Tuple
import numpy as np

from fdsreader.utils import Quantity, Mesh, Extent
import fdsreader.utils.fortran_data as fdtype


class Boundary:
    """Boundary file data container including metadata. Consists of multiple subslices, one for each mesh
     the slice cuts through.

    :ivar id: The ID of this boundary file.
    :ivar root_path: Path to the directory containing all boundary files.
    :ivar quantities: List with quantity objects containing information about the quantities.
     calculated for this slice with the corresponding label and unit.
    :ivar cell_centered: Indicates whether centered positioning for data is used.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar sub_boundaries: List of SubBoundary objects containing all boundary data in a single mesh.
    """

    def __init__(self, boundary_id: int, root_path: str, cell_centered: bool, quantity: str,
                 label: str, unit: str):
        self.id = boundary_id
        self.root_path = root_path
        self.cell_centered = cell_centered
        self.quantity = Quantity(quantity, label, unit)

        self.sub_boundaries: List[_SubBoundary] = list()

        self.times = None

    def _add_subboundary(self, filename: str, mesh: Mesh):
        """Created a SubBoundary object and adds it to the list of sub_boundaries.
        """
        file_path = os.path.join(self.root_path, filename)
        self.sub_boundaries.append(_SubBoundary(file_path, mesh, self))


class Patch:
    """Container for the actual data which is stored as rectangular plane with specific orientation and
     extent.

    :ivar extent: Extent object containing 3-dimensional extent information.
    :ivar orientation: The direction the patch is facing (x={-1;1}, y={-2;2}, z={-3;3}).
    :ivar obst_index: Index of the obstacle
    :ivar data: Numpy ndarray with the actual data.
    """

    def __init__(self, extent: Extent, orientation: int, obst_index: int):
        self.extent = extent
        self.orientation = orientation
        self.obst_index = obst_index

    def _init(self, t_n: int):
        """Initialize the data array as soon as the number of calculated time steps is known.
        """
        self.data = np.empty((t_n,) + self._get_dimension())

    def _get_dimension(self):
        """Convenience function to calculate the shape of the array containing the data for this patch.
        """
        if abs(self.orientation) == 1:
            dim = (1, self.extent.y + 2, self.extent.z + 2)
        elif abs(self.orientation) == 2:
            dim = (self.extent.x + 2, 1, self.extent.z + 2)
        else:
            dim = (self.extent.x + 2, self.extent.y + 2, 1)
        return dim

    def read_data(self, infile: BinaryIO, t: int) -> Tuple[float, int]:
        """Method to load the quantity data for a single patch.
        """
        time = fdtype.read(infile, fdtype.FLOAT, 1)
        dtype_data = fdtype.new((('f', str(self._get_dimension())),))
        self.data[t, :] = fdtype.read(infile, dtype_data, 1)
        return time


class _SubBoundary:
    """Contains all boundary data for a single mesh subdivided into patches.

    :ivar file_path: The path to the file containing data for this specific SubBoundary.
    :ivar mesh: The mesh containing all boundary data in this SubBoundary.
    """

    def __init__(self, file_path: str, mesh: Mesh, parent_boundary: Boundary):
        self.file_path = file_path
        self.mesh = mesh

        self._patches = list()
        with open(self.file_path, 'rb') as infile:
            # Offset of the binary file to the end of the file header.
            self._offset = 3 * fdtype.new((('c', 30),)).itemsize
            infile.seek(self._offset)

            n_patches = fdtype.read(infile, fdtype.INT, 1)[0][0][0]
            dtype_patches = fdtype.new((('i', 9),))
            patch_infos = fdtype.read(infile, dtype_patches, n_patches)[0]

            self._offset += fdtype.INT.itemsize + dtype_patches.itemsize * n_patches

            for patch in patch_infos:
                co = self.mesh.coordinates
                self._patches.append(
                    Patch(Extent(co[0][patch[0]], co[0][patch[1]], co[1][patch[2]],
                                 co[1][patch[3]], co[2][patch[4]], co[2][patch[5]]),
                          patch[6], patch[7]))

        t_n = (os.stat(file_path).st_size - self._offset) // (fdtype.FLOAT.itemsize + fdtype.new(
            (('f', str(self._patches[0]._get_dimension())),)).itemsize)
        for patch in self._patches:
            patch._init(t_n)

        parent_boundary.times = np.empty(shape=(t_n,))
        self._times = parent_boundary.times

    @property
    def patches(self) -> List[Patch]:
        """Method to lazy load the boundary data for all patches in a single mesh.

        :returns: The actual data in form of a list of patches (objects containing numpy ndarrays).
        """
        if not hasattr(self._patches[0], "data"):
            with open(self.file_path, 'rb') as infile:
                infile.seek(self._offset)
                for t in range(self._times.shape[0]):
                    for patch in self._patches:
                        time = patch.read_data(infile, t)
                        self._times[t] = time
        return self._patches
