import os
from typing import List
import numpy as np

from utils import Quantity, Mesh
import utils.fortran_data as fdtype


class Plot3D:
    """

    """
    # Todo: Merge multiple Plot3Ds into one object and make Plot3D numpy-compliant
    def __init__(self, root_path: str, filename: str, time: float, quantities: List[Quantity],
                 mesh: Mesh):
        self.file_path = os.path.join(root_path, filename)
        self.time = time
        self.quantities = quantities
        self.mesh = mesh

        self._offset = fdtype.new((('i', 3),)).itemsize + fdtype.new((('i', 4),)).itemsize

    def get_data(self) -> np.ndarray:
        """
        Method to lazy load the 3D data.
        :returns: 4D numpy array wiht (x,y,z,q) as dimensions, while q represents the 5 quantities.
        """
        if not hasattr(self, "_data"):
            with open(self.file_path, 'rb') as infile:
                dtype_data = fdtype.new((('f', self.mesh.extent.size(cell_centered=False) * 5),))
                self._data = fdtype.read(infile, dtype_data, 1, offset=self._offset)[0][0].reshape(
                    (self.mesh.extent.x, self.mesh.extent.y, self.mesh.extent.z, 5))
        return self._data


class _SubPlot3D:
    def __init__(self):
        pass
