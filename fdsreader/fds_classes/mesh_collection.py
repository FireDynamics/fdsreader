from typing import Dict, Iterable, Tuple, Union

import numpy as np

from fdsreader.bndf.obstruction import _global_boundary_arrays
from fdsreader.fds_classes import Mesh
from fdsreader.utils import Quantity
from fdsreader.utils.data import FDSDataCollection


class MeshCollection(FDSDataCollection):
    """Collection of :class:`Obstruction` objects. Offers extensive functionality for filtering and
    using obstructions as well as dependent such as :class:`Boundary`.
    """

    def __init__(self, *meshes: Iterable[Mesh]):
        super().__init__(*meshes)

    def get_by_id(self, mesh_id: str):
        """Get the mesh with corresponding id if it exists."""
        return next((mesh for mesh in self if mesh.id == mesh_id), None)

    def get_global_boundary_data_arrays(
        self, quantity: Union[str, Quantity]
    ) -> Dict[int, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        """Combines the boundary data on the walls of the simulation domain, e.g. the floor, over all meshes into one
        array per orientation (-3=-z, -2=-y, -1=-x, 1=x, 2=y, 3=z; 3 is the floor, facing +z).

        Faces of obstructions are not included, see :meth:`Obstruction.get_global_boundary_data_arrays`, which also
        describes the shape of the arrays and when an orientation maps to a tuple of arrays. FDS writes these data
        per mesh face; with ``MB`` vents in a multi-mesh case (see the FDS User Guide), it may also write faces
        between two meshes, which then appear as further positions.

        :param quantity: The quantity's name or short name.
        """
        return _global_boundary_arrays([mesh.get_boundary_data for mesh in self], quantity)

    def __repr__(self):
        return "MeshCollection(" + super().__repr__() + ")"
