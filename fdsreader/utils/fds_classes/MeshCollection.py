from typing import Iterable

from fdsreader.utils import Mesh
from fdsreader.utils.data import FDSDataCollection


class MeshCollection(FDSDataCollection):
    """Collection of :class:`Obstruction` objects. Offers extensive functionality for filtering and
        using obstructions as well as dependent such as :class:`Boundary`.
    """

    def __init__(self, *meshes: Iterable[Mesh]):
        super().__init__(*meshes)

    def get_by_id(self, mesh_id: str):
        """Get the mesh with corresponding id if it exists.
        """
        return next((mesh for mesh in self if mesh.id == mesh_id), None)

    def __repr__(self):
        return "MeshCollection(" + super(MeshCollection, self).__repr__() + ")"
