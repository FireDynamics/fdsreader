from typing import Iterable

from fdsreader.fds_classes import Mesh
from fdsreader.utils.collection_mixins import IdLookupMixin
from fdsreader.utils.data import FDSDataCollection


class MeshCollection(IdLookupMixin, FDSDataCollection):
    """Collection of :class:`Mesh` objects."""

    def __init__(self, *meshes: Iterable[Mesh]):
        super().__init__(*meshes)

    def __repr__(self):
        return "MeshCollection(" + super().__repr__() + ")"
