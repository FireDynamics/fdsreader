from typing import Iterable

from fdsreader.isof import Isosurface
from fdsreader.utils.collection_mixins import QuantityFilterMixin
from fdsreader.utils.data import FDSDataCollection


class IsosurfaceCollection(QuantityFilterMixin, FDSDataCollection):
    """Collection of :class:`Isosurface` objects. Offers extensive functionality for filtering and
    using isosurfaces as well as its subclasses such as :class:`SubSurface`.
    """

    def __init__(self, *isosurfaces: Iterable[Isosurface]):
        super().__init__(*isosurfaces)

    def __repr__(self):
        return "IsosurfaceCollection(" + super().__repr__() + ")"
