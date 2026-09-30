from typing import Iterable

from fdsreader.slcf import GeomSlice
from fdsreader.utils.collection_mixins import IdLookupMixin, NearestByExtentMixin, QuantityFilterMixin
from fdsreader.utils.data import FDSDataCollection


class GeomSliceCollection(QuantityFilterMixin, IdLookupMixin, NearestByExtentMixin, FDSDataCollection):
    """Collection of :class:`GeomSlice` objects. Offers extensive functionality for filtering and
    using geomslices as well as its subclasses such as :class:`SubSlice`.
    """

    def __init__(self, *geomslices: Iterable[GeomSlice]):
        super().__init__(*geomslices)

    def __repr__(self):
        return "GeomSliceCollection(" + super().__repr__() + ")"
