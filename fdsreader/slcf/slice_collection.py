from typing import Iterable

from fdsreader.slcf import Slice
from fdsreader.utils.collection_mixins import IdLookupMixin, NearestByExtentMixin, QuantityFilterMixin
from fdsreader.utils.data import FDSDataCollection


class SliceCollection(QuantityFilterMixin, IdLookupMixin, NearestByExtentMixin, FDSDataCollection):
    """Collection of :class:`Slice` objects. Offers additional functionality for filtering and
    using slices as well as its subclasses such as :class:`SubSlice`.
    """

    def __init__(self, *slices: Iterable[Slice]):
        super().__init__(*slices)

    def __repr__(self):
        return "SliceCollection(" + super().__repr__() + ")"
