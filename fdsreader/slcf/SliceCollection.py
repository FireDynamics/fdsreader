from typing import Iterable

from fdsreader.slcf import Slice
from fdsreader.utils.data import FDSDataCollection


class SliceCollection(FDSDataCollection):
    """Collection of :class:`Slice` objects. Offers extensive functionality for filtering and
        using slices as well as its subclasses such as :class:`SubSlice`.
    """

    def __init__(self, *slices: Iterable[Slice]):
        super().__init__(*slices)

    def __repr__(self):
        return "SliceCollection(" + super(SliceCollection, self).__repr__() + ")"
