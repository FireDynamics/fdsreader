from typing import Iterable, Iterator

from fdsreader.slcf import Slice
from fdsreader.utils.data import FDSDataCollection


class SliceCollection(FDSDataCollection):
    """Collection of :class:`Slice` objects. Offers extensive functionality for filtering and
        using slices as well as its subclasses such as :class:`SubSlice`.
    """

    def __init__(self, *slices: Iterable[Slice]):
        super().__init__(*slices)
