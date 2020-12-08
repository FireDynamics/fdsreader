from typing import Iterable

from fdsreader.slcf import Slice


class SliceCollection:
    """Collection of :class:`Slice` objects. Offers extensive functionality for filtering and
        using slices as well as its subclasses such as :class:`SubSlice`.
    """

    def __init__(self, *slices: Iterable[Slice]):
        self._slices = tuple(*slices)

    def __getitem__(self, index):
        return self._slices[index]

    def __len__(self):
        return len(self._slices)

    def __contains__(self, value):
        return value in self._slices

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for slc in self._slices:
            slc.clear_cache()
