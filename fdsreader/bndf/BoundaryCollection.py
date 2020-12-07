from typing import Iterable

from fdsreader.bndf import Boundary


class BoundaryCollection:
    """Collection of :class:`Boundary` objects. Offers extensive functionality for filtering and
        using boundaries as well as its subclasses such as :class:`SubBoundary`.
    """

    def __init__(self, *slices: Iterable[Boundary]):
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
        # TODO
        pass
