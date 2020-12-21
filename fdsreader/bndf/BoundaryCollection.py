from typing import Iterable, Dict

from fdsreader.bndf import Boundary
from fdsreader.utils import Obstruction


class BoundaryCollection:
    """Collection of :class:`Boundary` objects. Offers extensive functionality for filtering and
        using boundaries as well as its subclasses such as :class:`SubBoundary`.
    """

    def __init__(self, *boundaries: Iterable[Boundary]):
        # Maps from obstruction-id to
        self._boundaries: Dict[int, Boundary] = dict()

    def __getitem__(self, key):
        if type(key) == int:
            return self._boundaries[key]
        elif type(key) == Obstruction:
            return self._boundaries[key.id]
        else:
            raise KeyError()

    def __iter__(self):
        return self._elements.__iter__()

    def __len__(self):
        return len(self._elements)

    def __contains__(self, value):
        return value in self._elements

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for element in self._elements:
            element.clear_cache()
