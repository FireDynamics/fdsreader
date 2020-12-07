from typing import Iterable

from fdsreader.isof import Isosurface


class IsosurfaceCollection:
    """Collection of :class:`Isosurface` objects. Offers extensive functionality for filtering and
        using isosurfaces as well as its subclasses such as :class:`SubSurface`.
    """

    def __init__(self, *slices: Iterable[Isosurface]):
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
