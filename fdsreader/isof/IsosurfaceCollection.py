from typing import Iterable

from fdsreader.isof import Isosurface


class IsosurfaceCollection:
    """Collection of :class:`Isosurface` objects. Offers extensive functionality for filtering and
        using isosurfaces as well as its subclasses such as :class:`SubSurface`.
    """

    def __init__(self, *isosurfaces: Iterable[Isosurface]):
        self._isosurfaces = tuple(*isosurfaces)

    def __getitem__(self, index):
        return self._isosurfaces[index]

    def __len__(self):
        return len(self._isosurfaces)

    def __contains__(self, value):
        return value in self._isosurfaces

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for isosurface in self._isosurfaces:
            isosurface.clear_cache()
