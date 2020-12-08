from typing import Iterable

from fdsreader.plot3d import Plot3D


class Plot3DCollection:
    """Collection of :class:`Plot3D` objects. Offers extensive functionality for filtering and
        using plot3Ds as well as its subclasses such as :class:`SubPlot3D`.
    """

    def __init__(self, *plot3ds: Iterable[Plot3D]):
        self._plot3ds = tuple(*plot3ds)

    def __getitem__(self, index):
        return self._plot3ds[index]

    def __len__(self):
        return len(self._plot3ds)

    def __contains__(self, value):
        return value in self._plot3ds

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for plot3d in self._plot3ds:
            plot3d.clear_cache()
