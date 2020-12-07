from typing import Iterable

from fdsreader.plot3d import Plot3D


class Plot3DCollection:
    """Collection of :class:`Plot3D` objects. Offers extensive functionality for filtering and
        using plot3Ds as well as its subclasses such as :class:`SubPlot3D`.
    """

    def __init__(self, *slices: Iterable[Plot3D]):
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
