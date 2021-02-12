from typing import Iterable

from fdsreader.pl3d import Plot3D
from fdsreader.utils.data import FDSDataCollection


class Plot3DCollection(FDSDataCollection):
    """Collection of :class:`Plot3D` objects. Offers extensive functionality for filtering and
        using plot3Ds as well as its subclasses such as :class:`SubPlot3D`.
    """

    def __init__(self, times: Iterable[float], *plot3ds: Iterable[Plot3D]):
        super().__init__(*plot3ds)
        self.times = list(times)

    def __repr__(self):
        return "Plot3DCollection(" + super(Plot3DCollection, self).__repr__() + ")"
