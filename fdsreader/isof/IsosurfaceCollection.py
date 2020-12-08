from typing import Iterable, Iterator

from fdsreader.isof import Isosurface
from fdsreader.utils.data import FDSDataCollection


class IsosurfaceCollection(FDSDataCollection):
    """Collection of :class:`Isosurface` objects. Offers extensive functionality for filtering and
        using isosurfaces as well as its subclasses such as :class:`SubSurface`.
    """

    def __init__(self, *isosurfaces: Iterable[Isosurface]):
        super().__init__(*isosurfaces)
