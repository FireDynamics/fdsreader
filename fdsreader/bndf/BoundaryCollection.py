from typing import Iterable, Iterator

from fdsreader.bndf import Boundary
from fdsreader.utils.data import FDSDataCollection


class BoundaryCollection(FDSDataCollection):
    """Collection of :class:`Boundary` objects. Offers extensive functionality for filtering and
        using boundaries as well as its subclasses such as :class:`SubBoundary`.
    """

    def __init__(self, *boundaries: Iterable[Boundary]):
        super().__init__(*boundaries)
