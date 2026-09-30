from typing import Iterable

from fdsreader import settings
from fdsreader.geom import GeomBoundary
from fdsreader.utils.collection_mixins import QuantityFilterMixin
from fdsreader.utils.data import FDSDataCollection


class GeometryCollection(QuantityFilterMixin, FDSDataCollection):
    """Collection of :class:`GeomBoundary` objects. Offers extensive functionality for filtering and
    using geometry data.
    """

    def __init__(self, *geom_boundaries: Iterable[GeomBoundary]):
        super().__init__(*geom_boundaries)

        if not settings.LAZY_LOAD:
            for geom in self:
                geom._load_data()

    def __repr__(self):
        return "GeometryCollection(" + super().__repr__() + ")"
