from typing import Iterable, Union, List

from fdsreader import settings
from fdsreader.geom import GeomBoundary
from fdsreader.utils.data import FDSDataCollection, Quantity


class GeometryCollection(FDSDataCollection):
    """Collection of :class:`GeomBoundary` objects. Offers extensive functionality for filtering and
        using geometry data.
    """

    def __init__(self, *geom_boundaries: Iterable[GeomBoundary]):
        super().__init__(*geom_boundaries)

        if not settings.LAZY_LOAD:
            for geom in self:
                geom._load_data()

    @property
    def quantities(self) -> List[Quantity]:
        return list({geom.quantity for geom in self})

    def filter_by_quantity(self, quantity: Union[str, Quantity]):
        """Filters all GeomBoundaries by a specific quantity.
        """
        if type(quantity) != str:
            quantity = quantity.quantity
        return GeometryCollection(x for x in self if
                                  x.quantity.quantity.lower() == quantity.lower() or x.quantity.label.lower() == quantity.lower())

    def __repr__(self):
        return "GeometryCollection(" + super(GeometryCollection, self).__repr__() + ")"
