from typing import Iterable, Union, List

from fdsreader.isof import Isosurface
from fdsreader.utils.data import FDSDataCollection, Quantity


class IsosurfaceCollection(FDSDataCollection):
    """Collection of :class:`Isosurface` objects. Offers extensive functionality for filtering and
        using isosurfaces as well as its subclasses such as :class:`SubSurface`.
    """

    def __init__(self, *isosurfaces: Iterable[Isosurface]):
        super().__init__(*isosurfaces)

    def filter_by_quantity(self, quantity: Union[str, Quantity]) -> List[Isosurface]:
        """Get the isosurface for a specific quantity.
        """
        if type(quantity) != str:
            quantity = quantity.quantity
        return [x for x in self if x.quantity.quantity.lower() == quantity.lower() or
                x.quantity.label.lower() == quantity.lower()]

    def __repr__(self):
        return "IsosurfaceCollection(" + super(IsosurfaceCollection, self).__repr__() + ")"
