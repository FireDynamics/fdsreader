from typing import Iterable, List, Union

from fdsreader.smoke3d import Smoke3D
from fdsreader.utils.data import FDSDataCollection, Quantity


class Smoke3DCollection(FDSDataCollection):
    """Collection of :class:`Smoke3D` objects. Offers extensive functionality for filtering and
        using Smoke3Ds as well as its subclasses such as :class:`SubSmoke3D` .
    """

    def __init__(self, *smoke3ds: Iterable[Smoke3D]):
        super().__init__(*smoke3ds)

    @property
    def quantities(self) -> List[Quantity]:
        return [smoke3d.name for smoke3d in self]

    def get_by_quantity(self, quantity: Union[Quantity, str]):
        """Gets the :class:`Smoke3D`s with a specific quantity.
        """
        if type(quantity) == Quantity:
            quantity = quantity.name
        return next(x for x in self if
                    x.quantity.name.lower() == quantity.lower() or x.quantity.short_name.lower() == quantity.lower())

    def __repr__(self):
        return "Smoke3DCollection(" + super(Smoke3DCollection, self).__repr__() + ")"
