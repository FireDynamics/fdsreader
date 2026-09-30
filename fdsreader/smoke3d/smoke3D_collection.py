from typing import Iterable, List, Union

from fdsreader.smoke3d import Smoke3D
from fdsreader.utils.collection_mixins import matches_quantity
from fdsreader.utils.data import FDSDataCollection, Quantity


class Smoke3DCollection(FDSDataCollection):
    """Collection of :class:`Smoke3D` objects. Offers extensive functionality for filtering and using Smoke3Ds as well
    as its subclasses such as :class:`SubSmoke3D`.
    """

    def __init__(self, *smoke3ds: Iterable[Smoke3D]):
        super().__init__(*smoke3ds)

    @property
    def quantities(self) -> List[Quantity]:
        return [smoke3d.quantity for smoke3d in self._elements]

    def get_by_quantity(self, quantity: Union[Quantity, str]):
        """Gets the :class:`Smoke3D` with a specific quantity."""
        if isinstance(quantity, Quantity):
            quantity = quantity.name
        x = next((x for x in self._elements if matches_quantity(x, quantity)), None)
        if x is None:
            raise KeyError(f"No Smoke3D found for quantity '{quantity}'")
        return x

    def __repr__(self):
        return "Smoke3DCollection(" + super().__repr__() + ")"
