from typing import Iterable, Union, List
import numpy as np

from fdsreader.pl3d import Plot3D
from fdsreader.utils.data import FDSDataCollection, Quantity


class Plot3DCollection(FDSDataCollection):
    """Collection of :class:`Plot3D` objects. Offers extensive functionality for filtering and
        using plot3Ds as well as its subclasses such as :class:`SubPlot3D`.
    """

    def __init__(self, *plot3ds: Iterable[Plot3D]):
        super().__init__(*plot3ds)

    @property
    def times(self) -> np.ndarray:
        return np.array(self._elements[0].times)

    @property
    def quantities(self) -> List[Quantity]:
        return [pl.quantity for pl in self._elements]

    def get_by_quantity(self, quantity: Union[str, Quantity]):
        """Filters all plot3d data by a specific quantity.
        """
        if type(quantity) == Quantity:
            quantity = quantity.name
        return next(x for x in self._elements if
                    x.quantity.name.lower() == quantity.lower() or x.quantity.short_name.lower() == quantity.lower())

    def __repr__(self):
        return "Plot3DCollection(" + super(Plot3DCollection, self).__repr__() + ")"
