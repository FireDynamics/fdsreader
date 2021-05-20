from typing import Iterable, Union, List

from fdsreader.pl3d import Plot3D
from fdsreader.utils.data import FDSDataCollection, Quantity


class Plot3DCollection(FDSDataCollection):
    """Collection of :class:`Plot3D` objects. Offers extensive functionality for filtering and
        using plot3Ds as well as its subclasses such as :class:`SubPlot3D`.
    """

    def __init__(self, times: Iterable[float], *plot3ds: Iterable[Plot3D]):
        super().__init__(*plot3ds)
        self.times = list(times)

    @property
    def quantities(self) -> List[Quantity]:
        qs = set()
        for pl3d in self:
            for q in pl3d.quantities:
                qs.add(q)
        return list(qs)

    def filter_by_quantity(self, quantity: Union[str, Quantity]):
        """Filters all plot3d data by a specific quantity.
        """
        if type(quantity) != str:
            quantity = quantity.quantity
        return Plot3DCollection(x for x in self if any(
            q.quantity.lower() == quantity.lower() or q.label.lower() == quantity.lower() for q in
            x.quantities))

    def __repr__(self):
        return "Plot3DCollection(" + super(Plot3DCollection, self).__repr__() + ")"
