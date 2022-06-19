from typing import Iterable, List, Union

import numpy as np

from fdsreader.slcf import GeomSlice
from fdsreader.utils.data import FDSDataCollection, Quantity


class GeomSliceCollection(FDSDataCollection):
    """Collection of :class:`GeomSlice` objects. Offers extensive functionality for filtering and
        using geomslices as well as its subclasses such as :class:`SubSlice`.
    """

    def __init__(self, *geomslices: Iterable[GeomSlice]):
        super().__init__(*geomslices)

    @property
    def quantities(self) -> List[Quantity]:
        return list({slc.name for slc in self})

    def filter_by_quantity(self, quantity: Union[str, Quantity]):
        """Filters all geomslices by a specific quantity.
        """
        if type(quantity) == Quantity:
            quantity = quantity.name
        return GeomSliceCollection(x for x in self if x.quantity.name.lower() == quantity.lower()
                               or x.quantity.short_name.lower() == quantity.lower())

    def get_by_id(self, geomslice_id: str):
        """Get the geomslice with corresponding id if it exists.
        """
        return next((slc for slc in self if slc.id == geomslice_id), None)

    def get_nearest(self, x: float = None, y: float = None, z: float = None) -> GeomSlice:
        """Filters the geomslice with the shortest distance to the given point.
            If there are multiple geomslices with the same distance, a random one will be selected.
        """
        d_min = np.finfo(float).max
        geomslices_min = list()

        for slc in self:
            dx = max(slc.extent.x_start - x, 0, x - slc.extent.x_end) if x is not None else 0
            dy = max(slc.extent.y_start - y, 0, y - slc.extent.y_end) if y is not None else 0
            dz = max(slc.extent.z_start - z, 0, z - slc.extent.z_end) if z is not None else 0
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= d_min:
                d_min = d
                geomslices_min.append(slc)

        if x is not None:
            geomslices_min.sort(key=lambda slc: (slc.extent.x_end - slc.extent.x_start))
        if y is not None:
            geomslices_min.sort(key=lambda slc: (slc.extent.y_end - slc.extent.y_start))
        if z is not None:
            geomslices_min.sort(key=lambda slc: (slc.extent.z_end - slc.extent.z_start))

        if len(geomslices_min) > 0:
            return geomslices_min[0]
        return None

    def __repr__(self):
        return "GeomSliceCollection(" + super(GeomSliceCollection, self).__repr__() + ")"
