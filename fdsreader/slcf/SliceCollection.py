from typing import Iterable, List, Union, Tuple

import numpy as np

from fdsreader.slcf import Slice
from fdsreader.utils.data import FDSDataCollection, Quantity


class SliceCollection(FDSDataCollection):
    """Collection of :class:`Slice` objects. Offers extensive functionality for filtering and
        using slices as well as its subclasses such as :class:`SubSlice`.
    """

    def __init__(self, *slices: Iterable[Slice]):
        super().__init__(*slices)

    def filter_by_quantity(self, quantity: Union[str, Quantity]):
        """Filters all slices by a specific quantity.
        """
        if type(quantity) != str:
            quantity = quantity.quantity
        return SliceCollection(x for x in self if x.quantity.quantity.lower() == quantity.lower()
                               or x.quantity.label.lower() == quantity.lower())

    def get_by_id(self, slice_id: str):
        """Get the slice with corresponding id if it exists.
        """
        return next((slc for slc in self if slc.id == slice_id), None)

    def get_nearest(self, x: float = None, y: float = None, z: float = None) -> Slice:
        """Filters the slice with the shortest distance to the given point.
            If there are multiple slices with the same distance, a random one will be selected.
        """
        d_min = np.finfo(float).max
        slices_min = list()

        for slc in self:
            dx = max(slc.extent.x_start - x, 0, x - slc.extent.x_end) if x is not None else 0
            dy = max(slc.extent.y_start - y, 0, y - slc.extent.y_end) if y is not None else 0
            dz = max(slc.extent.z_start - z, 0, z - slc.extent.z_end) if z is not None else 0
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d <= d_min:
                d_min = d
                slices_min.append(slc)

        if x is not None:
            slices_min.sort(key=lambda slc: (slc.extent.x_end - slc.extent.x_start))
        if y is not None:
            slices_min.sort(key=lambda slc: (slc.extent.y_end - slc.extent.y_start))
        if z is not None:
            slices_min.sort(key=lambda slc: (slc.extent.z_end - slc.extent.z_start))

        if len(slices_min) > 0:
            return slices_min[0]
        return None

    def __repr__(self):
        return "SliceCollection(" + super(SliceCollection, self).__repr__() + ")"
