from typing import Iterable, List, Union

from fdsreader.slcf import Slice
from fdsreader.utils.data import FDSDataCollection, Quantity


class SliceCollection(FDSDataCollection):
    """Collection of :class:`Slice` objects. Offers extensive functionality for filtering and
        using slices as well as its subclasses such as :class:`SubSlice`.
    """

    def __init__(self, *slices: Iterable[Slice]):
        super().__init__(*slices)

    def filter_by_quantity(self, quantity: Union[str, Quantity]) -> List[Slice]:
        """Filters all obstructions for its boundary data by a specific quantity.
        """
        if type(quantity) != str:
            quantity = quantity.quantity
        return [x for x in self if x.quantity.quantity.lower() == quantity.lower() or
                x.quantity.label.lower() == quantity.lower()]

    def get_by_id(self, slice_id: str):
        """Get the slice with corresponding id if it exists.
        """
        return next((slc for slc in self if slc.id == slice_id), None)

    def __repr__(self):
        return "SliceCollection(" + super(SliceCollection, self).__repr__() + ")"
