"""Shared mixins that unify duplicated lookup/filter contracts across the various
``*Collection`` classes (each of which is built on top of :class:`FDSDataCollection`).

Not every ``*Collection`` uses every mixin here: a collection only picks up the mixins whose
contract actually fits its elements (e.g. a single `.quantity` per element vs. several). Some
near-identical-looking methods are deliberately left un-unified because they aren't actually the
same contract - see CONTRIBUTING.md's collection-contract entry for the specifics (e.g.
``ObstructionCollection.get_nearest`` iterates sub-obstructions and has no tie-break sort, unlike
this module's :class:`NearestByExtentMixin`; ``DeviceCollection.__getitem__`` also matches
grouped lists of same-named devices, unlike this module's :class:`ClassNameLookupMixin`).
"""

from typing import List, Union

import numpy as np

from fdsreader.utils.data import Quantity


def matches_quantity(element, quantity: str) -> bool:
    """Whether `element.quantity` (a :class:`Quantity`) matches `quantity` by name or short_name,
    case-insensitively. Shared by :class:`QuantityFilterMixin` and the single-match
    `get_by_quantity` of `Smoke3DCollection`/`Plot3DCollection`, which use the same matching rule
    but return a single element (raising `KeyError`) instead of a filtered collection.
    """
    return element.quantity.name.lower() == quantity.lower() or element.quantity.short_name.lower() == quantity.lower()


class QuantityFilterMixin:
    """Mixin for collections whose elements each expose a single `.quantity` (a :class:`Quantity`)."""

    @property
    def quantities(self) -> List[Quantity]:
        return list({element.quantity for element in self})

    def filter_by_quantity(self, quantity: Union[str, Quantity]):
        """Filters all elements by a specific quantity."""
        if isinstance(quantity, Quantity):
            quantity = quantity.name
        return type(self)(element for element in self if matches_quantity(element, quantity))


class IdLookupMixin:
    """Mixin for collections whose elements each expose an `.id`."""

    def get_by_id(self, element_id: str):
        """Get the element with the corresponding id if it exists."""
        return next((element for element in self if element.id == element_id), None)


class NearestByExtentMixin:
    """Mixin for collections whose elements each expose an `.extent` (an :class:`Extent`)."""

    def get_nearest(self, x: float = None, y: float = None, z: float = None):
        """Filters the element with the shortest distance to the given point.
        If there are multiple elements with the same distance, a random one will be selected.
        """
        d_min = np.finfo(float).max
        candidates = list()

        for element in self:
            dx = max(element.extent.x_start - x, 0, x - element.extent.x_end) if x is not None else 0
            dy = max(element.extent.y_start - y, 0, y - element.extent.y_end) if y is not None else 0
            dz = max(element.extent.z_start - z, 0, z - element.extent.z_end) if z is not None else 0
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d < d_min:
                d_min = d
                candidates = [element]
            elif d == d_min:
                candidates.append(element)

        if x is not None:
            candidates.sort(key=lambda element: element.extent.x_end - element.extent.x_start)
        if y is not None:
            candidates.sort(key=lambda element: element.extent.y_end - element.extent.y_start)
        if z is not None:
            candidates.sort(key=lambda element: element.extent.z_end - element.extent.z_start)

        if len(candidates) > 0:
            return candidates[0]
        return None


class ClassNameLookupMixin:
    """Mixin for collections whose elements each expose a `.class_name`, additionally
    look-up-able by that name (besides plain integer indexing into `._elements`).
    """

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._elements[key]
        for element in self:
            if element.class_name == key:
                return element

    def __contains__(self, value):
        if value in self._elements:
            return True
        for element in self:
            if element.class_name == value:
                return True
        return False
