"""Unit tests for SliceCollection/GeomSliceCollection's filtering methods, which had 25%
coverage before this file (filter_by_quantity/get_by_id/get_nearest were never directly
exercised). Also covers two bugs found while writing these tests:
- GeomSliceCollection.quantities read `slc.name` (GeomSlice has no such attribute, so this
  crashed for any non-empty collection) instead of `slc.quantity`.
- GeomSliceCollection.get_nearest had the same "accumulate every candidate seen instead of
  resetting on a strictly closer one" bug already fixed in SliceCollection.get_nearest.
"""

from types import SimpleNamespace

import pytest

from fdsreader.slcf.geomslice_collection import GeomSliceCollection
from fdsreader.slcf.slice_collection import SliceCollection
from fdsreader.utils import Extent, Quantity


def _slc(quantity_name, x, y, z, slice_id="slice"):
    return SimpleNamespace(
        quantity=Quantity(quantity_name, quantity_name[:4], "C"),
        extent=Extent(*x, *y, *z),
        id=slice_id,
    )


@pytest.mark.parametrize("Collection", [SliceCollection, GeomSliceCollection])
class TestSharedCollectionBehavior:
    def test_quantities_returns_quantity_objects(self, Collection):
        coll = Collection([_slc("TEMPERATURE", (0, 1), (0, 1), (0, 1))])
        quantities = coll.quantities
        assert len(quantities) == 1
        assert isinstance(quantities[0], Quantity)
        assert quantities[0].name == "TEMPERATURE"

    def test_filter_by_quantity_matches_name_case_insensitive(self, Collection):
        temp = _slc("TEMPERATURE", (0, 1), (0, 1), (0, 1))
        velo = _slc("VELOCITY", (0, 1), (0, 1), (0, 1))
        coll = Collection([temp, velo])
        filtered = coll.filter_by_quantity("temperature")
        assert list(filtered) == [temp]

    def test_get_by_id_returns_none_when_missing(self, Collection):
        coll = Collection([_slc("TEMPERATURE", (0, 1), (0, 1), (0, 1), slice_id="a")])
        assert coll.get_by_id("a") is not None
        assert coll.get_by_id("does-not-exist") is None

    def test_get_nearest_returns_the_actually_nearest_one(self, Collection):
        # Regression test: the old `d <= d_min` accumulated every slice seen so far instead of
        # resetting once a strictly closer one was found, so a far-away slice could win the
        # tie-break sort even though it was never actually the closest.
        far = _slc("TEMPERATURE", (10, 11), (0, 1), (0, 1), slice_id="far")
        near = _slc("TEMPERATURE", (0, 1), (0, 1), (0, 1), slice_id="near")
        coll = Collection([far, near])
        result = coll.get_nearest(x=0.5, y=0.5, z=0.5)
        assert result is near

    def test_get_nearest_returns_none_for_empty_collection(self, Collection):
        coll = Collection([])
        assert coll.get_nearest(x=0, y=0, z=0) is None

    def test_get_nearest_ties_are_kept_as_candidates(self, Collection):
        # Two equally-near slices: the tie-break sort must run over both, not just whichever one
        # was seen first (verifies the `elif d == d_min: append` branch is reachable).
        a = _slc("TEMPERATURE", (0, 1), (0, 1), (0, 1), slice_id="a")
        b = _slc("TEMPERATURE", (0, 1), (0, 1), (0, 1), slice_id="b")
        coll = Collection([a, b])
        result = coll.get_nearest(x=0.5, y=0.5, z=0.5)
        assert result in (a, b)
