"""Unit tests for the shared mixins in `fdsreader.utils.collection_mixins` that unify the
duplicated filter/lookup contract across the various `*Collection` classes:
- QuantityFilterMixin (`quantities`/`filter_by_quantity`): SliceCollection, GeomSliceCollection,
  IsosurfaceCollection, GeometryCollection.
- IdLookupMixin (`get_by_id`): SliceCollection, GeomSliceCollection, ObstructionCollection,
  MeshCollection.
- NearestByExtentMixin (`get_nearest`): SliceCollection, GeomSliceCollection.
- ClassNameLookupMixin (`__getitem__`/`__contains__` by `.class_name`): ParticleCollection,
  EvacCollection.

Also covers two bugs found while originally writing the Slice/GeomSlice tests below (now also
regression-tested for Isosurface/GeometryCollection, which had the same class of bug and were
NOT covered by the earlier fix):
- GeomSliceCollection.quantities read `slc.name` (GeomSlice has no such attribute, so this
  crashed for any non-empty collection) instead of `slc.quantity`. IsosurfaceCollection and
  GeometryCollection had the exact same bug (`iso.name`/`geom.name` - neither class has a `.name`
  attribute) until the QuantityFilterMixin extraction fixed all four at once.
- GeomSliceCollection.get_nearest had the same "accumulate every candidate seen instead of
  resetting on a strictly closer one" bug already fixed in SliceCollection.get_nearest.
"""

from types import SimpleNamespace

import pytest

from fdsreader.bndf.obstruction_collection import ObstructionCollection
from fdsreader.evac.evac_collection import EvacCollection
from fdsreader.fds_classes.mesh_collection import MeshCollection
from fdsreader.geom.geometry_collection import GeometryCollection
from fdsreader.isof.isosurface_collection import IsosurfaceCollection
from fdsreader.part.particle_collection import ParticleCollection
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


def _with_quantity(quantity_name):
    return SimpleNamespace(quantity=Quantity(quantity_name, quantity_name[:4], "C"))


@pytest.mark.parametrize("Collection", [SliceCollection, GeomSliceCollection, IsosurfaceCollection, GeometryCollection])
class TestQuantityFilterMixinContract:
    """IsosurfaceCollection/GeometryCollection share the exact same QuantityFilterMixin as
    Slice/GeomSlice, so this exercises all four against the same contract."""

    def test_quantities_returns_quantity_objects(self, Collection):
        coll = Collection([_with_quantity("TEMPERATURE")])
        quantities = coll.quantities
        assert len(quantities) == 1
        assert isinstance(quantities[0], Quantity)
        assert quantities[0].name == "TEMPERATURE"

    def test_filter_by_quantity_matches_short_name_case_insensitive(self, Collection):
        temp = _with_quantity("TEMPERATURE")
        velo = _with_quantity("VELOCITY")
        coll = Collection([temp, velo])
        filtered = coll.filter_by_quantity("temp")
        assert list(filtered) == [temp]
        # filter_by_quantity must return the same collection type it was called on, not a
        # hardcoded base type - relevant now that the mixin uses `type(self)(...)`.
        assert type(filtered) is Collection

    def test_filter_by_quantity_accepts_a_quantity_object(self, Collection):
        temp = _with_quantity("TEMPERATURE")
        coll = Collection([temp])
        filtered = coll.filter_by_quantity(Quantity("TEMPERATURE", "TEMP", "C"))
        assert list(filtered) == [temp]


def _with_id(element_id):
    return SimpleNamespace(id=element_id)


@pytest.mark.parametrize("Collection", [ObstructionCollection, MeshCollection])
class TestIdLookupMixinContract:
    def test_get_by_id_finds_matching_element(self, Collection):
        coll = Collection([_with_id("a"), _with_id("b")])
        assert coll.get_by_id("b").id == "b"

    def test_get_by_id_returns_none_when_missing(self, Collection):
        coll = Collection([_with_id("a")])
        assert coll.get_by_id("does-not-exist") is None


def _with_class_name(class_name):
    return SimpleNamespace(class_name=class_name)


class TestClassNameLookupMixinContract:
    """ParticleCollection/EvacCollection both look up elements by `.class_name` via
    ClassNameLookupMixin, on top of plain integer indexing. Constructed directly with fake
    elements to avoid needing real particle/evac binary fixtures."""

    def test_particle_collection_getitem_by_class_name(self):
        coll = ParticleCollection(times=[], particles=[_with_class_name("WATER PARTICLES")])
        assert coll["WATER PARTICLES"].class_name == "WATER PARTICLES"
        assert coll[0].class_name == "WATER PARTICLES"

    def test_particle_collection_contains_by_class_name(self):
        coll = ParticleCollection(times=[], particles=[_with_class_name("WATER PARTICLES")])
        assert "WATER PARTICLES" in coll
        assert "FUEL PARTICLES" not in coll

    def test_evac_collection_getitem_by_class_name(self):
        coll = EvacCollection(evacs=[_with_class_name("ADULT")], base_path="does-not-exist", times=[])
        assert coll["ADULT"].class_name == "ADULT"
        assert coll[0].class_name == "ADULT"

    def test_evac_collection_contains_by_class_name(self):
        coll = EvacCollection(evacs=[_with_class_name("ADULT")], base_path="does-not-exist", times=[])
        assert "ADULT" in coll
        assert "CHILD" not in coll
