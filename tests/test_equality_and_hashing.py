"""Regression tests for __eq__/__hash__ added to Device, Mesh, Surface, Dimension, Profile,
Obstruction, and Quantity during this session's bug-fix round. Before these fixes, these classes
either had no __eq__/__hash__ at all (falling back to identity comparison, so two objects
describing the same FDS entity compared unequal) or, in Quantity's case, could crash when compared
against a non-Quantity/non-str value. Also covers the intentional string-comparison branch on
Quantity.__eq__ (needed so `"TEMPERATURE" in sim.slices.quantities` keeps working now that
`.quantities` returns Quantity objects instead of strings)."""

import numpy as np
import pytest

from fdsreader.bndf.obstruction import Obstruction
from fdsreader.devc.device import Device
from fdsreader.fds_classes.mesh import Mesh
from fdsreader.fds_classes.surface import Surface
from fdsreader.utils import Quantity
from fdsreader.utils.data import Profile
from fdsreader.utils.dimension import Dimension


def test_device_eq_by_id_and_string_comparison():
    q = Quantity("TEMPERATURE", "temp", "C")
    d1 = Device("d1", q, (0, 0, 0), (0, 0, 1))
    d2 = Device("d1", q, (1, 1, 1), (1, 0, 0))
    d3 = Device("d2", q, (0, 0, 0), (0, 0, 1))
    assert d1 == d2
    assert hash(d1) == hash(d2)
    assert d1 != d3
    assert d1 == "d1"
    assert d1 != "d2"
    assert (d1 == 42) is False


def test_mesh_eq_by_id():
    coords = {"x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0]), "z": np.array([0.0, 1.0])}
    extents = {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)}
    m1 = Mesh(coords, extents, "mesh1")
    m2 = Mesh(coords, {"x": (0.0, 2.0), "y": (0.0, 1.0), "z": (0.0, 1.0)}, "mesh1")
    m3 = Mesh(coords, extents, "mesh2")
    assert m1 == m2
    assert hash(m1) == hash(m2)
    assert m1 != m3
    assert (m1 == "mesh1") is False


def test_surface_eq_by_name():
    s1 = Surface("INERT", 0.0, 0.9, 0, 1.0, 1.0, None, (1.0, 1.0, 1.0), 1.0)
    s2 = Surface("INERT", 100.0, 0.5, 1, 2.0, 2.0, "texture.png", (0.0, 0.0, 0.0), 0.5)
    s3 = Surface("BURNER", 0.0, 0.9, 0, 1.0, 1.0, None, (1.0, 1.0, 1.0), 1.0)
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert s1 != s3
    assert (s1 == 42) is False


def test_dimension_eq_by_xyz():
    d1 = Dimension(2, 3, 4)
    d2 = Dimension(2, 3, 4)
    d3 = Dimension(2, 3, 5)
    assert d1 == d2
    assert hash(d1) == hash(d2)
    assert d1 != d3
    assert (d1 == (2, 3, 4)) is False


def test_profile_eq_by_id():
    empty = np.array([])
    p1 = Profile("prof1", empty, empty, empty, empty)
    p2 = Profile("prof1", np.array([1.0]), np.array([2]), np.array([0.1]), np.array([300.0]))
    p3 = Profile("prof2", empty, empty, empty, empty)
    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert p1 != p3
    assert (p1 == "prof1") is False


def test_obstruction_eq_by_id():
    o1 = Obstruction("obst1", -1, -1, (0.0, 0.0, 0.0))
    o2 = Obstruction("obst1", -2, 0, (1.0, 1.0, 1.0))
    o3 = Obstruction("obst2", -1, -1, (0.0, 0.0, 0.0))
    assert o1 == o2
    assert hash(o1) == hash(o2)
    assert o1 != o3
    assert (o1 == 42) is False


class TestQuantityStringComparison:
    def test_quantity_eq_by_full_fields(self):
        q1 = Quantity("TEMPERATURE", "temp", "C")
        q2 = Quantity("TEMPERATURE", "temp", "C")
        q3 = Quantity("TEMPERATURE", "temp", "K")
        assert q1 == q2
        assert hash(q1) == hash(q2)
        assert q1 != q3

    def test_quantity_eq_returns_false_for_unrelated_type(self):
        q = Quantity("TEMPERATURE", "temp", "C")
        assert (q == 42) is False

    @pytest.mark.parametrize("candidate", ["TEMPERATURE", "temperature", "temp", "TEMP"])
    def test_quantity_matches_name_or_short_name_case_insensitively(self, candidate):
        q = Quantity("TEMPERATURE", "temp", "C")
        assert q == candidate

    def test_quantity_does_not_match_unrelated_string(self):
        q = Quantity("TEMPERATURE", "temp", "C")
        assert q != "VELOCITY"

    def test_quantities_list_supports_string_membership(self):
        # Regression test: SliceCollection.quantities/GeomSliceCollection.quantities now return
        # Quantity objects (not strings), but "TEMPERATURE" in sim.slices.quantities must still
        # work via this __eq__ branch (list membership uses == directly, no __hash__ involved).
        quantities = [Quantity("TEMPERATURE", "temp", "C"), Quantity("VELOCITY", "vel", "m/s")]
        assert "TEMPERATURE" in quantities
        assert "temperature" in quantities
        assert "does_not_exist" not in quantities
