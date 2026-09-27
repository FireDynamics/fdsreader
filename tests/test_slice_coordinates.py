"""Regression test for GitHub issue #103: IndexError when a mesh has only a single
cell along one axis (e.g. a 2D-ish mesh like IJK=100,1,40), reproduced by the
Pohlhausen validation case from the FDS test suite."""

from types import SimpleNamespace

import numpy as np
import pytest

from fdsreader.slcf.slice import Slice, SubSlice
from fdsreader.utils import Extent


def _make_subslice(cell_centered: bool) -> SubSlice:
    mesh = SimpleNamespace(
        coordinates={
            "x": np.linspace(0, 1.25, 101),
            "y": np.array([-0.1, 0.1]),  # single cell along y, like IJK=100,1,40
            "z": np.linspace(0, 0.5, 41),
        }
    )
    parent_slice = SimpleNamespace(cell_centered=cell_centered)
    extent = Extent(0, 1.25, -0.1, 0.1, 0, 0.5)
    return SubSlice(parent_slice, "", None, extent, mesh)


def test_get_coordinates_single_cell_axis_cell_centered():
    subslice = _make_subslice(cell_centered=True)
    coords = subslice.get_coordinates()
    assert coords["y"] == pytest.approx([0.0])


def test_get_coordinates_single_cell_axis_not_cell_centered():
    subslice = _make_subslice(cell_centered=False)
    coords = subslice.get_coordinates()
    assert list(coords["y"]) == pytest.approx([-0.1, 0.1])


def test_slice_get_coordinates_cell_centered_oriented():
    """Regression test for GitHub issue #118: Slice.get_coordinates() raised
    AttributeError: 'str' object has no attribute 'coordinates' for cell-centered
    2D slices, because it iterated `self._subslices.keys()` (mesh ids) instead of
    `self._subslices.values()` (SubSlice objects) to look up a mesh's coordinates."""
    mesh = SimpleNamespace(
        id="mesh1",
        coordinates={
            "x": np.linspace(0, 3, 31),
            "y": np.linspace(0, 4, 41),
            "z": np.linspace(0, 3, 31),
        },
    )

    slc = Slice.__new__(Slice)
    slc.cell_centered = True
    slc.orientation = 2  # PBY slice, fixed in y
    slc.extent = Extent(0, 3, 2, 2, 0, 3)
    slc._subslices = {mesh.id: SubSlice(slc, "", None, slc.extent, mesh)}

    coords = slc.get_coordinates()

    # The fixed dimension's single coordinate is shifted to the center of the cell
    # below y=2, i.e. the midpoint between the two nearest grid lines (1.9 and 2.0).
    assert coords["y"] == pytest.approx([1.95])


def test_slice_get_coordinates_empty_dim_fallback_uses_mesh_object(monkeypatch):
    """Regression test: the "no coordinates found" fallback in Slice.get_coordinates()
    (fdsreader/slcf/slice.py) had the same self._subslices.keys()/values() mix-up as
    GitHub issue #118 and would raise AttributeError: 'str' object has no attribute
    'coordinates' if it were ever triggered, e.g. if a subslice's coordinates for a
    dimension ended up not overlapping the slice's extent at all.

    The extent's x-coordinate (5.0) is placed beyond the mesh's last x-coordinate
    (3.0) on purpose: this drives np.searchsorted's index to mesh_coords.size, which
    also exercised a second, independent bug in the same fallback (it incremented the
    index instead of decrementing it, causing an IndexError one past the array end)."""
    mesh = SimpleNamespace(
        id="mesh1",
        coordinates={
            "x": np.linspace(0, 3, 31),
            "y": np.linspace(0, 4, 41),
            "z": np.linspace(0, 3, 31),
        },
    )

    slc = Slice.__new__(Slice)
    slc.cell_centered = False
    slc.orientation = 0  # 3D slice: every dimension goes through the general branch
    slc.extent = Extent(5, 5, 0, 4, 0, 3)
    subslice = SubSlice(slc, "", None, slc.extent, mesh)
    slc._subslices = {mesh.id: subslice}

    # Simulate the "no overlapping coordinate found" case for the x dimension, which
    # is what forces Slice.get_coordinates() into its buggy fallback branch.
    monkeypatch.setattr(
        subslice,
        "get_coordinates",
        lambda ignore_cell_centered=False: {"x": np.array([]), "y": np.array([2.0]), "z": np.array([1.0])},
    )

    coords = slc.get_coordinates()
    # The nearest available mesh coordinate is the last one (3.0), not an out-of-bounds index.
    assert coords["x"] == pytest.approx([3.0])
