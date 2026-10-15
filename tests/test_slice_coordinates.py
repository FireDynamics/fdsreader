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


def test_slice_get_coordinates_extent_outside_the_mesh_falls_back_to_nearest_node():
    """Regression test: a subslice whose extent does not overlap its mesh's coordinates along an axis used to
    leave Slice.get_coordinates() without coordinates there, and its own fallback had the same
    self._subslices.keys()/values() mix-up as GitHub issue #118 and an IndexError one past the array end.
    SubSlice.get_coordinates() now falls back to the nearest node itself (extent_index_range), so the slice
    gets that node.

    The extent's x-coordinate (5.0) lies beyond the mesh's last x-coordinate (3.0) on purpose."""
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
    slc._subslices = {mesh.id: SubSlice(slc, "", None, slc.extent, mesh)}

    coords = slc.get_coordinates()
    # The nearest available mesh coordinate is the last one (3.0), not an out-of-bounds index.
    assert coords["x"] == pytest.approx([3.0])
