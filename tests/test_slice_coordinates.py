"""Regression test for GitHub issue #103: IndexError when a mesh has only a single
cell along one axis (e.g. a 2D-ish mesh like IJK=100,1,40), reproduced by the
Pohlhausen validation case from the FDS test suite."""

from types import SimpleNamespace

import numpy as np
import pytest

from fdsreader.slcf.slice import SubSlice
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
