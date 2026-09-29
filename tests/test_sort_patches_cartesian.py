"""Unit tests for bndf/utils.py::sort_patches_cartesian, a pure function with 0% coverage
before this file (cheap to test directly with stand-in Patch objects, no simulation needed)."""

from types import SimpleNamespace

from fdsreader.bndf.utils import sort_patches_cartesian
from fdsreader.utils import Extent


def _patch(orientation, x, y, z):
    return SimpleNamespace(orientation=orientation, extent=Extent(*x, *y, *z))


def test_empty_list_returns_empty_list():
    assert sort_patches_cartesian([]) == []


def test_single_patch_returns_single_row():
    p = _patch(3, (0, 1), (0, 1), (0, 0))
    assert sort_patches_cartesian([p]) == [[p]]


def test_z_orientation_groups_by_x_start_sorted_by_y_start():
    # A 2x2 grid of z-normal patches (orientation=3): grouped into rows by x_start, each row
    # sorted by y_start.
    p_00 = _patch(3, (0, 1), (0, 1), (0, 0))
    p_01 = _patch(3, (0, 1), (1, 2), (0, 0))
    p_10 = _patch(3, (1, 2), (0, 1), (0, 0))
    p_11 = _patch(3, (1, 2), (1, 2), (0, 0))

    result = sort_patches_cartesian([p_11, p_00, p_10, p_01])  # shuffled input

    assert result == [[p_00, p_01], [p_10, p_11]]


def test_x_orientation_groups_by_y_start_sorted_by_z_start():
    p_00 = _patch(1, (0, 0), (0, 1), (0, 1))
    p_01 = _patch(1, (0, 0), (0, 1), (1, 2))
    p_10 = _patch(1, (0, 0), (1, 2), (0, 1))

    result = sort_patches_cartesian([p_10, p_01, p_00])

    assert result == [[p_00, p_01], [p_10]]


def test_y_orientation_groups_by_x_start_sorted_by_z_start():
    p_00 = _patch(2, (0, 1), (0, 0), (0, 1))
    p_01 = _patch(2, (0, 1), (0, 0), (1, 2))
    p_10 = _patch(2, (1, 2), (0, 0), (0, 1))

    result = sort_patches_cartesian([p_10, p_01, p_00])

    assert result == [[p_00, p_01], [p_10]]


def test_negative_orientation_uses_absolute_value():
    # A patch on the "negative" x face (orientation=-1) must be treated the same as orientation=1.
    p_00 = _patch(-1, (0, 0), (0, 1), (0, 1))
    p_10 = _patch(-1, (0, 0), (1, 2), (0, 1))

    result = sort_patches_cartesian([p_10, p_00])

    assert result == [[p_00], [p_10]]
