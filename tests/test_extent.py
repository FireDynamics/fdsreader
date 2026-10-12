"""Regression tests for Extent's argument validation, __getitem__ bounds, and hash/eq contract
(bugs P0/P2 fixed earlier: a constructed-but-never-raised ValueError, silent acceptance of a
wrong-but-even argument count, and item=0 silently wrapping to the z-extent via Python's
negative indexing instead of raising)."""

import pytest

from fdsreader.utils import Extent


def test_wrong_odd_argument_count_raises_value_error():
    with pytest.raises(ValueError):
        Extent(1, 2, 3)


def test_wrong_even_argument_count_is_no_longer_silently_accepted():
    # 8 args (one pair too many) used to be silently accepted (only even-ness was checked);
    # now anything other than exactly 6 (or 4 with skip_dimension) must raise.
    with pytest.raises(ValueError):
        Extent(1, 2, 3, 4, 5, 6, 7, 8)


def test_correct_six_arguments_is_accepted():
    e = Extent(0, 1, 2, 3, 4, 5)
    assert (e.x_start, e.x_end) == (0.0, 1.0)
    assert (e.y_start, e.y_end) == (2.0, 3.0)
    assert (e.z_start, e.z_end) == (4.0, 5.0)


def test_skip_dimension_accepts_four_arguments():
    e = Extent(2, 3, 4, 5, skip_dimension="x")
    assert (e.x_start, e.x_end) == (0.0, 0.0)
    assert (e.y_start, e.y_end) == (2.0, 3.0)
    assert (e.z_start, e.z_end) == (4.0, 5.0)


def test_skip_dimension_with_wrong_argument_count_raises():
    with pytest.raises(ValueError):
        Extent(2, 3, 4, 5, 6, skip_dimension="x")


@pytest.mark.parametrize("item", [1, 2, 3])
def test_getitem_accepts_valid_integer_indices(item):
    e = Extent(0, 1, 2, 3, 4, 5)
    assert e[item] == e._extents[item - 1]


@pytest.mark.parametrize("dim", ["x", "y", "z"])
def test_getitem_accepts_valid_string_indices(dim):
    e = Extent(0, 1, 2, 3, 4, 5)
    assert e[dim] == (getattr(e, f"{dim}_start"), getattr(e, f"{dim}_end"))


def test_getitem_zero_raises_instead_of_silently_returning_z_extent():
    # Regression test: `self._extents[item - 1]` with item=0 used to silently wrap around to
    # `self._extents[-1]` (the z-extent) via Python's negative indexing.
    e = Extent(0, 1, 2, 3, 4, 5)
    with pytest.raises(IndexError):
        e[0]


def test_getitem_out_of_range_raises():
    e = Extent(0, 1, 2, 3, 4, 5)
    with pytest.raises(IndexError):
        e[4]


def test_hash_matches_equal_objects():
    e1 = Extent(0, 1, 2, 3, 4, 5)
    e2 = Extent(0, 1, 2, 3, 4, 5)
    e3 = Extent(0, 1, 2, 3, 4, 6)
    assert e1 == e2
    assert hash(e1) == hash(e2)
    assert e1 != e3
    assert len({e1, e2, e3}) == 2


def test_eq_returns_notimplemented_for_unrelated_type_instead_of_crashing():
    e = Extent(0, 1, 2, 3, 4, 5)
    assert (e == 42) is False
    assert (e == "not an extent") is False
