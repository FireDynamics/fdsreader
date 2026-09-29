"""Regression tests for the numpy __array_ufunc__/__array__/mean/std protocol shared
(as copy-pasted code) by Slice, GeomSlice, Smoke3D and Plot3D. None of these bugs were
caught by the existing acceptance tests before this file was added:
- Slice.vmin/vmax raised TypeError on NumPy >= 2 (registered under np.amin/np.amax only).
- std() read `self.mean` (a bound method) instead of `self.mean()`.
- __array_ufunc__ discarded operand order (`2 - slc` computed `slc - 2`).
- Smoke3D/Plot3D's __array_ufunc__ mutated the original object instead of a copy.
- __array_ufunc__ forwarded numpy's `out` kwarg, causing infinite recursion on `+=`.
"""

import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def slcf_sim():
    return Simulation("./steckler_data")


@pytest.fixture(scope="module")
def pl3d_sim():
    return Simulation("./pl3d_data")


@pytest.fixture(scope="module")
def geomslice_sim():
    return Simulation("./geomslice_data_fds611")


def _check_array_like(obj, subitems):
    """Shared checks for Slice/GeomSlice/Smoke3D/Plot3D: vmin/vmax, std(), operand order,
    non-mutation of the original, __array__ raising TypeError, and the `out=` in-place path.
    `subitems` is the list of per-mesh sub-objects (e.g. obj.subslices) to read raw data from.
    """
    # vmin/vmax must work at all (previously TypeError on NumPy >= 2 for Slice). Not every
    # class exposes both (Smoke3D only has vmax, Plot3D has neither).
    if hasattr(obj, "vmin") and hasattr(obj, "vmax"):
        assert obj.vmin <= obj.vmax
    elif hasattr(obj, "vmax"):
        assert obj.vmax is not None

    # std() must not crash (previously "unsupported operand type(s) for -: 'float' and 'method'").
    std = np.std(obj)
    assert std >= 0
    assert not np.isnan(std)

    # __array__ must refuse to convert, but as a TypeError, not a UserWarning.
    with pytest.raises(TypeError):
        np.asarray(obj)

    # Operand order must be preserved: `2 - obj` must be the negation of `obj - 2`.
    before = subitems[0].data.copy()
    left = 2.0 - obj
    right = obj - 2.0
    assert np.allclose(_first_subitem(left).data, -_first_subitem(right).data)

    # The original object must be untouched by either operation.
    assert np.array_equal(subitems[0].data, before)

    # In-place operators must not recurse infinitely via numpy's `out` kwarg.
    import copy

    clone = copy.deepcopy(obj)
    clone += 1.0
    assert np.allclose(_first_subitem(clone).data, before + 1.0)

    # Combining two instances of the same class is explicitly unsupported and must raise a
    # clear UserWarning instead of crashing with an unrelated IndexError.
    with pytest.raises(UserWarning):
        np.add(obj, obj)


def _first_subitem(obj):
    for attr in ("_subslices", "_subgeomslices", "_subsmokes", "_subplots"):
        if hasattr(obj, attr):
            return next(iter(getattr(obj, attr).values()))
    raise AssertionError(f"Could not find sub-items dict on {obj!r}")


def test_slice_array_protocol(slcf_sim):
    slc = slcf_sim.slices[0]
    _check_array_like(slc, slc.subslices)


def test_smoke3d_array_protocol(slcf_sim):
    smoke = slcf_sim.smoke_3d[0]
    _check_array_like(smoke, smoke.subsmokes)


def test_plot3d_array_protocol(pl3d_sim):
    pl3d = next(p for p in pl3d_sim.data_3d if len(p._subplots) > 0)
    _check_array_like(pl3d, list(pl3d._subplots.values()))


def test_geomslice_array_protocol(geomslice_sim):
    geomslice = geomslice_sim.geomslices[0]
    _check_array_like(geomslice, list(geomslice._subgeomslices.values()))


def test_plot3d_placeholder_has_no_quantity(pl3d_sim):
    """Plot3D placeholders that were never assigned data must not crash __repr__."""
    empty = [p for p in pl3d_sim.data_3d if len(p._subplots) == 0]
    for p in empty:
        assert p.quantity is None
        assert "Plot3D" in repr(p)
