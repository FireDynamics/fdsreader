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


@pytest.fixture(scope="module")
def steckler_fds6100_sim():
    return Simulation("./steckler_data_fds6100")


@pytest.fixture(scope="module")
def pl3d_fds6100_sim():
    return Simulation("./pl3d_data_fds6100")


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

    # np.mean(obj) must dispatch through __array_function__/_HANDLED_FUNCTIONS to obj.mean(),
    # not silently fall through to something else. Only np.std(obj) was covered here before,
    # which internally calls self.mean() as a plain method call - a class whose module-level
    # _HANDLED_FUNCTIONS registration for np.mean was ever dropped would still pass every check
    # in this function except this one (confirmed by reproducing that exact regression during a
    # code review of the mixin extraction: np.mean(obj) silently returned a bound method object
    # instead of raising or computing anything).
    assert np.mean(obj) == obj.mean()

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


# The fixtures above cover FDS 6.11 output only (via geomslice_data_fds611) for GeomSlice; the
# rest use whatever the default fixtures happen to be. The two tests below explicitly exercise
# the shared NumpyArrayMixin against real FDS 6.10.1 output for Slice/Smoke3D/Plot3D, since
# CONTRIBUTING.md notes the *_fds6100.tgz fixtures were previously "unused by any test" - this
# also served as the manual verification requested when extracting the mixin (does it still work
# against a second, real FDS version, not just the one everything else already happens to use).
def test_slice_and_smoke3d_array_protocol_on_fds6100_output(steckler_fds6100_sim):
    slc = steckler_fds6100_sim.slices[0]
    _check_array_like(slc, slc.subslices)

    # smoke_3d[0] (SOOT DENSITY) is all-zero for this fixture; HRRPUV has real non-zero data and
    # actually exercises the numeric checks below instead of trivially satisfying them at 0.0.
    smoke = steckler_fds6100_sim.smoke_3d.get_by_quantity("HRRPUV")
    _check_array_like(smoke, smoke.subsmokes)


def test_plot3d_array_protocol_on_fds6100_output(pl3d_fds6100_sim):
    pl3d = next(p for p in pl3d_fds6100_sim.data_3d if len(p._subplots) > 0)
    _check_array_like(pl3d, list(pl3d._subplots.values()))


def test_handled_functions_registry_is_not_shared_across_classes(slcf_sim, geomslice_sim):
    """Each of Slice/GeomSlice/Smoke3D/Plot3D keeps its own module-level _HANDLED_FUNCTIONS dict
    even though they all now share NumpyArrayMixin.mean/std - np.mean(a_slice) must not
    accidentally dispatch through GeomSlice's registry or vice versa."""
    from fdsreader.slcf.geomslice import _HANDLED_FUNCTIONS as geomslice_funcs
    from fdsreader.slcf.geomslice import GeomSlice
    from fdsreader.slcf.slice import _HANDLED_FUNCTIONS as slice_funcs
    from fdsreader.slcf.slice import Slice

    assert slice_funcs is not geomslice_funcs
    assert Slice._handled_functions is slice_funcs
    assert GeomSlice._handled_functions is geomslice_funcs

    slc = slcf_sim.slices[0]
    geomslice = geomslice_sim.geomslices[0]
    assert np.mean(slc) == slc.mean()
    assert np.mean(geomslice) == geomslice.mean()
