"""Regression tests for fdsreader.export, which was completely unimportable (and therefore
0% covered and never actually exercised) until a circular import in sim_exporter.py was fixed:
`from . import export_obst_raw, export_slcf_raw, export_smoke_raw` tried to read names off the
package's own __init__.py before it had finished initializing. Running these exporters against
real fixtures for the first time also surfaced two ZeroDivisionError guards that were checked
after the division instead of before.
"""

import os
from unittest.mock import MagicMock

import pytest

pytest.importorskip("multiprocess")
pytest.importorskip("pathos")
yaml = pytest.importorskip("yaml")

from fdsreader import Simulation  # noqa: E402
from fdsreader.export import export_obst_raw, export_sim, export_slcf_raw, export_smoke_raw  # noqa: E402


def test_export_package_imports():
    """Regression test for the circular import that made the whole export subpackage
    unimportable (sim_exporter.py importing sibling functions via the not-yet-initialized
    package __init__ instead of directly from their modules). The module-level import above
    already exercises this fix (a regression would fail collection of this whole file), so this
    just names the four functions the fix restored access to."""
    assert callable(export_sim)
    assert callable(export_obst_raw)
    assert callable(export_slcf_raw)
    assert callable(export_smoke_raw)


@pytest.fixture(scope="module")
def bndf_sim():
    return Simulation("./bndf_data")


@pytest.fixture(scope="module")
def steckler_sim():
    return Simulation("./steckler_data")


def test_export_obst_raw(tmp_path, bndf_sim):
    obst = bndf_sim.obstructions[0]
    meta_path = export_obst_raw(obst, str(tmp_path))
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert meta["NumQuantities"] == len(obst.quantities)
    for quantity_dir in os.listdir(tmp_path):
        full = os.path.join(tmp_path, quantity_dir)
        if os.path.isdir(full):
            assert len(os.listdir(full)) > 0


def test_export_obst_raw_guard_precedes_division():
    """Regression test for a ZeroDivisionError: the "no useful data" guard (`if vmax <= 0:
    return`) must run before `"ScaleFactor": 255.0 / vmax` is computed, not after. Testing this
    end-to-end would need a fixture with an all-zero boundary quantity (e.g. burning rate on a
    cold wall) and a multiprocessing worker isn't easily mockable, so this checks source order
    directly instead.
    """
    import inspect

    import fdsreader.export.obst_exporter as obst_exporter_module

    source = inspect.getsource(obst_exporter_module.export_obst_raw)
    guard_pos = source.index("if vmax <= 0")
    division_pos = source.index("255.0 / vmax")
    assert guard_pos < division_pos


def test_export_slcf_raw(tmp_path, steckler_sim):
    slc = steckler_sim.slices[0]
    meta_path = export_slcf_raw(slc, str(tmp_path))
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert meta["Quantity"] == slc.quantity.name
    assert meta["MeshNum"] == len(slc.subslices)


def test_export_slcf_raw_constant_slice_returns_empty(tmp_path):
    """A slice with vmax == vmin (no useful data range) must return "" instead of raising
    ZeroDivisionError from 255.0 / (vmax - vmin)."""
    fake_slice = MagicMock()
    fake_slice.type = "2D"
    fake_slice.vmax = 5.0
    fake_slice.vmin = 5.0
    result = export_slcf_raw(fake_slice, str(tmp_path))
    assert result == ""


def test_export_smoke_raw(tmp_path, steckler_sim):
    smoke = steckler_sim.smoke_3d.get_by_quantity("HRRPUV")  # steckler_data's SOOT DENSITY is all zero
    meta_path = export_smoke_raw(smoke, str(tmp_path))
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert meta["Quantity"] == smoke.quantity.name


def test_export_smoke_raw_zero_data_returns_empty(tmp_path, steckler_sim):
    """A quantity with no useful data range (e.g. SOOT DENSITY where nothing burns) must
    return "" instead of raising ZeroDivisionError from 255.0 / DataValMax."""
    smoke = steckler_sim.smoke_3d.get_by_quantity("SOOT DENSITY")
    assert max(s.data.max() for s in smoke.subsmokes) == 0.0
    assert export_smoke_raw(smoke, str(tmp_path)) == ""


def test_export_sim(tmp_path, bndf_sim):
    export_sim(bndf_sim, str(tmp_path))
    # export_sim writes one <chid>-smv.yaml meta file plus a per-obstruction/slice/volume raw
    # export; check both the top-level meta and at least one real per-item output landed, not
    # just that the tmp_path directory (which pytest already creates) still exists.
    meta_path = os.path.join(str(tmp_path), bndf_sim.chid + "-smv.yaml")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert meta["NumObstructions"] == len(bndf_sim.obstructions)
    obst_dir = os.path.join(str(tmp_path), "obst")
    assert os.path.isdir(obst_dir)
    assert len(os.listdir(obst_dir)) > 0
