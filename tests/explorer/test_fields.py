"""Slice fields read one time step at a time, checked against the eager path of fdsreader.

Uses the stored cases steckler_data (one node-based slice in one mesh) and slcf_masks_fds611 (three meshes, node-based
and cell-centred slices, 3D slices, and a cell-centred slice on a mesh border).
"""

import shutil

import numpy as np
import pytest
from fds_cases import CASES_DIR, load

from fdsreader.explorer.fields import SliceField, _bnd_range, _dedup, fields_of


@pytest.fixture(scope="module")
def steckler():
    return load(CASES_DIR / "steckler_data")


@pytest.fixture(scope="module")
def masks():
    return load(CASES_DIR / "slcf_masks_fds611")


def test_only_planar_slices_are_fields(masks):
    fields = fields_of(masks)
    assert {f.source.id for f in fields} == {s.id for s in masks.slices if s.orientation != 0}
    assert all(f.drawable for f in fields)
    assert not SliceField(masks.slices.get_by_id("vol")).drawable  # a 3D slice has no plane to draw


@pytest.mark.parametrize("case, slice_id", [("steckler", ""), ("masks", "z04"), ("masks", "z04_cc"), ("masks", "y08")])
def test_a_frame_equals_the_eager_global_array(case, slice_id, request):
    sim = request.getfixturevalue(case)
    slc = sim.slices.get_by_id(slice_id) if slice_id else sim.slices[0]
    field = SliceField(slc)
    eager = slc.to_global()
    for t in (0, len(field.times) - 1):
        np.testing.assert_allclose(field.frame(t), eager[t], equal_nan=True)


def test_axes_and_coordinates(masks):
    field = SliceField(masks.slices.get_by_id("y08"))
    assert field.axes == ("x", "z")
    assert field.shape == (21, 11)
    np.testing.assert_allclose(field.coords["x"][[0, -1]], [0.0, 2.0])
    np.testing.assert_allclose(field.coords["y"], [0.8], atol=1e-6)


def test_a_cell_centred_slice_on_a_mesh_border_has_two_sides(masks):
    # x = 0.5 is the upper x face of M3: FDS writes the cells on either side of it
    slc = masks.slices.get_by_id("x05_cc")
    meshes = [{sub.mesh.id for sub, *_ in SliceField(slc, side=side)._placement} for side in (0, 1, 5)]
    assert meshes[0] != meshes[1]
    assert meshes[2] == meshes[1]  # a side beyond the last falls back to the last one


def test_value_range_from_the_bnd_files(steckler):
    field = SliceField(steckler.slices[0])
    low, high, diverging = field.value_range()
    assert (low, high) == pytest.approx(_bnd_range(steckler.slices[0]))
    assert not diverging
    assert field.cmap == "inferno"


def test_value_range_without_bnd_files_scans_the_frames(tmp_path):
    case = shutil.copytree(CASES_DIR / "steckler_data", tmp_path / "steckler")
    for bnd in case.glob("*.sf.bnd"):
        bnd.unlink()
    slc = load(case).slices[0]
    assert _bnd_range(slc) is None
    field = SliceField(slc)
    frames = np.stack([field.frame(t) for t in range(len(field.times))])
    assert field.value_range()[:2] == pytest.approx((np.nanmin(frames), np.nanmax(frames)))


def test_dedup_drops_coordinates_written_with_limited_precision():
    np.testing.assert_array_equal(_dedup([0.2, 0.1, 0.1000001, 0.3]), [0.1, 0.2, 0.3])
    assert _dedup([]).size == 0


def test_clear_cache_keeps_the_field_usable(steckler):
    field = SliceField(steckler.slices[0])
    before = field.frame(1)
    field.clear_cache()
    np.testing.assert_array_equal(field.frame(1), before)
