"""Obstruction masks and the planes of cell-centred slices (#132), on slcf_masks_fds611 (FDS 6.11.1; input in
tests/cases/fds_inputs/slcf_masks.fds).

Two meshes (x 0..1 and 1..2, 0.1 m cells) with a fire on the floor and three obstructions: 'Box' inside the first
mesh, 'Across' through the mesh boundary at x = 1 and 'Span' up to the outside of the domain at x = 2. FDS writes a
placeholder (the ambient temperature, 20 C) into the cells inside an obstruction and into the nodes inside it,
while the nodes on its faces get values averaged from the gas next to it, unless there is no gas beyond the face.
After 10 s of fire, no gas cell or node is at exactly 20 C, so the mask must block exactly the placeholder values.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from fdsreader import Simulation, settings

CASES_DIR = Path(__file__).parent / "cases"
SLICE_IDS = ["vol", "vol_cc", "z04", "z04_cc", "z045_cc", "z02", "z02_cc", "x05_cc", "y08"]
PLACEHOLDER = 20.0

# As in test_slice_partial_xb.py: also run against the output of the newest FDS release in the compatibility workflow
_CASES = [CASES_DIR / "slcf_masks_fds611"]
_FRESH = Path(os.environ.get("FDS_FRESH_OUTPUT_DIR") or "/nonexistent") / "slcf_masks"
if _FRESH.is_dir():
    _CASES.append(_FRESH)


@pytest.fixture(scope="module", params=_CASES, ids=lambda case: "fresh" if case == _FRESH else "fds611")
def sim(request):
    original = settings.ENABLE_CACHING
    settings.ENABLE_CACHING = False
    try:
        yield Simulation(str(request.param))
    finally:
        settings.ENABLE_CACHING = original


@pytest.mark.parametrize("slice_id", SLICE_IDS)
def test_mask_blocks_exactly_the_placeholder_values(sim, slice_id):
    for sub in sim.slices.get_by_id(slice_id).subslices:
        data = sub.data[-1]
        if sub.orientation:
            data = np.expand_dims(data, sub.orientation - 1)
        blocked = ~sub.mesh.get_obstruction_mask_slice(sub)[-1]
        np.testing.assert_array_equal(blocked, data == PLACEHOLDER, err_msg=f"{slice_id} in {sub.mesh.id}")


def test_masks_differ_for_nodes_and_cells(sim):
    # 'Box' spans the nodes 3..7, 3..6, 2..6 of M1: its cells are 3..6, 3..5, 2..5, its inner nodes 4..6, 4..5, 3..5
    mesh = sim.meshes[0]
    times = sim.slices.get_by_id("vol").times
    cells = ~mesh.get_obstruction_mask(times, cell_centered=True)[-1]
    nodes = ~mesh.get_obstruction_mask(times)[-1]
    assert cells[3:7, 3:6, 2:6].all() and cells.sum() == 4 * 3 * 4 + 2 * 2 * 4  # Box and the part of 'Across'
    assert nodes[4:7, 4:6, 3:6].all()
    assert not nodes[3, 4, 4] and not nodes[7, 4, 4]  # the faces of 'Box' at x = 0.3 and 0.7 hold gas values


def test_nodes_on_a_face_without_gas_beyond_are_blocked(sim):
    m1, m2 = sim.meshes[0], sim.meshes[1]
    times = sim.slices.get_by_id("vol").times
    # 'Across' continues from M1 into M2 at x = 1.0 (M1's last node, M2's first node)
    assert not m1.get_obstruction_mask(times)[-1, 10, 8, 4]
    assert not m2.get_obstruction_mask(times)[-1, 0, 8, 4]
    # 'Span' ends at the outside of the domain at x = 2.0
    assert m2.exterior_faces["x"] == (False, True)
    assert not m2.get_obstruction_mask(times)[-1, 10, 4, 4]


@pytest.mark.parametrize(
    "slice_id, axis, layer, centre",
    [("z04_cc", "z", 4, 0.45), ("z045_cc", "z", 4, 0.45), ("z02_cc", "z", 2, 0.25), ("x05_cc", "x", 5, 0.55)],
)
def test_cell_centred_plane_is_the_cell_below_the_node_in_the_smv_file(sim, slice_id, axis, layer, centre):
    # FDS writes a cell-centred plane SLCC ... k k for the cell below node k; its data equals that layer of the
    # cell-centred 3D slice
    plane = sim.slices.get_by_id(slice_id)
    volume = sim.slices.get_by_id("vol_cc")
    sub = next(s for s in plane.subslices if s.mesh.id == "M1")
    ref = next(s for s in volume.subslices if s.mesh.id == "M1")
    frames = [(i, int(np.argmin(np.abs(volume.times - t)))) for i, t in enumerate(plane.times)]
    frames = [(i, j) for i, j in frames if np.isclose(plane.times[i], volume.times[j], atol=1e-3)]
    assert len(frames) > 1
    axis_index = "xyz".index(axis)
    for i, j in frames:
        np.testing.assert_array_equal(sub.data[i], np.take(ref.data[j], layer, axis=axis_index))
    assert sub.get_coordinates()[axis] == pytest.approx([centre])
    assert plane.get_coordinates()[axis] == pytest.approx([centre])


def test_to_global_coordinates_of_cell_centred_slices_are_the_cell_centres(sim):
    _, coords = sim.slices.get_by_id("z04_cc").to_global(return_coordinates=True)
    np.testing.assert_allclose(coords["x"], np.arange(0.05, 2.0, 0.1), atol=1e-6)
    np.testing.assert_allclose(coords["y"], np.arange(0.05, 1.0, 0.1), atol=1e-6)
    assert coords["z"] == pytest.approx([0.45])
    _, coords = sim.slices.get_by_id("vol_cc").to_global(return_coordinates=True)
    np.testing.assert_allclose(coords["z"], np.arange(0.05, 1.0, 0.1), atol=1e-6)
    _, coords = sim.slices.get_by_id("z04").to_global(return_coordinates=True)
    np.testing.assert_allclose(coords["x"], np.arange(0.0, 2.05, 0.1), atol=1e-6)


@pytest.mark.parametrize("slice_id", ["z04", "z04_cc", "vol", "vol_cc"])
def test_to_global_masked_hides_exactly_the_placeholder_values(sim, slice_id):
    slc = sim.slices.get_by_id(slice_id)
    values = slc.to_global(masked=True, fill=np.nan)[-1]
    raw = slc.to_global()[-1]
    np.testing.assert_array_equal(np.isnan(values), raw == PLACEHOLDER)
