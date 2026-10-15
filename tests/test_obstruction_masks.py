"""Obstruction masks and the planes of cell-centred slices (#132), on slcf_masks_fds611 (FDS 6.11.1; input in
tests/cases/fds_inputs/slcf_masks.fds).

Three meshes: M1 (x 0..1) and M2 (x 1..2) with 0.1 m cells, y 0..1, and M3 (x 0..0.5, y 1..2), which adjoins M1
only partly. A fire covers the floor. Obstructions: inside a mesh ('Box'), across the mesh boundary ('Across'), up
to the outside of the domain ('Span'), two sharing a face ('Left', 'Right'), one with a hole ('Holey'), and two
ending at y = 1, where M3 adjoins ('Under') or the domain ends ('Outside').

FDS writes a placeholder, the ambient value (20 C), into the cells inside an obstruction, and into a node if all
cells of its mesh around it are solid (CORNER_VALUE in dump.f90; Smokeview blanks nodes the same way). After 10 s of
fire no gas cell or node is at exactly 20 C, so the mask must block exactly the placeholder values.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from fdsreader import Simulation, settings

CASES_DIR = Path(__file__).parent / "cases"
SLICE_IDS = ["vol", "vol_cc", "z04", "z04_cc", "z045_cc", "z02", "z02_cc", "z07", "x05_cc", "y08"]
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


def _mesh(sim, mesh_id):
    return next(mesh for mesh in sim.meshes if mesh.id == mesh_id)


def _shared_frames(a, b):
    """Index pairs of the frames two slices have in common."""
    pairs = [(i, int(np.argmin(np.abs(b.times - t)))) for i, t in enumerate(a.times)]
    pairs = [(i, j) for i, j in pairs if np.isclose(a.times[i], b.times[j], atol=1e-3)]
    assert len(pairs) > 1
    return pairs


@pytest.mark.parametrize("slice_id", SLICE_IDS)
def test_mask_blocks_exactly_the_placeholder_values(sim, slice_id):
    for sub in sim.slices.get_by_id(slice_id).subslices:
        data = sub.data[-1]
        if sub.orientation:
            data = np.expand_dims(data, sub.orientation - 1)
        blocked = ~sub.mesh.get_obstruction_mask_slice(sub)[-1]
        np.testing.assert_array_equal(blocked, data == PLACEHOLDER, err_msg=f"{slice_id} in {sub.mesh.id}")


def test_cells_and_nodes_of_an_obstruction(sim):
    # 'Box' spans the nodes 3..7, 3..6, 2..6 of M1: its cells are 3..6, 3..5, 2..5, its inner nodes 4..6, 4..5, 3..5
    times = sim.slices.get_by_id("vol").times
    m1 = _mesh(sim, "M1")
    cells = ~m1.get_obstruction_mask(times, cell_centered=True)[-1]
    nodes = ~m1.get_obstruction_mask(times)[-1]
    assert cells[3:7, 3:6, 2:6].all() and not cells[7, 4, 4] and not cells[2, 4, 4]
    assert nodes[4:7, 4:6, 3:6].all()
    assert not nodes[3, 4, 4] and not nodes[7, 4, 4]  # the faces of 'Box' at x = 0.3 and 0.7 hold gas values


def test_nodes_where_obstructions_touch_or_meet_the_mesh_edge(sim):
    times = sim.slices.get_by_id("vol").times
    m1, m2, m3 = (_mesh(sim, mesh_id) for mesh_id in ("M1", "M2", "M3"))
    nodes = {mesh.id: ~mesh.get_obstruction_mask(times)[-1] for mesh in (m1, m2, m3)}
    assert nodes["M2"][4, 2, 7]  # the face shared by 'Left' (x 2..4) and 'Right' (x 4..6)
    assert nodes["M1"][10, 8, 4] and nodes["M2"][0, 8, 4]  # 'Across' on both sides of the mesh boundary
    assert nodes["M2"][10, 4, 4]  # 'Span' at the outside of the domain
    assert nodes["M1"][3, 10, 7]  # 'Under' ends at the edge of M1, although M3 has gas beyond
    assert not nodes["M3"][3, 0, 7]  # ... which M3 writes on its own edge
    assert nodes["M1"][7, 10, 7]  # 'Outside' ends at y = 1, where the domain ends


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
    axis_index = "xyz".index(axis)
    for i, j in _shared_frames(plane, volume):
        np.testing.assert_array_equal(sub.data[i], np.take(ref.data[j], layer, axis=axis_index))
    assert sub.get_coordinates()[axis] == pytest.approx([centre])
    assert plane.get_coordinates()[axis] == pytest.approx([centre])


def test_to_global_coordinates_of_cell_centred_slices_are_the_cell_centres(sim):
    _, coords = sim.slices.get_by_id("z04_cc").to_global(return_coordinates=True)
    np.testing.assert_allclose(coords["x"], np.arange(0.05, 2.0, 0.1), atol=1e-6)
    np.testing.assert_allclose(coords["y"], np.arange(0.05, 2.0, 0.1), atol=1e-6)
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
    covered = ~np.isnan(raw)  # the global grid also spans x > 0.5, y > 1, where there is no mesh
    np.testing.assert_array_equal(np.isnan(values)[covered], (raw == PLACEHOLDER)[covered])


@pytest.mark.parametrize("volume_id, plane_id, value", [("vol", "z04", 0.4), ("vol_cc", "z04_cc", 0.45)])
@pytest.mark.parametrize("masked", [False, True])
def test_2d_cut_of_a_3d_slice_equals_the_2d_slice(sim, volume_id, plane_id, value, masked):
    volume, plane = sim.slices.get_by_id(volume_id), sim.slices.get_by_id(plane_id)
    cut, cut_coords = volume.get_2d_slice_from_3d("z", value).to_global(
        masked=masked, fill=np.nan, return_coordinates=True
    )
    reference, coords = plane.to_global(masked=masked, fill=np.nan, return_coordinates=True)
    for i, j in _shared_frames(plane, volume):
        np.testing.assert_array_equal(cut[j], reference[i])
    assert cut_coords["z"] == pytest.approx(coords["z"])


def test_2d_cut_of_a_partial_3d_slice():
    # The data of a slice defined with XB starts at its own first node, not at the mesh's
    original = settings.ENABLE_CACHING
    settings.ENABLE_CACHING = False
    try:
        sim = Simulation(str(CASES_DIR / "slcf_partial_fds611"))
    finally:
        settings.ENABLE_CACHING = original
    volume, plane = sim.slices.get_by_id("vol_part"), sim.slices.get_by_id("z_part")
    cut = volume.get_2d_slice_from_3d("z", 0.4).to_global(masked=True, fill=np.nan)
    reference = plane.to_global(masked=True, fill=np.nan)
    for i, j in _shared_frames(plane, volume):
        np.testing.assert_array_equal(cut[j], reference[i])
