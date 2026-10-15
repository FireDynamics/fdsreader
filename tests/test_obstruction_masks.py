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

import numpy as np
import pytest
from fds_cases import CASES_DIR, case_dirs, case_id, load

SLICE_IDS = ["vol", "vol_cc", "z04", "z04_cc", "z045_cc", "z02", "z02_cc", "z07", "x05_cc", "y08"]
PLACEHOLDER = 20.0


@pytest.fixture(scope="module", params=case_dirs("slcf_masks_fds611", "slcf_masks"), ids=case_id)
def sim(request):
    return load(request.param)


def _mesh(sim, mesh_id):
    return next(mesh for mesh in sim.meshes if mesh.id == mesh_id)


def _shared_frames(a, b):
    """Index pairs of the frames two slices have in common."""
    pairs = [(i, int(np.argmin(np.abs(b.times - t)))) for i, t in enumerate(a.times)]
    pairs = [(i, j) for i, j in pairs if np.isclose(a.times[i], b.times[j], atol=1e-3)]
    assert len(pairs) > 1
    return pairs


@pytest.mark.parametrize("slice_id", SLICE_IDS)
def test_placeholder_mask_blocks_exactly_the_placeholder_values(sim, slice_id):
    for sub in sim.slices.get_by_id(slice_id).subslices:
        data = sub.data[-1]
        if sub.orientation:
            data = np.expand_dims(data, sub.orientation - 1)
        blocked = ~sub.mesh.get_obstruction_mask_slice(sub, mode="placeholders")[-1]
        np.testing.assert_array_equal(blocked, data == PLACEHOLDER, err_msg=f"{slice_id} in {sub.mesh.id}")


def test_cells_and_nodes_of_an_obstruction(sim):
    # 'Box' spans the nodes 3..7, 3..6, 2..6 of M1: its cells are 3..6, 3..5, 2..5, its inner nodes 4..6, 4..5, 3..5
    times = sim.slices.get_by_id("vol").times
    m1 = _mesh(sim, "M1")
    for mode in ("obstructions", "placeholders"):
        cells = ~m1.get_obstruction_mask(times, cell_centered=True, mode=mode)[-1]
        assert cells[3:7, 3:6, 2:6].all() and not cells[7, 4, 4] and not cells[2, 4, 4]
    nodes = ~m1.get_obstruction_mask(times, mode="placeholders")[-1]
    assert nodes[4:7, 4:6, 3:6].all()
    assert not nodes[3, 4, 4] and not nodes[7, 4, 4]  # the faces of 'Box' at x = 0.3 and 0.7 hold gas values
    nodes = ~m1.get_obstruction_mask(times)[-1]
    assert nodes[3:8, 3:7, 2:7].all() and not nodes[2, 4, 4] and not nodes[8, 4, 4]  # with its faces


def test_placeholder_nodes_where_obstructions_touch_or_meet_the_mesh_edge(sim):
    times = sim.slices.get_by_id("vol").times
    m1, m2, m3 = (_mesh(sim, mesh_id) for mesh_id in ("M1", "M2", "M3"))
    nodes = {mesh.id: ~mesh.get_obstruction_mask(times, mode="placeholders")[-1] for mesh in (m1, m2, m3)}
    assert nodes["M2"][4, 2, 7]  # the face shared by 'Left' (x 2..4) and 'Right' (x 4..6)
    assert nodes["M1"][10, 8, 4] and nodes["M2"][0, 8, 4]  # 'Across' on both sides of the mesh boundary
    assert nodes["M2"][10, 4, 4]  # 'Span' at the outside of the domain
    assert nodes["M1"][3, 10, 7]  # 'Under' ends at the edge of M1, although M3 has gas beyond
    assert not nodes["M3"][3, 0, 7]  # ... which M3 writes on its own edge
    assert nodes["M1"][7, 10, 7]  # 'Outside' ends at y = 1, where the domain ends


def _inside_obstructions(sub, cell_centered):
    """Expected obstruction mask of a subslice from the coordinates of the obstructions in the .smv file: the cells
    whose centre lies inside an obstruction, or the nodes inside or on it."""
    coords = sub.get_coordinates()
    grids = np.meshgrid(*(coords[dim] for dim in "xyz"), indexing="ij")
    spacing = {dim: np.diff(sub.mesh.coordinates[dim]).min() for dim in "xyz"}
    blocked = np.zeros(grids[0].shape, dtype=bool)
    for obst in sub.mesh.obstructions:
        for subobst in obst.get_subobstructions(sub.mesh):
            inside = np.ones(blocked.shape, dtype=bool)
            for grid, dim in zip(grids, "xyz"):
                low, high = subobst.extent[dim]
                tol = 1e-4 * spacing[dim]
                inside &= (grid > low - tol) & (grid < high + tol)
            blocked |= inside
    return blocked


@pytest.mark.parametrize("slice_id", SLICE_IDS)
def test_obstruction_mask_covers_the_obstructions_and_their_faces(sim, slice_id):
    for sub in sim.slices.get_by_id(slice_id).subslices:
        blocked = ~sub.mesh.get_obstruction_mask_slice(sub)[-1]
        expected = _inside_obstructions(sub, sub.cell_centered)
        np.testing.assert_array_equal(blocked, expected, err_msg=f"{slice_id} in {sub.mesh.id}")


@pytest.mark.parametrize("slice_id", SLICE_IDS)
def test_obstruction_mask_contains_every_placeholder(sim, slice_id):
    # Everything FDS did not compute lies inside an obstruction or on its faces
    for sub in sim.slices.get_by_id(slice_id).subslices:
        obstructions = ~sub.mesh.get_obstruction_mask_slice(sub)[-1]
        placeholders = ~sub.mesh.get_obstruction_mask_slice(sub, mode="placeholders")[-1]
        assert not (placeholders & ~obstructions).any()


# Smokeview 6.11.2 on this case, read off by hand (1 = blanked) for the areas of 0.1 m between the nodes, x 0..2 left to
# right, the other axis from 1.0 (top) down to 0.0: the slices PBZ=0.7 seen from above, PBY=0.8 seen from the front and
# PBZ=0.4 with CELL_CENTERED=T seen from above, at t = 10 s with the obstacles hidden
SMOKEVIEW = {
    "z07": """
        00110011000000000000
        01111011000000001110
        01001000000000001110
        00000000000000001110
        00000000000000000000
        00000000000000000000
        00000000000000000000
        00000000000011110000
        00000000110011110000
        00000000000000000000
    """,
    "y08": """
        00000000000000000000
        00000000000000001110
        01111011000000001010
        01111011000000001110
        00000000111110000000
        00000000111110000000
        00000000111110000000
        00000000111110000000
        00000000000000000000
        00000000000000000000
    """,
    # cell-centred PBZ=0.4 (the cells z 0.4..0.5) seen from above, per cell
    "z04_cc": """
        00000000000000000000
        00000000111110000000
        00000000111110000000
        00000000000000000000
        00011110000000000111
        00011110000000000111
        00011110000000000111
        00000000000000000000
        00000000000000000000
        00000000000000000000
    """,
}


@pytest.mark.parametrize("slice_id, axis, node", [("z07", 2, 7), ("y08", 1, 8)])
def test_obstruction_mask_matches_smokeview(sim, slice_id, axis, node):
    # Smokeview blanks an area of the plane if a cell next to it, on either side, is solid
    smokeview = np.array([[int(c) for c in row] for row in SMOKEVIEW[slice_id].split()])[::-1].T
    times = sim.slices.get_by_id(slice_id).times
    blanked = np.zeros((20, 10), dtype=int)
    for mesh, offset in ((_mesh(sim, "M1"), 0), (_mesh(sim, "M2"), 10)):
        cells = ~mesh.get_obstruction_mask(times, cell_centered=True)[-1]
        sides = np.take(cells, node - 1, axis=axis) | np.take(cells, node, axis=axis)
        blanked[offset : offset + 10] = sides
    np.testing.assert_array_equal(blanked, smokeview)


def test_masked_cell_centred_slice_matches_smokeview(sim):
    # Through the whole path: the cell layer of the plane, its mask and to_global
    smokeview = np.array([[int(c) for c in row] for row in SMOKEVIEW["z04_cc"].split()])[::-1].T
    values, coords = sim.slices.get_by_id("z04_cc").to_global(masked=True, fill=np.nan, return_coordinates=True)
    centres = np.arange(0.05, 1.0, 0.1)
    ix = np.searchsorted(coords["x"], np.arange(0.05, 2.0, 0.1) - 1e-6)
    iy = np.searchsorted(coords["y"], centres - 1e-6)
    np.testing.assert_array_equal(np.isnan(values[-1][np.ix_(ix, iy)]), smokeview.astype(bool))


def test_unknown_mask_mode_is_rejected(sim):
    with pytest.raises(ValueError, match="mode"):
        sim.meshes[0].get_obstruction_mask([0.0], mode="solid")


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
def test_to_global_with_placeholder_mask_hides_exactly_the_placeholder_values(sim, slice_id):
    slc = sim.slices.get_by_id(slice_id)
    values = slc.to_global(masked=True, fill=np.nan, mask="placeholders")[-1]
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
    sim = load(CASES_DIR / "slcf_partial_fds611")
    volume, plane = sim.slices.get_by_id("vol_part"), sim.slices.get_by_id("z_part")
    cut = volume.get_2d_slice_from_3d("z", 0.4).to_global(masked=True, fill=np.nan)
    reference = plane.to_global(masked=True, fill=np.nan)
    for i, j in _shared_frames(plane, volume):
        np.testing.assert_array_equal(cut[j], reference[i])


def test_2d_cut_of_a_cell_centred_slice_on_a_node_takes_the_cell_above(sim):
    # Like FDS for &SLCF PBZ=0.4, CELL_CENTERED=T: the cell from 0.4 to 0.5
    volume, plane = sim.slices.get_by_id("vol_cc"), sim.slices.get_by_id("z04_cc")
    cut_slice = volume.get_2d_slice_from_3d("z", 0.4)
    cut, coords = cut_slice.to_global(return_coordinates=True)
    reference = plane.to_global()
    for i, j in _shared_frames(plane, volume):
        np.testing.assert_array_equal(cut[j], reference[i])
    assert coords["z"] == pytest.approx([0.45])
    for sub in cut_slice.subslices:
        assert sub.dimension.z == 1
        assert sub.data.shape[1:] == sub.dimension.shape(cell_centered=True)[:2]


def test_time_indices_of_visible_times():
    from fdsreader.fds_classes.mesh import Mesh

    times = np.array([0.0, 1.0, 2.0, 3.0])
    assert list(Mesh._time_indices(times, [0.0, 2.0, 3.0])) == [0, 2, 3]
    assert list(Mesh._time_indices(times, [])) == []
    assert list(Mesh._time_indices(np.array([5.0]), [5.0])) == [0]
