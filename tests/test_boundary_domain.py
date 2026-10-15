"""Boundary data on the walls of the simulation domain (#81), on bndf_domain_fds611 (FDS 6.11.1; input in
tests/cases/fds_inputs/bndf_domain.fds).

Two meshes: 'West' (x 0..1, 0.1 m cells) and 'East' (x 1..2, 0.2 m cells), y 0..1. A burner vent at x 0.6..1.6,
y 0.3..0.7 on the floor splits the floor of each mesh into four patches; in 'East', FDS snaps the vent to y 0.4..0.8.
The obstruction 'Box' stands on the floor. BURNING RATE is cell-centred, WALL TEMPERATURE node-based.
"""

import warnings

import numpy as np
import pytest
from fds_cases import CASES_DIR, case_dirs, case_id, load

CELLS = "BURNING RATE"
NODES = "WALL TEMPERATURE"
FLOOR = 3


@pytest.fixture(scope="module", params=case_dirs("bndf_domain_fds611", "bndf_domain"), ids=case_id)
def sim(request):
    return load(request.param)


def _floor_patches(sim, mesh_id, quantity):
    boundary = sim.meshes.get_by_id(mesh_id).get_boundary_data(quantity)
    return [p for p in boundary.patches if p.orientation == FLOOR]


# The tests also run on the output of new FDS builds, so they do not rely on how FDS splits the floor into patches
# x = 0 and the top are open, x = 1 lies between the meshes, so 'West' has no faces facing +-x and 'East' only -x
@pytest.mark.parametrize(
    "mesh_id, n_cells, orientations", [("West", 10 * 10, [-2, 2, 3]), ("East", 5 * 5, [-2, -1, 2, 3])]
)
def test_all_patches_of_a_face_are_kept(sim, mesh_id, n_cells, orientations):
    patches = _floor_patches(sim, mesh_id, CELLS)
    assert len(patches) > 1  # the vent splits the floor
    assert sum(p.data[0].size for p in patches) == n_cells  # together they cover the floor of the mesh
    boundary = sim.meshes.get_by_id(mesh_id).get_boundary_data(CELLS)
    assert boundary.orientations == orientations
    assert boundary.vmax(FLOOR) == max(np.max(p.data) for p in patches)


def test_boundary_data_warns_when_it_drops_patches(sim):
    boundary = sim.meshes.get_by_id("West").get_boundary_data(CELLS)
    with pytest.warns(UserWarning, match="Boundary.patches") as record:
        _ = boundary.data
    warning = next(w for w in record if "Boundary.patches" in str(w.message))
    assert warning.filename == __file__  # points at the caller, not into fdsreader


def test_vmin_and_vmax_of_a_missing_face_are_nan(sim):
    boundary = sim.meshes.get_by_id("West").get_boundary_data(CELLS)
    assert np.isnan(boundary.vmin(1)) and np.isnan(boundary.vmax(1))


def test_obstruction_boundary_data_does_not_warn(sim):
    box = sim.obstructions.get_by_id("Box")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for sub in box._all_subobstructions:
            _ = sub.get_data(CELLS).data


def test_the_floor_over_both_meshes(sim):
    floor = sim.meshes.get_global_boundary_data_arrays(CELLS)[FLOOR]
    assert floor.shape == (sim.meshes[0].get_boundary_data(CELLS).n_t(count_duplicates=False), 20, 10)
    assert not np.isnan(floor).any()

    burning = np.zeros((20, 10), dtype=bool)
    burning[6:10, 3:7] = True  # 'West': x 0.6..1.0, y 0.3..0.7
    burning[10:16, 4:8] = True  # 'East': x 1.0..1.6, y 0.4..0.8
    np.testing.assert_array_equal(floor[-1] > 0, burning)


@pytest.mark.parametrize("mesh_id, repeat", [("West", 1), ("East", 2)])  # 'East' has cells twice as large
def test_the_floor_holds_each_patch(sim, mesh_id, repeat):
    floor = sim.meshes.get_global_boundary_data_arrays(CELLS)[FLOOR]
    for patch in _floor_patches(sim, mesh_id, CELLS):
        x = slice(round(patch.extent.x_start / 0.1), round(patch.extent.x_end / 0.1))
        y = slice(round(patch.extent.y_start / 0.1), round(patch.extent.y_end / 0.1))
        np.testing.assert_array_equal(floor[:, x, y], patch.data.repeat(repeat, axis=1).repeat(repeat, axis=2))


def test_node_based_floor(sim):
    floor = sim.meshes.get_global_boundary_data_arrays(NODES)[FLOOR]
    assert floor.shape[1:] == (21, 11)
    assert not np.isnan(floor).any()


def test_obstruction_faces_are_not_included(sim):
    # The top of 'Box' at z = 0.2 also faces +z, but belongs to the obstruction
    assert isinstance(sim.meshes.get_global_boundary_data_arrays(CELLS)[FLOOR], np.ndarray)


def test_unknown_quantity(sim):
    with pytest.raises(KeyError):
        sim.meshes.get_global_boundary_data_arrays("VELOCITY")


def test_a_case_without_boundary_data_gives_no_arrays():
    sim = load(CASES_DIR / "slcf_partial_fds611")
    assert not any(mesh.has_boundary_data for mesh in sim.meshes)
    assert sim.meshes.get_global_boundary_data_arrays("WALL TEMPERATURE") == {}
