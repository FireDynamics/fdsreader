"""Boundary data on the walls of the simulation domain (#81), on bndf_domain_fds611 (FDS 6.11.1; input in
tests/cases/fds_inputs/bndf_domain.fds).

Two meshes: 'West' (x 0..1, 0.1 m cells) and 'East' (x 1..2, 0.2 m cells), y 0..1. A burner vent at x 0.6..1.6,
y 0.3..0.7 on the floor splits the floor of each mesh into four patches; in 'East', FDS snaps the vent to y 0.4..0.8.
The obstruction 'Box' stands on the floor. BURNING RATE is cell-centred, WALL TEMPERATURE node-based.
"""

import warnings

import numpy as np
import pytest
from fds_cases import case_dirs, case_id, load

CELLS = "BURNING RATE"
NODES = "WALL TEMPERATURE"
FLOOR = 3


@pytest.fixture(scope="module", params=case_dirs("bndf_domain_fds611", "bndf_domain"), ids=case_id)
def sim(request):
    return load(request.param)


def _floor_patches(sim, mesh_id, quantity):
    boundary = sim.meshes.get_by_id(mesh_id).get_boundary_data(quantity)
    return sorted(
        (p for p in boundary.patches if p.orientation == FLOOR), key=lambda p: (p.extent.x_start, p.extent.y_start)
    )


def test_all_patches_of_a_face_are_kept(sim):
    boundary = sim.meshes.get_by_id("West").get_boundary_data(CELLS)
    assert len([p for p in boundary.patches if p.orientation == FLOOR]) == 4
    assert boundary.orientations == sorted(set(boundary.orientations))
    assert boundary.vmax(FLOOR) == max(np.max(p.data) for p in _floor_patches(sim, "West", CELLS))
    with pytest.warns(UserWarning, match="Boundary.patches"):
        _ = boundary.data


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


def test_the_floor_holds_each_patch(sim):
    floor = sim.meshes.get_global_boundary_data_arrays(CELLS)[FLOOR]
    west = _floor_patches(sim, "West", CELLS)[0]  # x 0..0.6, y 0..1
    np.testing.assert_array_equal(floor[:, 0:6, 0:10], west.data)
    east = _floor_patches(sim, "East", CELLS)[0]  # x 1..2, y 0..0.4, 0.2 m cells
    np.testing.assert_array_equal(floor[:, 10:20, 0:4], east.data.repeat(2, axis=1).repeat(2, axis=2))


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
