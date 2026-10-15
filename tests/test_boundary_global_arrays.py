"""Obstruction.get_global_boundary_data_arrays on bndf_global_fds611 (FDS 6.11.1; input in
tests/cases/fds_inputs/bndf_global.fds).

Two meshes: 'Fine' (x 0..1, 0.1 m cells) and 'Coarse' (x 1..2, 0.2 m cells). 'Table' spans both; its top at
z = 0.53 snaps to 0.5 in 'Fine' and to 0.6 in 'Coarse'. 'Strip' is one cell wide in x. Each face has the node-based
WALL TEMPERATURE and the cell-centred TOTAL HEAT FLUX.
"""

import numpy as np
import pytest
from fds_cases import case_dirs, case_id, load

NODES = "WALL TEMPERATURE"
CELLS = "TOTAL HEAT FLUX"


@pytest.fixture(scope="module", params=case_dirs("bndf_global_fds611", "bndf_global"), ids=case_id)
def sim(request):
    return load(request.param)


def _patch(obst, mesh_id, quantity, orientation):
    sub = next(sub for sub in obst._all_subobstructions if sub.mesh.id == mesh_id)
    return sub.get_data(quantity).data[orientation].data


@pytest.mark.parametrize(
    "orientation, nodes, cells",
    [(-1, (7, 4), (6, 3)), (1, (7, 4), (6, 3)), (-2, (2, 4), (1, 3)), (2, (2, 4), (1, 3)), (3, (2, 7), (1, 6))],
)
def test_time_comes_first_on_faces_one_cell_wide(sim, orientation, nodes, cells):
    strip = sim.obstructions.get_by_id("Strip")
    n_t = strip.n_t(count_duplicates=False)
    assert strip.get_global_boundary_data_arrays(NODES)[orientation].shape == (n_t,) + nodes
    assert strip.get_global_boundary_data_arrays(CELLS)[orientation].shape == (n_t,) + cells


@pytest.mark.parametrize("quantity", [NODES, CELLS])
def test_a_face_in_one_mesh_is_its_patch(sim, quantity):
    strip = sim.obstructions.get_by_id("Strip")
    for orientation, grid in strip.get_global_boundary_data_arrays(quantity).items():
        np.testing.assert_array_equal(grid, _patch(strip, "Fine", quantity, orientation))


@pytest.mark.parametrize("quantity", [NODES, CELLS])
def test_a_face_at_two_heights_keeps_both(sim, quantity):
    table = sim.obstructions.get_by_id("Table")
    lower, upper = table.get_global_boundary_data_arrays(quantity)[3]
    np.testing.assert_array_equal(lower, _patch(table, "Fine", quantity, 3))
    np.testing.assert_array_equal(upper, _patch(table, "Coarse", quantity, 3))


def test_coarse_values_are_repeated_on_the_finest_grid(sim):
    # The face at y = 0.2 spans x 0.6..1.4 and z 0..0.5 in 'Fine', z 0..0.6 in 'Coarse': nodes every 0.1 m
    table = sim.obstructions.get_by_id("Table")
    grid = table.get_global_boundary_data_arrays(NODES)[-2]
    fine = _patch(table, "Fine", NODES, -2)
    coarse = _patch(table, "Coarse", NODES, -2)
    assert grid.shape == (table.n_t(count_duplicates=False), 9, 7)
    np.testing.assert_array_equal(grid[:, :4, :6], fine[:, :4, :])
    np.testing.assert_array_equal(grid[:, 5:7, 2], coarse[:, :2, 1])  # x 1.1 and 1.2 take the nodes 1.0 and 1.2
    np.testing.assert_array_equal(grid[:, 6, :], coarse[:, 1, :].repeat(2, axis=1)[:, :7])  # x 1.2, z 0.0..0.6
    assert np.isnan(grid[:, :4, 6]).all()  # 'Fine' ends at z = 0.5

    grid = table.get_global_boundary_data_arrays(CELLS)[-2]
    coarse = _patch(table, "Coarse", CELLS, -2)
    assert grid.shape == (table.n_t(count_duplicates=False), 8, 6)
    np.testing.assert_array_equal(grid[:, 4:8, :], coarse.repeat(2, axis=1).repeat(2, axis=2)[:, :, :6])


def test_node_based_and_cell_centred_quantities_on_one_obstruction(sim):
    table = sim.obstructions.get_by_id("Table")
    for quantity in (NODES, CELLS):
        assert list(table.get_global_boundary_data_arrays(quantity)) == [-2, -1, 1, 2, 3]  # no face on the floor


def test_unknown_quantity(sim):
    with pytest.raises(KeyError):
        sim.obstructions.get_by_id("Table").get_global_boundary_data_arrays("VELOCITY")


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_export_of_a_face_at_two_heights(sim, tmp_path):
    from fdsreader.export import export_obst_raw

    meta = export_obst_raw(sim.obstructions.get_by_id("Table"), str(tmp_path))
    assert (tmp_path / "WALL_TEMPERATURE" / "obst-Table_quantity-WALL_TEMPERATURE.dat").stat().st_size > 0
    assert meta.endswith("obst-Table.yaml")


def test_vmin_and_vmax_of_a_face_only_some_parts_have(sim):
    # Only the part of 'Table' in 'Coarse' has a +x face, the part in 'Fine' ends at the mesh boundary
    table = sim.obstructions.get_by_id("Table")
    coarse = next(sub for sub in table._all_subobstructions if sub.mesh.id == "Coarse")
    assert table.vmax(NODES, 1) == coarse.get_data(NODES).vmax(1)
    assert table.vmin(NODES, 1) == coarse.get_data(NODES).vmin(1)


def test_coordinates_of_a_chosen_quantity(sim):
    # The -x face of 'Table' at x = 0.6 spans y 0.2..0.8: 7 nodes, or 6 cell centres
    fine = next(sub for sub in sim.obstructions.get_by_id("Table")._all_subobstructions if sub.mesh.id == "Fine")
    np.testing.assert_allclose(fine.get_coordinates(quantity=NODES)[-1]["y"], np.linspace(0.2, 0.8, 7), atol=1e-6)
    np.testing.assert_allclose(fine.get_coordinates(quantity=CELLS)[-1]["y"], np.linspace(0.25, 0.75, 6), atol=1e-6)
    assert fine.get_nearest_index("y", -1, 0.75, quantity=CELLS) == 5


def test_boundary_data_of_one_orientation(sim):
    assert list(sim.obstructions.get_by_id("Table").get_boundary_data(NODES, 1)) == ["Coarse"]


def test_parts_without_the_quantity_are_skipped(sim):
    table = load(sim.root_path).obstructions.get_by_id("Table")  # a fresh copy, as the test removes data
    fine = next(sub for sub in table._all_subobstructions if sub.mesh.id == "Fine")
    coarse = next(sub for sub in table._all_subobstructions if sub.mesh.id == "Coarse")
    bid = next(bid for bid, b in fine._boundary_data.items() if b.quantity == NODES)
    del fine._boundary_data[bid]
    assert table.vmax(NODES) == coarse.get_data(NODES).vmax()
    np.testing.assert_array_equal(table.get_global_boundary_data_arrays(NODES)[1], coarse.get_data(NODES).data[1].data)
    with pytest.raises(KeyError):
        table.vmax("VELOCITY")


def test_node_based_and_cell_centred_patches_are_not_combined(sim):
    table = load(sim.root_path).obstructions.get_by_id("Table")  # a fresh copy, as the test changes the patches
    coarse = next(sub for sub in table._all_subobstructions if sub.mesh.id == "Coarse")
    for patch in coarse.get_data(NODES).patches:
        patch.cell_centered = True
    with pytest.raises(ValueError, match="node-based"):
        table.get_global_boundary_data_arrays(NODES)
