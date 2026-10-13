import numpy as np
import pytest

from fdsreader.fds_classes.mesh import Mesh
from fdsreader.slcf.slice import Slice, SubSlice
from fdsreader.utils.extent import Extent


class NodeSlice:
    """Horizontal node-based slice at z = 1 over the given meshes, holding K = x + 10 y."""

    to_global = Slice.to_global
    subslices = Slice.subslices
    orientation, id, n_t, times, cell_centered = 3, "s", 1, np.array([0.0]), False

    def __init__(self, x_ranges):
        self._subslices = {}
        for i, (x0, x1) in enumerate(x_ranges):
            nodes = {"x": np.arange(x0, x1 + 0.25, 0.5), "y": np.array([0.0, 0.5, 1.0]), "z": np.array([0.0, 1.0, 2.0])}
            mesh = Mesh(nodes, {"x": (x0, x1), "y": (0.0, 1.0), "z": (0.0, 2.0)}, f"M{i}")
            sub = SubSlice(self, "", None, Extent(x0, x1, 0.0, 1.0, 1.0, 1.0), mesh)
            X, Y = np.meshgrid(nodes["x"], nodes["y"], indexing="ij")
            sub._data = (X + 10 * Y)[None].astype(np.float32)
            self._subslices[mesh.id] = sub
        self.extent = Extent(x_ranges[0][0], x_ranges[-1][1], 0.0, 1.0, 1.0, 1.0)


def expected(x_max):
    X, Y = np.meshgrid(np.arange(0.0, x_max + 0.25, 0.5), np.array([0.0, 0.5, 1.0]), indexing="ij")
    return X + 10 * Y


def test_to_global_keeps_upper_border_nodes_single_mesh():
    grid = np.asarray(NodeSlice([(0.0, 1.0)]).to_global())[0]
    np.testing.assert_allclose(grid, expected(1.0))


def test_to_global_keeps_upper_border_nodes_two_meshes():
    grid = np.asarray(NodeSlice([(0.0, 1.0), (1.0, 2.0)]).to_global())[0]
    np.testing.assert_allclose(grid, expected(2.0))


class BoxSlice(NodeSlice):
    """Like NodeSlice, but over meshes given as (x0, x1, y0, y1, cell size), holding K = x + 10 y."""

    def __init__(self, boxes):
        self._subslices = {}
        for i, (x0, x1, y0, y1, h) in enumerate(boxes):
            nodes = {
                "x": np.arange(x0, x1 + h / 2, h),
                "y": np.arange(y0, y1 + h / 2, h),
                "z": np.array([0.0, 1.0, 2.0]),
            }
            mesh = Mesh(nodes, {"x": (x0, x1), "y": (y0, y1), "z": (0.0, 2.0)}, f"M{i}")
            sub = SubSlice(self, "", None, Extent(x0, x1, y0, y1, 1.0, 1.0), mesh)
            X, Y = np.meshgrid(nodes["x"], nodes["y"], indexing="ij")
            sub._data = (X + 10 * Y)[None].astype(np.float32)
            self._subslices[mesh.id] = sub
        self.extent = Extent(
            min(b[0] for b in boxes),
            max(b[1] for b in boxes),
            min(b[2] for b in boxes),
            max(b[3] for b in boxes),
            1.0,
            1.0,
        )


# L-shaped domain: A at the bottom left, B above A, C right of A; nothing above C / right of B
L_SHAPE = [(0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 1.0, 2.0), (1.0, 2.0, 0.0, 1.0)]


def test_to_global_keeps_upper_face_nodes_without_neighbour():
    grid = np.asarray(BoxSlice([box + (0.25,) for box in L_SHAPE]).to_global())[0]

    X, Y = np.meshgrid(np.arange(0.0, 2.125, 0.25), np.arange(0.0, 2.125, 0.25), indexing="ij")
    in_l = (X <= 1.0) | (Y <= 1.0)
    np.testing.assert_allclose(grid[in_l], (X + 10 * Y)[in_l])
    assert np.isnan(grid[~in_l]).all()


def test_to_global_keeps_upper_face_nodes_without_neighbour_of_a_coarse_mesh():
    # C has twice the cell size; its nodes are repeated onto the finer global grid
    boxes = [L_SHAPE[0] + (0.25,), L_SHAPE[1] + (0.25,), L_SHAPE[2] + (0.5,)]
    grid = np.asarray(BoxSlice(boxes).to_global())[0]

    X, Y = np.meshgrid(np.arange(0.0, 2.125, 0.25), np.arange(0.0, 2.125, 0.25), indexing="ij")
    in_c = (X >= 1.0) & (Y <= 1.0)
    # Snap to C's nodes the way to_global repeats coarse values; the global x maximum is a node of C itself
    Xc = np.where(X == 2.0, 2.0, 1.0 + np.floor((X - 1.0) / 0.5) * 0.5)
    Yc = np.where(Y == 1.0, 1.0, np.floor(Y / 0.5) * 0.5)
    np.testing.assert_allclose(grid[in_c], (Xc + 10 * Yc)[in_c])
    assert not np.isnan(grid[(X <= 1.0) | (Y <= 1.0)]).any()


def test_to_global_keeps_a_nan_the_neighbouring_mesh_wrote_at_a_shared_face():
    slc = NodeSlice([(0.0, 1.0), (1.0, 2.0)])
    slc._subslices["M1"]._data[0, 0, 1] = np.nan  # node x = 1, y = 0.5, written by both meshes
    grid = np.asarray(slc.to_global())[0]

    assert np.isnan(grid[2, 1])
    grid[2, 1] = expected(2.0)[2, 1]
    np.testing.assert_allclose(grid, expected(2.0))


def _union_grid(boxes, h, n_dims=2):
    """Global node grid over the bounding box and a mask of the nodes inside any of the boxes."""
    lo = [min(b[2 * d] for b in boxes) for d in range(n_dims)]
    hi = [max(b[2 * d + 1] for b in boxes) for d in range(n_dims)]
    axes = [np.arange(a, m + h / 2, h) for a, m in zip(lo, hi)]
    coords = np.meshgrid(*axes, indexing="ij")
    inside = np.zeros(coords[0].shape, dtype=bool)
    for b in boxes:
        in_box = np.ones_like(inside)
        for d in range(n_dims):
            in_box &= (coords[d] >= b[2 * d] - 1e-9) & (coords[d] <= b[2 * d + 1] + 1e-9)
        inside |= in_box
    return coords, inside


SHAPES = {
    "u": [(0, 1, 0, 1), (1, 2, 0, 1), (2, 3, 0, 1), (0, 1, 1, 2), (2, 3, 1, 2)],
    "t": [(0, 1, 1, 2), (1, 2, 1, 2), (2, 3, 1, 2), (1, 2, 0, 1)],
    "stairs": [(0, 1, 0, 1), (1, 2, 0, 1), (2, 3, 0, 1), (0, 1, 1, 2), (1, 2, 1, 2), (0, 1, 2, 3)],
    "courtyard": [(x, x + 1, y, y + 1) for x in range(3) for y in range(3) if (x, y) != (1, 1)],
    "diagonal_only": [(0, 1, 0, 1), (1, 2, 1, 2)],
    "l_reversed_order": [(1, 2, 0, 1), (0, 1, 1, 2), (0, 1, 0, 1)],
    "l_off_origin": [(1, 2, 1, 2), (1, 2, 2, 3), (2, 3, 1, 2)],
}


@pytest.mark.parametrize("name", SHAPES)
def test_to_global_fills_every_node_of_non_rectangular_domains(name):
    boxes = [tuple(float(v) for v in b) for b in SHAPES[name]]
    grid = np.asarray(BoxSlice([b + (0.25,) for b in boxes]).to_global())[0]

    (X, Y), inside = _union_grid(boxes, 0.25)
    np.testing.assert_allclose(grid[inside], (X + 10 * Y)[inside])
    assert np.isnan(grid[~inside]).all()


def test_to_global_fills_every_node_of_a_courtyard_with_a_coarse_mesh():
    boxes = [tuple(float(v) for v in b) for b in SHAPES["courtyard"]]
    sizes = [0.5 if b == (2.0, 3.0, 2.0, 3.0) else 0.25 for b in boxes]
    grid = np.asarray(BoxSlice([b + (h,) for b, h in zip(boxes, sizes)]).to_global())[0]

    (X, Y), inside = _union_grid(boxes, 0.25)
    assert not np.isnan(grid[inside]).any()
    assert np.isnan(grid[~inside]).all()
    fine = inside & ~((X >= 2.0) & (Y >= 2.0))  # nodes outside the coarse mesh, which repeats its values
    np.testing.assert_allclose(grid[fine], (X + 10 * Y)[fine])


class VolumeSlice(NodeSlice):
    """Node-based 3D slice over meshes given as (x0, x1, y0, y1, z0, z1), holding K = x + 10 y + 100 z."""

    orientation = 0

    def __init__(self, boxes, h=0.25):
        self._subslices = {}
        for i, b in enumerate(boxes):
            nodes = {dim: np.arange(b[2 * d], b[2 * d + 1] + h / 2, h) for d, dim in enumerate("xyz")}
            mesh = Mesh(nodes, {dim: (b[2 * d], b[2 * d + 1]) for d, dim in enumerate("xyz")}, f"M{i}")
            sub = SubSlice(self, "", None, Extent(*b), mesh)
            X, Y, Z = np.meshgrid(nodes["x"], nodes["y"], nodes["z"], indexing="ij")
            sub._data = (X + 10 * Y + 100 * Z)[None].astype(np.float32)
            self._subslices[mesh.id] = sub
        self.extent = Extent(*[f(b[k] for b in boxes) for k, f in enumerate([min, max] * 3)])


def test_to_global_fills_every_node_of_a_3d_slice_with_a_step_in_height():
    # Two meshes side by side, a third one on top of only the first: a step in z
    boxes = [(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (1.0, 2.0, 0.0, 1.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0, 1.0, 2.0)]
    grid = np.asarray(VolumeSlice(boxes).to_global())[0]

    (X, Y, Z), inside = _union_grid(boxes, 0.25, n_dims=3)
    np.testing.assert_allclose(grid[inside], (X + 10 * Y + 100 * Z)[inside])
    assert np.isnan(grid[~inside]).all()


class FakeObstruction:
    """Obstruction visible at all times, covering the given node index ranges of one mesh."""

    def __init__(self, mesh, x, y, z):
        self._sub = {mesh: self}
        self.bound_indices = {"x": x, "y": y, "z": z}

    def __getitem__(self, mesh):
        return self._sub[mesh]

    def get_visible_times(self, times):
        return times


def _add_obstruction(slc, mesh_id, x, y):
    """Puts an obstruction on a node of the given mesh at the slice height z = 1 (node index 1)."""
    mesh = slc._subslices[mesh_id].mesh
    mesh.obstructions.append(FakeObstruction(mesh, x, y, (1, 2)))


def test_to_global_masks_a_recovered_upper_face_node():
    slc = BoxSlice([box + (0.25,) for box in L_SHAPE])
    _add_obstruction(slc, "M1", x=(4, 5), y=(1, 2))  # B's right face at x = 1, y = 1.25: no neighbour supplies it
    grid = np.asarray(slc.to_global(masked=True, fill=np.nan))[0]

    (X, Y), inside = _union_grid(L_SHAPE, 0.25)
    obstructed = np.isclose(X, 1.0) & np.isclose(Y, 1.25)
    assert np.isnan(grid[obstructed]).all()
    np.testing.assert_allclose(grid[inside & ~obstructed], (X + 10 * Y)[inside & ~obstructed])


@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (2, 1, 0)])
def test_to_global_masks_a_recovered_node_if_any_supplying_mesh_masks_it(order):
    # Corner x = 1, y = 1 lies on upper faces of A, B and C, so pass 2 supplies it; only B has the obstruction
    boxes = [L_SHAPE[i] for i in order]
    slc = BoxSlice([box + (0.25,) for box in boxes])
    _add_obstruction(slc, f"M{order.index(1)}", x=(4, 5), y=(0, 1))
    grid = np.asarray(slc.to_global(masked=True, fill=np.nan))[0]

    (X, Y), inside = _union_grid(L_SHAPE, 0.25)
    corner = np.isclose(X, 1.0) & np.isclose(Y, 1.0)
    assert np.isnan(grid[corner]).all()
    np.testing.assert_allclose(grid[inside & ~corner], (X + 10 * Y)[inside & ~corner])


class CellSlice(NodeSlice):
    """Horizontal cell-centred slice over meshes given as (x0, x1, cell size) side by side in x, y from 0 to 1,
    holding K = x + 10 y at the cell centres."""

    cell_centered = True

    def __init__(self, x_meshes):
        self._subslices = {}
        for i, (x0, x1, h) in enumerate(x_meshes):
            nodes = {
                "x": np.arange(x0, x1 + h / 2, h),
                "y": np.arange(0.0, 1.0 + h / 2, h),
                "z": np.array([0.0, 1.0, 2.0]),
            }
            mesh = Mesh(nodes, {"x": (x0, x1), "y": (0.0, 1.0), "z": (0.0, 2.0)}, f"M{i}")
            sub = SubSlice(self, "", None, Extent(x0, x1, 0.0, 1.0, 1.0, 1.0), mesh)
            Xc, Yc = np.meshgrid(nodes["x"][:-1] + h / 2, nodes["y"][:-1] + h / 2, indexing="ij")
            sub._data = (Xc + 10 * Yc)[None].astype(np.float32)
            self._subslices[mesh.id] = sub
        self.extent = Extent(x_meshes[0][0], x_meshes[-1][1], 0.0, 1.0, 1.0, 1.0)


def test_to_global_places_cell_centred_meshes_side_by_side():
    grid = np.asarray(CellSlice([(0.0, 1.0, 0.25), (1.0, 2.0, 0.25)]).to_global())[0]

    Xc, Yc = np.meshgrid(np.arange(0.125, 2.0, 0.25), np.arange(0.125, 1.0, 0.25), indexing="ij")
    np.testing.assert_allclose(grid, Xc + 10 * Yc)


def test_to_global_repeats_values_of_a_coarse_cell_centred_mesh():
    grid = np.asarray(CellSlice([(0.0, 1.0, 0.25), (1.0, 2.0, 0.5)]).to_global())[0]

    Xc, Yc = np.meshgrid(np.arange(0.125, 2.0, 0.25), np.arange(0.125, 1.0, 0.25), indexing="ij")
    # Each coarse cell covers 2 x 2 fine cells, which all take its centre value
    coarse_x = np.where(Xc > 1.0, 1.0 + (np.floor((Xc - 1.0) / 0.5) + 0.5) * 0.5, Xc)
    coarse_y = np.where(Xc > 1.0, (np.floor(Yc / 0.5) + 0.5) * 0.5, Yc)
    np.testing.assert_allclose(grid, coarse_x + 10 * coarse_y)


def test_to_global_masks_a_recovered_face_node_of_a_coarse_mesh():
    # C has twice the cell size; an obstruction on its coarse node x = 1.5, y = 1 (top face, no neighbour above)
    # covers the fine nodes x = 1.5 and x = 1.75 once C's values are repeated onto the global grid
    slc = BoxSlice([L_SHAPE[0] + (0.25,), L_SHAPE[1] + (0.25,), L_SHAPE[2] + (0.5,)])
    _add_obstruction(slc, "M2", x=(1, 2), y=(2, 3))
    grid = np.asarray(slc.to_global(masked=True, fill=np.nan))[0]

    (X, Y), inside = _union_grid(L_SHAPE, 0.25)
    obstructed = np.isclose(Y, 1.0) & (np.isclose(X, 1.5) | np.isclose(X, 1.75))
    assert np.isnan(grid[obstructed]).all()
    assert not np.isnan(grid[inside & ~obstructed]).any()
    top_of_c = np.isclose(Y, 1.0) & (X > 1.0) & ~obstructed
    Xc = np.where(X == 2.0, 2.0, 1.0 + np.floor((X - 1.0) / 0.5) * 0.5)
    np.testing.assert_allclose(grid[top_of_c], (Xc + 10 * Y)[top_of_c])


class TimedFakeObstruction(FakeObstruction):
    """FakeObstruction that is only visible at the given times."""

    def __init__(self, mesh, x, y, z, visible):
        super().__init__(mesh, x, y, z)
        self._visible = visible

    def get_visible_times(self, times):
        return [t for t in times if t in self._visible]


class TwoStepVolumeSlice(VolumeSlice):
    """VolumeSlice with two timesteps (t = 0 and t = 1) holding the same data."""

    n_t, times = 2, np.array([0.0, 1.0])

    def __init__(self, boxes, h=0.25):
        super().__init__(boxes, h)
        for sub in self._subslices.values():
            sub._data = np.concatenate([sub._data, sub._data])


class TwoStepCellVolumeSlice(TwoStepVolumeSlice):
    """Cell-centred variant: holds K = x + 10 y + 100 z at the cell centres of each mesh."""

    cell_centered = True

    def __init__(self, boxes, h=0.25):
        super().__init__(boxes, h)
        for sub in self._subslices.values():
            centres = [sub.mesh.coordinates[dim][:-1] + h / 2 for dim in "xyz"]
            X, Y, Z = np.meshgrid(*centres, indexing="ij")
            sub._data = np.stack([X + 10 * Y + 100 * Z] * 2).astype(np.float32)


def _one_mesh_with_obstruction_at_second_timestep(slice_class):
    """One unit-cube mesh (0.25 m) with an obstruction between nodes 1 and 3 on every axis, visible at t = 1 only."""
    slc = slice_class([(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)])
    sub = slc._subslices["M0"]
    sub.mesh.obstructions.append(TimedFakeObstruction(sub.mesh, (1, 3), (1, 3), (1, 3), visible=[1.0]))
    return slc, sub


# These tests check the time handling of 3D slice masks (#125) only. Which nodes or cells on an obstruction's faces
# are masked is left open: they only assert points strictly inside the obstruction and points beyond its lower faces.


def test_obstruction_mask_of_a_3d_slice_keeps_every_timestep():
    slc, sub = _one_mesh_with_obstruction_at_second_timestep(TwoStepVolumeSlice)

    mask = sub.mesh.get_obstruction_mask_slice(sub)

    assert mask.shape == (2, 5, 5, 5)
    assert mask[0].all()  # the obstruction is hidden at t = 0
    assert not mask[1, 2, 2, 2]  # node strictly inside the obstruction
    assert mask[1, 0].all() and mask[1, :, 0].all() and mask[1, :, :, 0].all()  # nodes beyond its lower faces


@pytest.mark.parametrize("slice_class", [TwoStepVolumeSlice, TwoStepCellVolumeSlice])
def test_to_global_masks_a_3d_slice_per_timestep(slice_class):
    slc, sub = _one_mesh_with_obstruction_at_second_timestep(slice_class)

    grid = np.asarray(slc.to_global(masked=True, fill=np.nan))

    points = np.arange(0.0, 1.125, 0.25) if not slc.cell_centered else np.arange(0.125, 1.0, 0.25)
    X, Y, Z = np.meshgrid(points, points, points, indexing="ij")
    expected_values = X + 10 * Y + 100 * Z
    assert grid.shape == (2,) + X.shape
    np.testing.assert_allclose(grid[0], expected_values)  # nothing is masked at t = 0

    inside = (X > 0.25) & (X < 0.75) & (Y > 0.25) & (Y < 0.75) & (Z > 0.25) & (Z < 0.75)
    beyond_lower_faces = (X < 0.25) | (Y < 0.25) | (Z < 0.25)
    assert inside.any() and np.isnan(grid[1][inside]).all()
    np.testing.assert_allclose(grid[1][beyond_lower_faces], expected_values[beyond_lower_faces])
