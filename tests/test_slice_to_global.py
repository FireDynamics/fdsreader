import numpy as np

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
            nodes = {"x": np.arange(x0, x1 + 0.25, 0.5), "y": np.array([0.0, 0.5, 1.0]),
                     "z": np.array([0.0, 1.0, 2.0])}
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
