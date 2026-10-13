import numpy as np

from fdsreader.fds_classes.mesh import Mesh
from fdsreader.slcf.slice import Slice, SubSlice
from fdsreader.utils.extent import Extent

# Regression tests for #89. The domain lies below zero (z = -6.0 .. -0.4, 56 cells of 0.1 m). FDS stores a
# cell-centred slice at the top of a mesh (PBZ=-0.45, written as SLCC ... 56 56) with the extent of the mesh's top
# node, z = -0.4 (as float32), which lies above the topmost cell centre, so searchsorted returns len(centres). The old
# guard in Mesh.coordinate_to_index (co > 0 instead of idx > 0) only stepped back for positive coordinates, the
# obstruction mask came out empty and to_global(masked=True) failed with "cannot reshape array of size 0".

Z_NODES = np.linspace(-6.0, -0.4, 57).astype(np.float32)  # coordinates are read as float32
Z_SLICE = float(np.float32(-0.4))  # extent as read from the .smv


class CeilingSlice:
    """Cell-centred horizontal slice at the top of meshes below zero, side by side in x, holding K = x + 10 y."""

    to_global = Slice.to_global
    subslices = Slice.subslices
    orientation, id, n_t, times, cell_centered = 3, "s", 1, np.array([0.0]), True

    def __init__(self, n_meshes):
        self._subslices = {}
        for i in range(n_meshes):
            x0 = float(i)
            nodes = {dim: np.linspace(lo, lo + 1.0, 11, dtype=np.float32) for dim, lo in (("x", x0), ("y", 0.0))}
            nodes["z"] = Z_NODES
            mesh = Mesh(nodes, {"x": (x0, x0 + 1.0), "y": (0.0, 1.0), "z": (-6.0, -0.4)}, f"M{i}")
            sub = SubSlice(self, "", None, Extent(x0, x0 + 1.0, 0.0, 1.0, Z_SLICE, Z_SLICE), mesh)
            Xc, Yc = np.meshgrid(nodes["x"][:-1] + 0.05, nodes["y"][:-1] + 0.05, indexing="ij")
            sub._data = (Xc + 10 * Yc)[None].astype(np.float32)
            self._subslices[mesh.id] = sub
        self.extent = Extent(0.0, float(n_meshes), 0.0, 1.0, Z_SLICE, Z_SLICE)


def test_coordinate_to_index_resolves_the_topmost_cell_centre_below_zero():
    centres = Z_NODES[:-1] + np.diff(Z_NODES) / 2
    # Make sure the case of #89 actually occurs: the slice lies above the last cell centre
    assert np.searchsorted(centres, Z_SLICE) == len(centres)

    mesh = CeilingSlice(1)._subslices["M0"].mesh
    assert mesh.coordinate_to_index((Z_SLICE,), dimension=("z",), cell_centered=True) == (55,)


def test_obstruction_mask_of_a_slice_at_the_top_of_a_mesh_below_zero_is_not_empty():
    sub = CeilingSlice(1)._subslices["M0"]
    assert sub.mesh.get_obstruction_mask_slice(sub).shape == (1, 10, 10, 1)


def test_to_global_masks_a_slice_at_the_top_of_meshes_below_zero():
    grid = np.asarray(CeilingSlice(2).to_global(masked=True, fill=np.nan))

    Xc, Yc = np.meshgrid(np.arange(0.05, 2.0, 0.1), np.arange(0.05, 1.0, 0.1), indexing="ij")
    np.testing.assert_allclose(grid[0], Xc + 10 * Yc, rtol=1e-6)
