"""Plot3D.to_global() and Smoke3D.to_global() must place every node FDS wrote at its own coordinate,
also when a coarser mesh lies on the upper border of the simulation space (GitHub issue #99)."""

import numpy as np
import pytest

from fdsreader.fds_classes import Mesh
from fdsreader.pl3d import Plot3D
from fdsreader.smoke3d import Smoke3D
from fdsreader.utils import Quantity


def _make_mesh(mesh_id, x0, x1, y0, y1, z0, z1, step):
    coordinates = {
        "x": np.arange(x0, x1 + step / 2, step),
        "y": np.arange(y0, y1 + step / 2, step),
        "z": np.arange(z0, z1 + step / 2, step),
    }
    return Mesh(coordinates, {"x": (x0, x1), "y": (y0, y1), "z": (z0, z1)}, mesh_id)


def _field(mesh):
    """Node data K = x + 10 y + 100 z for a single time step, as FDS writes it, (t, x, y, z)."""
    X, Y, Z = np.meshgrid(mesh.coordinates["x"], mesh.coordinates["y"], mesh.coordinates["z"], indexing="ij")
    return (X + 10 * Y + 100 * Z)[None].astype(np.float32)


def _meshes(mixed):
    # A fine mesh below a mesh that is twice as coarse (if mixed) and touches the upper border on every axis.
    return [
        _make_mesh("A", 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.25),
        _make_mesh("B", 0.0, 1.0, 0.0, 1.0, 0.5, 1.0, 0.5 if mixed else 0.25),
    ]


def _plot3d(meshes):
    pl3d = Plot3D("")
    for mesh in meshes:
        pl3d._add_subplot(f"{mesh.id}.q", 0.0, Quantity("TEST", "test", "-"), 0, mesh)
        pl3d[mesh]._data = _field(mesh)
    return pl3d


def _smoke3d(meshes):
    smoke = Smoke3D("", np.array([0.0]), Quantity("TEST", "test", "-"))
    for mesh in meshes:
        smoke._add_subsmoke(f"{mesh.id}.s3d", mesh, upper_bounds=np.array([1.0]))._data = _field(mesh)
    return smoke


@pytest.mark.parametrize("build", [_plot3d, _smoke3d])
@pytest.mark.parametrize("masked", [False, True])
def test_to_global_keeps_nodes_in_place_uniform(build, masked):
    grid, coordinates = build(_meshes(mixed=False)).to_global(masked=masked, return_coordinates=True)

    X, Y, Z = np.meshgrid(coordinates["x"], coordinates["y"], coordinates["z"], indexing="ij")
    np.testing.assert_allclose(grid[0], X + 10 * Y + 100 * Z, atol=1e-5)


@pytest.mark.parametrize("build", [_plot3d, _smoke3d])
@pytest.mark.parametrize("masked", [False, True])
def test_to_global_keeps_nodes_in_place_mixed_resolution(build, masked):
    meshes = _meshes(mixed=True)
    grid, coordinates = build(meshes).to_global(masked=masked, return_coordinates=True)

    assert grid.shape == (1, 5, 5, 5)
    # Every node of the coarse mesh B, which owns z >= 0.5, sits at its own coordinate on the finer grid.
    for x in meshes[1].coordinates["x"]:
        for y in meshes[1].coordinates["y"]:
            for z in meshes[1].coordinates["z"]:
                i, j, k = (int(np.argmin(np.abs(coordinates[d] - v))) for d, v in zip("xyz", (x, y, z)))
                assert grid[0, i, j, k] == pytest.approx(x + 10 * y + 100 * z)
    # The fine mesh A owns z < 0.5.
    X, Y, Z = np.meshgrid(coordinates["x"], coordinates["y"], coordinates["z"][:2], indexing="ij")
    np.testing.assert_allclose(grid[0, :, :, :2], X + 10 * Y + 100 * Z, atol=1e-5)
