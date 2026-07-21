import numpy as np
import pytest

from fdsreader import Simulation
from fdsreader.slcf.geomslice import GeomSlice


@pytest.fixture(scope="module")
def geomslice_sim():
    return Simulation("./geomslice_data_fds611")


@pytest.fixture(scope="module")
def geomslice(geomslice_sim):
    return geomslice_sim.geomslices[0]


def test_geomslice_times_non_empty(geomslice):
    assert len(geomslice.times) > 0


def test_geomslice_faces_index_range(geomslice):
    assert geomslice.faces.min() >= 0
    assert geomslice.faces.max() < geomslice.vertices.shape[0]


def test_geomslice_data_shape_matches_faces_and_times(geomslice):
    assert geomslice.data.shape == (geomslice.n_t, geomslice.faces.shape[0])


def test_geomslice_no_nan_inf(geomslice):
    data = geomslice.data
    assert not np.isnan(data).any(), "NaN values found in geomslice data"
    assert not np.isinf(data).any(), "Inf values found in geomslice data"


def test_geomslice_vertices_finite(geomslice):
    vertices = geomslice.vertices
    assert not np.isnan(vertices).any(), "NaN in geomslice vertices"
    assert not np.isinf(vertices).any(), "Inf in geomslice vertices"


class _StubSubGeomSlice:
    """Minimal stand-in for SubGeomSlice, exposing only what vertices/faces aggregation reads."""

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces


def test_geomslice_vertices_faces_handle_empty_submesh():
    """A multi-mesh geomslice where the cutting geometry doesn't intersect one mesh (0 vertices/
    faces there) must not break aggregation of the other, non-empty submeshes."""
    gs = GeomSlice.__new__(GeomSlice)
    gs._subgeomslices = {
        "mesh_empty": _StubSubGeomSlice(np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)),
        "mesh_full": _StubSubGeomSlice(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0, 1, 2]])
        ),
    }

    vertices = gs.vertices
    faces = gs.faces

    assert vertices.shape == (3, 3)
    assert faces.shape == (1, 3)
    assert faces.max() < vertices.shape[0]
