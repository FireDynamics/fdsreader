import numpy as np
import pytest

from fdsreader import Simulation


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
