import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def geom_sim():
    return Simulation("./geom_data")


@pytest.fixture(scope="module")
def geom(geom_sim):
    return geom_sim.geom_data.filter_by_quantity("Radiative Heat Flux")[0]


def test_geom(geom):
    assert len(geom.faces) == len(geom.data[-1]) == 19816
    assert len(geom.vertices) == 40624


def test_geom_data_non_empty(geom):
    assert len(geom.data) > 0


def test_geom_faces_match_data(geom):
    assert len(geom.faces) == len(geom.data[-1])


def test_geom_vertices_finite(geom):
    vertices = np.array(geom.vertices)
    assert not np.isnan(vertices).any(), "NaN in Geometrie-Vertices"
    assert not np.isinf(vertices).any(), "Inf in Geometrie-Vertices"
