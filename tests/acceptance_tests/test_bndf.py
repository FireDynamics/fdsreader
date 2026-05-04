import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def bndf_sim():
    return Simulation("./bndf_data")


@pytest.fixture(scope="module")
def bndf_obst(bndf_sim):
    return bndf_sim.obstructions.get_nearest(-0.8, 1, 1)


def test_bndf(bndf_obst):
    face = bndf_obst.get_global_boundary_data_arrays("Wall Temperature")[1]
    assert len(face[-1]) == 68


def test_bndf_times_non_empty(bndf_obst):
    assert len(bndf_obst.times) > 0


def test_bndf_times_start_at_zero(bndf_obst):
    assert bndf_obst.times[0] == pytest.approx(0.0)


def test_bndf_times_monotonic(bndf_obst):
    assert np.all(np.diff(bndf_obst.times) > 0), "BNDF time steps must be monotonically increasing"


def test_bndf_data_shape_matches_times(bndf_obst):
    face = bndf_obst.get_global_boundary_data_arrays("Wall Temperature")[1]
    assert face.shape[0] == len(bndf_obst.times)


def test_bndf_no_nan(bndf_obst):
    face = bndf_obst.get_global_boundary_data_arrays("Wall Temperature")[1]
    assert not np.isnan(face).any(), "NaN values found in BNDF data"
