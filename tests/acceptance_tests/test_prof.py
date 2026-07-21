import numpy as np
import pytest

from fdsreader import Simulation


@pytest.fixture(scope="module")
def prof_sim():
    return Simulation("./prof_data_fds611")


@pytest.fixture(scope="module")
def profile(prof_sim):
    return prof_sim.profiles["PROFILE 1"]


def test_prof_times_non_empty(profile):
    assert len(profile.times) > 0


def test_prof_times_start_at_zero(profile):
    assert profile.times[0] == pytest.approx(0.0)


def test_prof_times_monotonic(profile):
    assert np.all(np.diff(profile.times) > 0), "PROF time steps must be monotonically increasing"


def test_prof_depths_values_shape_matches_npoints(profile):
    for n, depths_at_t, values_at_t in zip(profile.npoints, profile.depths, profile.values):
        assert len(depths_at_t) == n
        assert len(values_at_t) == n


def test_prof_no_nan_inf(profile):
    values = np.concatenate(profile.values)
    assert not np.isnan(values).any(), "NaN values found in PROF data"
    assert not np.isinf(values).any(), "Inf values found in PROF data"
