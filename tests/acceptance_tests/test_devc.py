import os

import numpy as np
import pytest

from fdsreader import Simulation

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def devc_sim():
    return Simulation("./steckler_data")


def test_devc(devc_sim):
    assert abs(devc_sim.devices["TC_Door"][0].data - 23.58) < 1e-6


def test_devc_time_channel_non_empty(devc_sim):
    time_dev = devc_sim.devices["Time"]
    assert len(time_dev.data) > 0


def test_devc_time_channel_starts_at_zero(devc_sim):
    time_dev = devc_sim.devices["Time"]
    assert time_dev.data[0] == pytest.approx(0.0)


def test_devc_time_channel_monotonic(devc_sim):
    time_dev = devc_sim.devices["Time"]
    assert np.all(np.diff(time_dev.data) > 0), "Time channel must be monotonically increasing"


def test_devc_quantity_has_name(devc_sim):
    assert devc_sim.devices["TC_Door"][0].quantity_name != ""


def test_clear_cache_with_line_devices():
    """Test that clear_cache works with line DEVC devices (issue #104)."""
    sim = Simulation(os.path.join(TEST_DIR, "../cases/devc_data"))
    assert any(isinstance(d, list) for d in sim.devices), "Test data should contain line devices"
    sim.clear_cache()


def test_activation_times_type(devc_sim):
    """activation_times entries must be (float, bool) tuples (PR #101)."""
    for entry in devc_sim.devices["TC_Room"]:
        for t, s in entry.activation_times:
            assert isinstance(t, float), f"Expected float time, got {type(t)}"
            assert isinstance(s, bool), f"Expected bool state, got {type(s)}"


def test_activation_times_monotonic(devc_sim):
    """activation_times must be sorted by time (PR #101)."""
    for entry in devc_sim.devices["TC_Room"]:
        times = [t for t, _ in entry.activation_times]
        assert times == sorted(times), "activation_times not sorted by time"
