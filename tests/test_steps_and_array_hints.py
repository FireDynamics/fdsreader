"""sim.steps follows the column names of the FDS User Guide (section "Time Step Data"), and the
error message for converting a container to a numpy array names a method the class has."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from fdsreader import Simulation, settings
from fdsreader.simulation import StepData

CASES_DIR = Path(__file__).parent / "cases"


@pytest.fixture
def steps(monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_CACHING", False)
    monkeypatch.setattr(StepData, "_warned", False)
    return Simulation(str(CASES_DIR / "steckler_data")).steps


def test_steps_columns(steps):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert steps["Step"].dtype.kind == "i"
        assert steps["Step"][:3].tolist() == [1, 2, 3]
        assert steps["Wall Time"].dtype == np.dtype("datetime64[ms]")
        assert steps["Wall Time"][0] == np.datetime64("2024-03-14T08:39:53.199")  # +01:00 as UTC
        assert steps["Simulation Time"][0] == pytest.approx(0.10823)
        assert len({len(column) for column in steps.values()}) == 1


def test_time_step_key_warns_once_and_still_holds_wall_time(steps):
    with pytest.warns(FutureWarning, match="Wall Time") as record:
        old = steps["Time Step"]
    assert record[0].filename == __file__  # points at the caller, not into fdsreader
    np.testing.assert_array_equal(old, steps["Wall Time"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        steps["Time Step"]


def test_time_step_key_warns_through_get(steps):
    with pytest.warns(FutureWarning):
        assert steps.get("Time Step") is not None
    assert steps.get("missing", 1) == 1


def test_geomslice_array_hint_names_its_properties():
    geomslice = Simulation(str(CASES_DIR / "geomslice_data_fds611")).geomslices[0]
    with pytest.raises(TypeError, match="'data', 'vertices' and 'faces'"):
        np.asarray(geomslice)


def test_slice_array_hint_names_to_global():
    slc = Simulation(str(CASES_DIR / "steckler_data")).slices[0]
    with pytest.raises(TypeError, match="to_global"):
        np.asarray(slc)
