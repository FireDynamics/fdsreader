"""Regression tests for several Simulation-level edge cases fixed in this session:
- GeometryCollection eagerly loading data when settings.LAZY_LOAD is False.
- The _line.csv path being built from `chid` rather than derived from another CSV's path/name
  (a fixture directory whose name itself contains "devc" is exactly the case that broke a
  string-replace-based approach).
- Profile parsing tolerating rows whose Npoints shrinks/grows between timesteps.
- &CTRL CSV support (previously unhandled CSVF type, silently ignored).
- log_error stripping the traceback before storing a failed loader's exception in load_errors.
"""

from types import SimpleNamespace

from fdsreader import Simulation, settings
from fdsreader.simulation import Simulation as SimulationClass
from fdsreader.utils.misc import log_error


def test_geometry_collection_eager_loads_when_lazy_load_disabled():
    settings.LAZY_LOAD = False
    sim = Simulation("./geom_data")
    assert len(sim.geom_data) > 0
    for geom in sim.geom_data:
        assert hasattr(geom, "_data")


def test_line_csv_is_found_via_chid_even_when_directory_name_contains_devc():
    # The fixture directory is named "devc_data" - a naive derivation of the line-csv path from
    # the devc-csv path/name (e.g. via str.replace on the containing path) would be fragile here.
    # The current implementation builds the path from `self.chid` directly, which is unaffected.
    sim = Simulation("./devc_data")
    devices = sim.devices["TC_Room"]
    assert isinstance(devices, list)
    assert len(devices) > 1
    _ = devices[0].data  # triggers _load_DEVC_data, which reads chid + "_line.csv"
    assert hasattr(devices[0], "_data")
    assert devices[0].quantity.unit == "C"


def test_ctrl_csv_is_loaded_into_sim_ctrl():
    sim = Simulation("./part_data")
    assert "Add" in sim.ctrl
    assert len(sim.ctrl["Add"]) > 0


def test_profile_parsing_tolerates_shrinking_npoints_between_rows(tmp_path):
    prof_file = tmp_path / "test_prof_1.csv"
    prof_file.write_text(
        "ID, IOR, face center x(m), y(m), z(m)\n"
        "profile1, 1, 0.0, 0.0, 0.0\n"
        "Time(s), Npoints, depth(1), ..., value(1), ...\n"
        " 0.0, 3, 0.1, 0.2, 0.3, 10.0, 20.0, 30.0\n"
        " 1.0, 2, 0.1, 0.2, 10.0, 20.0\n"
    )
    fake_sim = SimpleNamespace(root_path=str(tmp_path), chid="test", profiles={}, load_errors=[])
    SimulationClass._load_profiles(fake_sim)
    assert fake_sim.load_errors == []  # surface the real exception instead of a KeyError below

    profile = fake_sim.profiles["profile1"]
    assert list(profile.npoints) == [3, 2]
    assert list(profile.depths[0]) == [0.1, 0.2, 0.3]
    assert list(profile.depths[1]) == [0.1, 0.2]
    assert list(profile.values[1]) == [10.0, 20.0]


def test_profile_parsing_skips_incomplete_trailing_row(tmp_path):
    # A row that's shorter than its own declared Npoints (e.g. the last line of a profile file
    # still being written by a running simulation) must be skipped, not crash the whole file.
    prof_file = tmp_path / "test_prof_1.csv"
    prof_file.write_text(
        "ID, IOR, face center x(m), y(m), z(m)\n"
        "profile1, 1, 0.0, 0.0, 0.0\n"
        "Time(s), Npoints, depth(1), ..., value(1), ...\n"
        " 0.0, 2, 0.1, 0.2, 10.0, 20.0\n"
        " 1.0, 2, 0.1\n"
    )
    fake_sim = SimpleNamespace(root_path=str(tmp_path), chid="test", profiles={}, load_errors=[])
    SimulationClass._load_profiles(fake_sim)
    assert fake_sim.load_errors == []  # surface the real exception instead of a KeyError below

    profile = fake_sim.profiles["profile1"]
    assert len(profile.times) == 1
    assert profile.times[0] == 0.0


def test_log_error_strips_traceback_before_storing_exception():
    fake_sim = SimpleNamespace(load_errors=[])

    @log_error("test-module")
    def failing_loader(self):
        raise ValueError("boom")

    failing_loader(fake_sim)

    assert len(fake_sim.load_errors) == 1
    module, exc = fake_sim.load_errors[0]
    assert module == "test-module"
    assert isinstance(exc, ValueError)
    assert exc.__traceback__ is None
