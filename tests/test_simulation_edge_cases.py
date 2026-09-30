"""Regression tests for several Simulation-level edge cases fixed in this session:
- GeometryCollection eagerly loading data when settings.LAZY_LOAD is False.
- The _line.csv path being built from `chid` rather than derived from another CSV's path/name
  (a fixture directory whose name itself contains "devc" is exactly the case that broke a
  string-replace-based approach).
- Profile parsing tolerating rows whose Npoints shrinks/grows between timesteps.
- &CTRL CSV support (previously unhandled CSVF type, silently ignored).
- log_error stripping the traceback before storing a failed loader's exception in load_errors.

Also covers two gaps found while investigating why local coverage of simulation.py looked far
lower than it should: stale *.pickle caches left over in tests/cases/ (now cleaned up by
conftest.py) meant Simulation.__new__'s pickle-cache branches, and _toggle_obst (no fixture uses
&HIDE_OBST/&SHOW_OBST), were essentially never exercised.
"""

import io
import pickle
import shutil
from types import SimpleNamespace

import pytest

from fdsreader import Simulation, __version__, settings
from fdsreader.bndf.obstruction import SubObstruction
from fdsreader.simulation import Simulation as SimulationClass
from fdsreader.utils import Extent
from fdsreader.utils.misc import log_error


def test_geometry_collection_eager_loads_when_lazy_load_disabled():
    # The on-disk pickle cache's validity check (Simulation.__new__) only compares fdsreader
    # version and smv-file hash, not settings.LAZY_LOAD - so without disabling caching here, this
    # test could silently hit a cache another test already wrote for this same fixture under the
    # default LAZY_LOAD=True, and would then assert on stale (non-eagerly-loaded) data.
    settings.ENABLE_CACHING = False
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


class TestToggleObst:
    """_toggle_obst (SMV keywords HIDE_OBST/SHOW_OBST) has no fixture exercising it, so it's
    tested directly against a fake Simulation carrying real SubObstruction instances."""

    @staticmethod
    def _fake_sim():
        sub = SubObstruction(
            side_surfaces=(), bound_indices=(0, 1, 0, 1, 0, 1), extent=Extent(0, 1, 0, 1, 0, 1), mesh=None
        )
        mesh = SimpleNamespace(id="mesh1")
        return SimpleNamespace(_meshes=[mesh], _subobstructions={"mesh1": [sub]}), sub

    def test_hide_obst_records_a_hide_time_on_the_addressed_subobstruction(self):
        fake_sim, sub = self._fake_sim()
        SimulationClass._toggle_obst(fake_sim, io.StringIO("1 12.5\n"), "HIDE_OBST 1")
        assert sub.hide_times == [12.5]
        assert sub.show_times == []

    def test_show_obst_records_a_show_time_on_the_addressed_subobstruction(self):
        fake_sim, sub = self._fake_sim()
        SimulationClass._toggle_obst(fake_sim, io.StringIO("1 3.0\n"), "SHOW_OBST 1")
        assert sub.show_times == [3.0]
        assert sub.hide_times == []

    def test_visibility_toggles_are_reflected_in_get_visible_times(self):
        fake_sim, sub = self._fake_sim()
        SimulationClass._toggle_obst(fake_sim, io.StringIO("1 1.0\n"), "HIDE_OBST 1")
        SimulationClass._toggle_obst(fake_sim, io.StringIO("1 2.0\n"), "SHOW_OBST 1")
        visible = sub.get_visible_times([0.0, 1.0, 1.5, 2.0, 3.0])
        assert list(visible) == [0.0, 2.0, 3.0]

    def test_addresses_the_correct_mesh_and_subobstruction_among_several(self):
        # With only one mesh and one subobstruction, int(line[-1])-1 and int(obst_index)-1 can
        # only ever resolve to index 0 - an off-by-one or a mesh/obst-index mix-up would still
        # pass. Build 2 meshes x 2 subobstructions each and toggle only one specific pair.
        def _sub():
            return SubObstruction(
                side_surfaces=(), bound_indices=(0, 1, 0, 1, 0, 1), extent=Extent(0, 1, 0, 1, 0, 1), mesh=None
            )

        subs = {"mesh1": [_sub(), _sub()], "mesh2": [_sub(), _sub()]}
        fake_sim = SimpleNamespace(
            _meshes=[SimpleNamespace(id="mesh1"), SimpleNamespace(id="mesh2")], _subobstructions=subs
        )

        # Mesh index 2 (mesh2), obst index 2 (second subobstruction in mesh2).
        SimulationClass._toggle_obst(fake_sim, io.StringIO("2 5.0\n"), "HIDE_OBST 2")

        assert subs["mesh2"][1].hide_times == [5.0]
        assert subs["mesh1"][0].hide_times == []
        assert subs["mesh1"][1].hide_times == []
        assert subs["mesh2"][0].hide_times == []


class TestNewCachingEdgeCases:
    """Simulation.__new__'s on-disk pickle-cache branches (corrupt file, stale version/hash,
    missing load_errors on an old cache) were essentially untested locally because leftover
    *.pickle files from earlier runs made every fixture load hit the cache-valid fast path
    instead. conftest.py now clears those caches before each test session; these tests exercise
    the cache-miss/invalid-cache paths directly using an isolated copy of a small fixture."""

    @pytest.fixture
    def isolated_case(self, tmp_path):
        case_dir = tmp_path / "devc_data"
        shutil.copytree("./devc_data", case_dir)
        # Another test may have left a valid *.pickle behind in the real devc_data directory
        # (e.g. via Simulation.clear_cache()'s default clear_persistent_cache=False) - copytree
        # would carry that into this "isolated" copy, making sim1 below a silent cache hit instead
        # of the fresh parse every test in this class assumes.
        for pickle_file in case_dir.glob("*.pickle"):
            pickle_file.unlink()
        settings.ENABLE_CACHING = True
        return case_dir

    def test_new_raises_valueerror_for_empty_smv_file(self, tmp_path):
        smv_path = tmp_path / "empty.smv"
        smv_path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            Simulation(str(smv_path))

    def test_new_raises_valueerror_when_chid_missing(self, tmp_path):
        smv_path = tmp_path / "no_chid.smv"
        smv_path.write_text("VERSION\n1\nTITLE\nfoo\n")
        with pytest.raises(ValueError, match="CHID"):
            Simulation(str(smv_path))

    def test_new_recovers_from_a_corrupt_pickle_cache_file(self, isolated_case):
        sim1 = Simulation(str(isolated_case))
        pickle_path = isolated_case / (sim1.chid + ".pickle")
        assert pickle_path.exists()

        pickle_path.write_bytes(b"not a valid pickle stream")
        sim2 = Simulation(str(isolated_case))  # must fall back to a fresh parse, not raise
        assert sim2.chid == sim1.chid

    def test_new_discards_cache_from_a_different_fdsreader_version(self, isolated_case):
        sim1 = Simulation(str(isolated_case))
        pickle_path = isolated_case / (sim1.chid + ".pickle")
        sim1.reader_version = "0.0.0-not-a-real-version"
        with open(pickle_path, "wb") as f:
            pickle.dump(sim1, f)

        sim2 = Simulation(str(isolated_case))
        assert sim2.reader_version == __version__

    def test_new_discards_cache_when_smv_file_hash_no_longer_matches(self, isolated_case):
        sim1 = Simulation(str(isolated_case))
        pickle_path = isolated_case / (sim1.chid + ".pickle")
        sim1._hash = "not-the-real-hash"
        with open(pickle_path, "wb") as f:
            pickle.dump(sim1, f)

        sim2 = Simulation(str(isolated_case))
        assert sim2._hash != "not-the-real-hash"

    def test_new_backfills_load_errors_on_a_pickle_cache_missing_the_attribute(self, isolated_case):
        sim1 = Simulation(str(isolated_case))
        pickle_path = isolated_case / (sim1.chid + ".pickle")
        del sim1.load_errors
        with open(pickle_path, "wb") as f:
            pickle.dump(sim1, f)

        sim2 = Simulation(str(isolated_case))
        assert sim2.load_errors == []


class TestKeywordHandlerRegistry:
    """parse_smv_file's if/elif chain was replaced with two frozen registries
    (_EXACT_KEYWORD_HANDLERS/_PREFIX_KEYWORD_HANDLERS_BEFORE_ISOG/_AFTER_ISOG) mapping SMV
    keywords to handler-method NAMES (strings), resolved via getattr() at parse time. A typo'd or
    renamed handler name would only surface as an AttributeError the next time a real .smv file
    happens to contain that specific keyword - not at import/class-definition time and not
    necessarily caught by the fixture-based acceptance tests, since not every fixture exercises
    every keyword. This test catches that class of regression immediately."""

    def test_every_registered_handler_name_resolves_to_a_real_method(self):
        handler_names = set(SimulationClass._EXACT_KEYWORD_HANDLERS.values())
        handler_names.update(name for _, name in SimulationClass._PREFIX_KEYWORD_HANDLERS_BEFORE_ISOG)
        handler_names.update(name for _, name in SimulationClass._PREFIX_KEYWORD_HANDLERS_AFTER_ISOG)
        handler_names.add("_load_isosurface")  # the ISOG substring fallback, not in either registry

        for name in handler_names:
            assert callable(getattr(SimulationClass, name, None)), f"{name!r} is not a callable attribute"

    def test_registries_are_frozen_against_accidental_mutation(self):
        with pytest.raises(TypeError):
            SimulationClass._EXACT_KEYWORD_HANDLERS["NEW"] = "_handle_new"
        with pytest.raises(TypeError):
            SimulationClass._PREFIX_KEYWORD_HANDLERS_BEFORE_ISOG[0] = ("NEW", "_handle_new")
        with pytest.raises(TypeError):
            SimulationClass._PREFIX_KEYWORD_HANDLERS_AFTER_ISOG[0] = ("NEW", "_handle_new")
