"""Converting Smoke3D display codes back to physical values (SubSmoke3D.to_physical).

FDS writes Smoke3D values as bytes between 0 and 254 (SMOKE3D_TO_FILE in Source/smvv.f90):

- gas density (soot): FDS < 6.11 (format version 0) stores the obscuration
  254 * (1 - exp(-K_m * dx * rho)); FDS >= 6.11 (version 1) stores 254 * rho / MAX_VAL,
  normalised by the maximum density of each frame (MAX_VAL in the .s3d.sz file)
- HRRPUV: 254 * q / q_max, clipped to [0, q_max]
- temperature and all other quantities: 254 * (T - T_min) / (T_max - T_min), clipped

Before FDS 6.11 HRRPUV and temperature were truncated, from 6.11 on rounded.
"""

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

import fdsreader.smoke3d.smoke3d as smoke3d_module
from fdsreader import Simulation
from fdsreader.fds_classes import Mesh
from fdsreader.smoke3d import Smoke3D
from fdsreader.utils import Quantity

CASES_DIR = Path(__file__).parent / "cases"

SOOT = Quantity("SOOT DENSITY", "rho_C", "kg/m3")
HRRPUV = Quantity("HRRPUV", "hrrpuv", "kW/m3")
TEMPERATURE = Quantity("EFFECTIVE FLAME TEMPERATURE", "temp", "C")


def _record(payload: bytes) -> bytes:
    return struct.pack("<i", len(payload)) + payload + struct.pack("<i", len(payload))


def _make_mesh(nx, ny, nz):
    coordinates = {
        "x": np.linspace(0.0, 0.1 * nx, nx + 1),
        "y": np.linspace(0.0, 0.1 * ny, ny + 1),
        "z": np.linspace(0.0, 0.1 * nz, nz + 1),
    }
    extents = {dim: (co[0], co[-1]) for dim, co in coordinates.items()}
    return Mesh(coordinates, extents, "1")


def _write_s3d(path: Path, times, codes, version):
    """Write display codes (shape: n_t, nx+1, ny+1, nz+1) in the .s3d format."""
    nx, ny, nz = (n - 1 for n in codes.shape[1:])
    content = _record(struct.pack("<8i", 1, version, 0, nx, 0, ny, 0, nz))
    for t, frame in zip(times, codes):
        values = frame.ravel(order="F").astype(np.uint8)
        # Values never exceed 254, so writing every byte as a literal is valid RLE output.
        content += _record(struct.pack("<f", t))
        content += _record(struct.pack("<2i", values.size, values.size))
        content += _record(values.tobytes())
    path.write_bytes(content)


def _subsmoke_with_codes(quantity, codes, version, mass_extinction_coefficient=0.0, limits=None, max_vals=None):
    """A SubSmoke3D whose codes are set directly, without a file."""
    smoke = Smoke3D("", np.arange(codes.shape[0], dtype=np.float32), quantity, limits=limits)
    upper = np.ones(codes.shape[0]) if max_vals is None else max_vals
    mesh = _make_mesh(*(n - 1 for n in codes.shape[1:]))
    sub = smoke._add_subsmoke("a.s3d", mesh, upper, version, mass_extinction_coefficient, max_vals)
    sub._data = codes.astype(np.float32)
    return smoke, sub


@pytest.fixture(autouse=True)
def _reset_raw_codes_warning():
    smoke3d_module._raw_codes_warned = False
    yield
    smoke3d_module._raw_codes_warned = False


def test_soot_v1_is_rescaled_per_frame(tmp_path):
    mesh = _make_mesh(2, 1, 1)
    times = np.array([0.0, 1.0], dtype=np.float32)
    rho = np.stack([np.linspace(0.0, 2.0e-4, 12).reshape(3, 2, 2), np.linspace(0.0, 8.0e-4, 12).reshape(3, 2, 2)])
    max_vals = rho.reshape(2, -1).max(axis=1)
    codes = np.rint(254.0 * rho / max_vals.reshape(-1, 1, 1, 1))
    _write_s3d(tmp_path / "case_1_1.s3d", times, codes, version=1)

    smoke = Smoke3D(str(tmp_path), times, SOOT)
    sub = smoke._add_subsmoke("case_1_1.s3d", mesh, max_vals, 1, 8700.0, max_vals)

    # The same relative position in both frames is stored as the same code...
    assert np.array_equal(sub._codes()[0], sub._codes()[1])
    # ...but represents a four times higher density in the second frame.
    values, saturated = sub.to_physical(return_saturated=True)
    np.testing.assert_allclose(values, rho, atol=max_vals.max() / 254 / 2)
    assert not saturated.any()  # 254 is the frame maximum, not a cut-off
    np.testing.assert_allclose(smoke.to_physical()[:, :3, :2, :2], values)


def test_species_without_extinction_coefficient_is_still_a_density():
    smoke, sub = _subsmoke_with_codes(
        Quantity("MY GAS DENSITY", "rho_X", "kg/m3"), np.full((1, 2, 2, 2), 127), 1, max_vals=np.array([2.0])
    )
    assert smoke.kind == "gas"
    np.testing.assert_allclose(sub.to_physical(), 1.0)


def test_soot_v0_inverts_obscuration_and_marks_saturation():
    k_m, dx = 8700.0, 0.1
    rho = np.array([0.0, 1e-4, 5e-4, 1e-2])
    codes = np.zeros((1, 2, 2, 2))
    codes.flat[:4] = np.rint(254 * (1 - np.exp(-k_m * dx * rho)))
    _, sub = _subsmoke_with_codes(SOOT, codes, 0, k_m)

    values, saturated = sub.to_physical(return_saturated=True)
    # 1e-2 kg/m3 saturates the byte at 254, which stays finite and is marked.
    assert codes.flat[3] == 254 and np.isfinite(values.flat[3]) and saturated.flat[3]
    assert not saturated.flat[:3].any()
    np.testing.assert_allclose(values.flat[:3], rho[:3], rtol=0.05, atol=1e-6)


def test_soot_v0_without_extinction_coefficient_cannot_be_recovered():
    _, sub = _subsmoke_with_codes(SOOT, np.zeros((1, 2, 2, 2)), 0, 0.0)
    with pytest.raises(ValueError):
        sub.to_physical()


@pytest.mark.parametrize("version", [0, 1])
def test_hrrpuv(version):
    q_max = 1200.0
    q = np.array([0.0, 100.0, 600.0, 1199.0, 5000.0])
    scaled = 254 * np.clip(q, 0, q_max) / q_max
    codes = np.zeros((1, 2, 2, 2))
    codes.flat[:5] = np.rint(scaled) if version >= 1 else np.floor(scaled)
    smoke, sub = _subsmoke_with_codes(HRRPUV, codes, version, limits={"HRRPUV_MINMAX": (0.0, q_max)})
    assert smoke.kind == "hrrpuv"

    values, saturated = sub.to_physical(return_saturated=True)
    step = q_max / 254
    # Within half a step: the middle of the truncated step, or the rounded code
    np.testing.assert_allclose(values.flat[1:4], q[1:4], atol=step / 2 + 1e-3)
    assert values.flat[0] == 0.0
    assert values.flat[4] == q_max and saturated.flat[4]
    assert not saturated.flat[:3].any()
    # 1199 is half a step below the limit: truncated it stays below (253), rounded it becomes 254,
    # which cannot be told apart from values above the limit and is marked as saturated
    assert saturated.flat[3] == (version >= 1)


@pytest.mark.parametrize("version", [0, 1])
def test_temperature(version):
    t_min, t_max = 20.0, 1000.0
    t = np.array([20.0, 300.0, 750.0, 1500.0])
    scaled = 254 * (np.clip(t, t_min, t_max) - t_min) / (t_max - t_min)
    codes = np.zeros((1, 2, 2, 2))
    codes.flat[:4] = np.rint(scaled) if version >= 1 else np.floor(scaled)
    smoke, sub = _subsmoke_with_codes(TEMPERATURE, codes, version, limits={"TEMP_MINMAX": (t_min, t_max)})
    assert smoke.kind == "temperature"

    values, saturated = sub.to_physical(return_saturated=True)
    step = (t_max - t_min) / 254
    np.testing.assert_allclose(values.flat[1:3], t[1:3], atol=step / 2 + 1e-3)
    assert values.flat[0] == t_min
    assert values.flat[3] == t_max and saturated.flat[3]


def test_limits_can_be_overridden():
    codes = np.full((1, 2, 2, 2), 127)
    _, sub = _subsmoke_with_codes(HRRPUV, codes, 1, limits={"HRRPUV_MINMAX": (0.0, 1200.0)})
    np.testing.assert_allclose(sub.to_physical(hrrpuv_max=2540.0), 1270.0)


def test_missing_limits_fall_back_to_fds_defaults_with_a_warning():
    codes = np.full((1, 2, 2, 2), 254)
    _, sub = _subsmoke_with_codes(HRRPUV, codes, 0)
    with pytest.warns(UserWarning, match="HRRPUV_MINMAX"):
        values = sub.to_physical()
    np.testing.assert_allclose(values, 1200.0)
    # Given limits need no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sub.to_physical(hrrpuv_max=1200.0)


def test_raw_codes_warn_once():
    _, sub = _subsmoke_with_codes(HRRPUV, np.zeros((1, 2, 2, 2)), 1, limits={"HRRPUV_MINMAX": (0.0, 1200.0)})
    with pytest.warns(UserWarning, match="display codes"):
        _ = sub.data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = sub.data
        sub.to_physical()


def test_return_saturated_needs_physical():
    smoke, _ = _subsmoke_with_codes(HRRPUV, np.zeros((1, 2, 2, 2)), 1)
    with pytest.raises(ValueError):
        smoke.to_global(return_saturated=True)


def test_legacy_steckler():
    sim = Simulation(str(CASES_DIR / "steckler_data"))
    soot = sim.smoke_3d.get_by_quantity("SOOT DENSITY")
    assert soot.version == 0 and soot.kind == "gas"
    assert soot.mass_extinction_coefficient == pytest.approx(8700.0)
    density = soot.subsmokes[0].to_physical()
    assert density.shape == soot.subsmokes[0]._codes().shape
    assert np.all(np.isfinite(density)) and np.all(density >= 0)

    hrrpuv = sim.smoke_3d.get_by_quantity("HRRPUV")
    assert hrrpuv.version == 0 and hrrpuv.kind == "hrrpuv" and hrrpuv.mass_extinction_coefficient == 0.0
    values, saturated, coordinates = hrrpuv.to_global(
        physical=True, return_saturated=True, return_coordinates=True, hrrpuv_max=1200.0
    )
    assert values.shape == saturated.shape and set(coordinates) == {"x", "y", "z"}
    assert np.nanmin(values) >= 0 and np.nanmax(values) <= 1200.0


def test_export_reads_codes_without_warning(tmp_path):
    pytest.importorskip("multiprocess")
    pytest.importorskip("yaml")
    from fdsreader.export import export_smoke_raw

    sim = Simulation(str(CASES_DIR / "steckler_data"))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Smoke3D data are")
        export_smoke_raw(sim.smoke_3d.get_by_quantity("HRRPUV"), str(tmp_path))


def test_eager_loading_does_not_warn(monkeypatch):
    from fdsreader import settings

    monkeypatch.setattr(settings, "LAZY_LOAD", False)
    monkeypatch.setattr(settings, "ENABLE_CACHING", False)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Smoke3D data are")
        Simulation(str(CASES_DIR / "steckler_data"))


def test_kind_can_be_overridden():
    # Total gas density chosen with &SM3D QUANTITY='DENSITY' (no SPEC_ID) is scaled like temperature
    smoke, sub = _subsmoke_with_codes(
        Quantity("DENSITY", "rho", "kg/m3"), np.full((1, 2, 2, 2), 127), 1, limits={"TEMP_MINMAX": (0.0, 2.54)}
    )
    assert smoke.kind == "temperature"
    np.testing.assert_allclose(sub.to_physical(), 1.27, rtol=1e-6)
    np.testing.assert_allclose(sub.to_physical(kind="hrrpuv", hrrpuv_max=254.0), 127.0)
    np.testing.assert_allclose(smoke.to_physical(kind="hrrpuv", hrrpuv_max=254.0)[:, :2, :2, :2], 127.0)
    with pytest.raises(ValueError):
        sub.to_physical(kind="soot")


def test_subsmoke_without_smoke3d_raises():
    from fdsreader.smoke3d.smoke3d import SubSmoke3D

    sub = SubSmoke3D("a.s3d", _make_mesh(1, 1, 1), np.ones(1), np.zeros(1))
    sub._data = np.zeros((1, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        sub.to_physical()


def test_smv_limits_and_max_val_column_fds610():
    sim = Simulation(str(CASES_DIR / "steckler_data_fds6100"))
    assert sim._smoke3d_limits["HRRPUV_MINMAX"] == (0.0, 1200.0)
    assert sim._smoke3d_limits["TEMP_MINMAX"] == (20.0, 2000.0)
    assert "AMBIENT_TEMPERATURE" not in sim._smoke3d_limits  # only needed before FDS 6.10
    soot = sim.smoke_3d.get_by_quantity("SOOT DENSITY").subsmokes[0]
    # FDS 6.10 writes six columns for soot: MAX_VAL is the 4th, MAX_DENSITY_VAL the last
    sz_rows = np.loadtxt(soot._file_path + ".sz", skiprows=1, ndmin=2)
    assert sz_rows.shape[1] == 6
    np.testing.assert_array_equal(soot.max_vals, sz_rows[:, 3])
    np.testing.assert_array_equal(soot.upper_bounds, sz_rows[:, -1])


def test_fds68_temperature_starts_at_ambient_temperature():
    # steckler_data was written by FDS 6.8.0 with TMPA = 22 C and has no TEMP_MINMAX
    sim = Simulation(str(CASES_DIR / "steckler_data"))
    assert "TEMP_MINMAX" not in sim._smoke3d_limits
    assert sim._smoke3d_limits["AMBIENT_TEMPERATURE"][0] == pytest.approx(22.0)
    temperature = sim.smoke_3d.get_by_quantity("TEMPERATURE")
    with pytest.warns(UserWarning, match="ambient temperature") as record:
        values = temperature.subsmokes[0].to_physical()
    assert record[0].filename == __file__  # points at the caller, not into fdsreader
    codes = temperature.subsmokes[0]._codes()
    assert np.all(values[codes == 0] == pytest.approx(22.0))


def test_default_limit_warning_per_keyword():
    _, sub = _subsmoke_with_codes(HRRPUV, np.zeros((1, 2, 2, 2)), 0)
    with pytest.warns(UserWarning, match="HRRPUV_MINMAX"):
        sub.to_physical()
    with pytest.warns(UserWarning, match="TEMP_MINMAX"):
        sub.to_physical(kind="temperature")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sub.to_physical()
        sub.to_physical(kind="temperature")


def test_cache_from_an_older_layout_is_parsed_again(monkeypatch, tmp_path):
    import shutil

    from fdsreader import settings

    case = tmp_path / "steckler"
    shutil.copytree(CASES_DIR / "steckler_data_fds6100", case)
    monkeypatch.setattr(settings, "ENABLE_CACHING", True)
    Simulation(str(case))  # writes the cache
    assert Simulation(str(case))._cache_format == Simulation._CACHE_FORMAT

    monkeypatch.setattr(Simulation, "_CACHE_FORMAT", Simulation._CACHE_FORMAT + 1)
    reparsed = Simulation(str(case))
    assert reparsed._cache_format == Simulation._CACHE_FORMAT
