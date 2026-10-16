"""The table that maps .smv keywords to the methods reading them (Simulation._handler_name)."""

import pytest

from fdsreader.simulation import Simulation


@pytest.mark.parametrize(
    "keyword, handler",
    [
        ("VERSION", "_read_version"),
        ("FDSVERSION", "_read_version"),
        ("CSVF", "_read_csvf"),
        ("TEMP_MINMAX", "_read_smoke3d_limits"),
        ("DEVICE", "_read_device"),  # exactly, while DEVICE_ACT is a prefix
        ("DEVICE_ACT", "_read_device_activation"),
        ("GRID   1", "_read_mesh"),
        ("GEOM 1", "_load_geoms"),
        ("SLCF     1 # STRUCTURED %z04", "_load_slice"),
        ("SLCC     1 # STRUCTURED %z04_cc", "_load_slice"),
        ("BNDS     1", "_load_geomslice"),
        ("TISOG     1", "_load_isosurface"),  # ISOG anywhere in the keyword
        ("ISOG     1", "_load_isosurface"),
        ("PL3D   1.00     1", "_load_plot_3d"),
        ("SMOKF3D 1", "_load_smoke_3d"),
        ("SMOKG3D 1", "_load_smoke_3d"),
        ("BNDF     1     1", "_read_boundary_data"),
        ("BNDC     1     1", "_read_boundary_data"),
        ("BNDE     1     1", "_load_boundary_data_geom"),
        ("PRT5     1", "_load_particle_data"),
        ("EVA5     1", "_load_evac_data"),
        ("SHOW_OBST     1", "_toggle_obst"),
        ("HIDE_OBST     1", "_toggle_obst"),
        ("OBST", None),  # read by the mesh, not on its own
        ("", None),
        ("SOMETHING_NEW", None),
    ],
)
def test_keyword_handlers(keyword, handler):
    assert Simulation._handler_name(keyword) == handler


def test_every_handler_exists():
    names = set(Simulation._EXACT_KEYWORDS.values()) | {name for *_, name in Simulation._KEYWORD_PATTERNS}
    assert all(callable(getattr(Simulation, name, None)) for name in names)
