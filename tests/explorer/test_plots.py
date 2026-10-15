"""Matplotlib drawing of the explorer, on the Agg backend so no display is needed."""

import numpy as np
import pytest

plt = pytest.importorskip("matplotlib.pyplot")

import fdsreader  # noqa: E402
from fdsreader.explorer.data import list_devices  # noqa: E402
from fdsreader.explorer.plots import plot_device, plot_series, plot_slice  # noqa: E402

pytestmark = pytest.mark.usefixtures("agg")


def test_nothing_selected():
    ax = plot_series([])
    assert ax.get_title() == "Nothing selected"
    assert ax.get_xlabel() == "Time [s]"


def test_one_series_names_its_range(fake_series):
    ax = plot_series(fake_series[:1], cursor_time=2.0)
    assert ax.get_title().startswith("DEVC  A")
    assert "max 60" in ax.get_title()
    assert len(ax.get_lines()) == 2  # the curve and the cursor


def test_several_series_get_a_legend(fake_series):
    ax = plot_series(fake_series)
    assert ax.get_title() == "2 series"
    assert ax.get_legend() is not None


def test_a_second_unit_goes_on_a_second_axis(fake_series):
    hrr = dict(fake_series[0], label="HRR [kW]", unit="kW", quantity="HRR", source="hrr")
    ax = plot_series([fake_series[0], hrr])
    twins = [other for other in ax.figure.axes if other is not ax]
    assert len(twins) == 1 and "kW" in twins[0].get_ylabel()


def test_a_device_shows_when_it_activated(tiny_case):
    fdsreader.settings.ENABLE_CACHING = False
    sim = fdsreader.Simulation(str(tiny_case))
    device = next(d for d in list_devices(sim) if d.id == "TC_2")
    ax = plot_device(sim, device)
    assert "position (2, 0, 2)" in ax.get_title()
    assert any(text.get_text() == "activated" for text in ax.texts)


@pytest.mark.filterwarnings("ignore:genfromtxt")
def test_a_device_without_data_draws_no_data(devc_without_data):
    fdsreader.settings.ENABLE_CACHING = False
    sim = fdsreader.Simulation(str(devc_without_data))
    ax = plot_device(sim, list_devices(sim)[0])
    assert all(len(line.get_xdata()) == 0 for line in ax.get_lines())


@pytest.mark.parametrize("render", ["image", "filled", "lines"])
def test_a_slice_in_each_render_mode(fake_field, render):
    ax = plot_slice(fake_field, 1, render=render, n_levels=6)
    assert ax.get_xlabel() == "y [m]" and ax.get_ylabel() == "z [m]"
    assert "t = 1.00 s" in ax.get_title()
    colorbar = ax.figure.axes[-1]
    assert colorbar.get_ylabel() == "TEMPERATURE [C]"
    if render == "image":
        assert ax.images[0].get_clim() == pytest.approx(fake_field.value_range()[:2])
    else:
        assert ax.get_xlim() == pytest.approx((-2.0, 2.0))


def test_a_slice_with_given_limits_and_frame(fake_field):
    frame = np.zeros(fake_field.shape)
    ax = plot_slice(fake_field, 0, vmin=0.0, vmax=1.0, cmap="viridis", frame=frame)
    image = ax.images[0]
    assert image.get_clim() == (0.0, 1.0)
    assert image.get_cmap().name == "viridis"
    assert not image.get_array().any()
