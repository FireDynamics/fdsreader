"""The rules every front end shares: which frame, and on what colour scale."""

import numpy as np
import pytest

from fdsreader.explorer.state import ExplorerState


def test_field_is_clamped_to_what_exists(fake_field):
    state = ExplorerState(field_index=7)
    assert state.field([fake_field]) is fake_field
    assert state.field_index == 0
    assert state.field([]) is None


def test_timestep_is_clamped(fake_field):
    state = ExplorerState(timestep=99)
    assert state.clamp([fake_field]) == len(fake_field.times) - 1


def test_seek_picks_the_nearest_time(fake_field):
    state = ExplorerState()
    assert state.seek([fake_field], 0.9) == 1     # times are 0.0, 1.0, 2.0
    assert state.time([fake_field]) == pytest.approx(1.0)
    assert state.seek([fake_field], 99.0) == 2    # past the end lands on the last step


def test_global_scale_is_the_same_for_every_step(fake_field):
    state = ExplorerState(scale_mode="global")
    first = state.limits(fake_field)
    state.timestep = 2
    assert state.limits(fake_field) == first
    assert first[2] is None


def test_step_scale_follows_the_frame(fake_field):
    state = ExplorerState(scale_mode="step", timestep=0)
    low, high, warning = state.limits(fake_field)
    assert warning is None
    assert high < fake_field.value_range()[1]     # one frame is cooler than the whole run


def test_step_scale_reuses_a_frame_that_was_already_read(fake_field):
    state = ExplorerState(scale_mode="step")
    frame = np.full(fake_field.shape, 100.0)
    low, high, _ = state.limits(fake_field, frame=frame)
    assert (low, high) == (99.5, 100.5)           # a flat frame still gets a finite range


def test_manual_scale_is_used_when_it_is_valid(fake_field):
    state = ExplorerState(scale_mode="manual", manual_limits=(10.0, 30.0))
    assert state.limits(fake_field) == (10.0, 30.0, None)


@pytest.mark.parametrize("limits", [(30.0, 10.0), (5.0, 5.0), None, (np.nan, 1.0)])
def test_manual_scale_falls_back_and_says_why(fake_field, limits):
    state = ExplorerState(scale_mode="manual", manual_limits=limits)
    low, high, warning = state.limits(fake_field)
    assert (low, high) == fake_field.value_range()[:2]
    assert "min must be below max" in warning


def test_auto_colormap_defers_to_the_field(fake_field):
    assert ExplorerState(cmap="auto").colormap(fake_field) == fake_field.cmap
    assert ExplorerState(cmap="viridis").colormap(fake_field) == "viridis"


def test_selected_series_ignores_stale_indices(fake_series):
    state = ExplorerState(curves=(1, 99, -1))
    assert state.selected_series(fake_series) == [fake_series[1]]
