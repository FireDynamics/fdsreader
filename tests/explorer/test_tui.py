"""The interactive front end's behaviour, without a terminal.

Key handling and layout are ordinary functions, so what the application *does* can be
tested directly; only the drawing needs curses.
"""

import pytest

from fdsreader.explorer import tui
from fdsreader.explorer.tui import SIDE_BY_SIDE_COLUMNS, Interactive, layout


class FakeSim:
    chid = "fake"
    root_path = "."


@pytest.fixture
def app(monkeypatch, fake_field, fake_series):
    """An application over one field and two curves."""
    monkeypatch.setattr(tui, "fields_of", lambda sim: [fake_field, fake_field])
    monkeypatch.setattr(tui, "build_series", lambda sim: fake_series)
    return Interactive(FakeSim())


# -- layout ---------------------------------------------------------------
def test_wide_terminal_puts_the_panes_side_by_side():
    boxes = layout(40, SIDE_BY_SIDE_COLUMNS)
    assert boxes["slice"][0] == boxes["curves"][0]      # same top row
    assert boxes["curves"][1] > 0                       # curves start further right


def test_narrow_terminal_stacks_the_panes():
    boxes = layout(40, SIDE_BY_SIDE_COLUMNS - 1)
    assert boxes["curves"][1] == 0                      # both start at the left
    assert boxes["curves"][0] > boxes["slice"][0]       # curves sit below


@pytest.mark.parametrize("rows,cols", [(6, 20), (24, 80), (60, 200), (1, 1)])
def test_layout_stays_inside_the_terminal(rows, cols):
    boxes = layout(rows, cols)
    for top, left, height, width in boxes.values():
        assert top >= 0 and left >= 0
        assert height >= 1 and width >= 1
        assert top + height <= max(rows, 6)
        assert left + width <= max(cols, 20)


# -- time -----------------------------------------------------------------
def test_stepping_moves_one_frame(app):
    app.handle_key("right")
    assert app.state.timestep == 1
    app.handle_key("h")
    assert app.state.timestep == 0


def test_stepping_is_clamped_at_both_ends(app):
    app.handle_key("left")
    assert app.state.timestep == 0
    app.handle_key("end")
    last = len(app.field.times) - 1
    assert app.state.timestep == last
    app.handle_key("right")
    assert app.state.timestep == last


def test_home_and_end(app):
    app.handle_key("end")
    app.handle_key("home")
    assert app.state.timestep == 0


def test_play_toggles_and_advance_stops_at_the_end(app):
    app.handle_key(" ")
    assert app.playing
    for _ in range(len(app.field.times) + 5):
        app.advance()
    assert app.state.timestep == len(app.field.times) - 1
    assert not app.playing          # it stops rather than looping


def test_advance_does_nothing_while_paused(app):
    app.advance()
    assert app.state.timestep == 0


def test_speed_is_clamped(app):
    for _ in range(20):
        app.handle_key("+")
    fastest = app.interval
    for _ in range(20):
        app.handle_key("-")
    assert app.interval > fastest


# -- what is shown ---------------------------------------------------------
def test_scale_keys(app):
    app.handle_key("s")
    assert app.state.scale_mode == "step"
    app.handle_key("g")
    assert app.state.scale_mode == "global"


def test_bracket_keys_change_field_and_clamp(app):
    app.handle_key("]")
    assert app.state.field_index == 1
    app.handle_key("]")
    assert app.state.field_index == 1          # only two fields
    app.handle_key("[")
    app.handle_key("[")
    assert app.state.field_index == 0


def test_changing_field_keeps_the_time_step_valid(app):
    app.handle_key("end")
    app.select_field(1)
    assert app.state.timestep <= len(app.field.times) - 1


def test_picker_keys_ask_for_a_list(app):
    assert app.handle_key("f") == "fields"
    assert app.handle_key("c") == "curves"
    assert len(app.field_choices()) == 2
    assert len(app.curve_choices()) == 2


def test_set_curves_sorts_and_deduplicates(app):
    app.set_curves([1, 0, 1])
    assert app.state.curves == (0, 1)


def test_quit_and_help(app):
    assert app.handle_key("q") == "quit"
    assert app.handle_key("?") is None
    assert app.showing_help
    app.handle_key("l")                        # any other key closes it again
    assert not app.showing_help


# -- text it produces -------------------------------------------------------
def test_title_names_the_field_and_the_time(app):
    app.handle_key("right")
    title = app.title()
    assert "fake" in title
    assert "step 2/" in title


def test_status_reports_the_scale_and_the_curves(app):
    app.handle_key("g")
    status = app.status()
    assert "whole run" in status
    assert "curves: 1" in status


def test_slice_and_curve_text_are_drawable(app):
    assert any(line.strip() for line in app.slice_text(20, 60))
    assert any(line.strip() for line in app.curves_text(20, 60))


def test_no_curves_selected_says_so(app):
    app.set_curves([])
    assert "press c" in " ".join(app.curves_text(20, 60))


def test_a_simulation_without_slices(monkeypatch, fake_series):
    monkeypatch.setattr(tui, "fields_of", lambda sim: [])
    monkeypatch.setattr(tui, "build_series", lambda sim: fake_series)
    app = Interactive(FakeSim())
    assert app.field is None
    assert "no slice output" in app.title()
    assert "no slice" in " ".join(app.slice_text(20, 60)).lower()
    app.handle_key("right")                    # must not raise
    app.advance()
