"""The curses side of the interactive front end, on a fake window that records what is written and plays back
keys, so no terminal is needed."""

import curses

import pytest

from fdsreader.explorer import tui
from fdsreader.explorer.render import RAMP
from fdsreader.explorer.tui import Interactive


class FakeSim:
    chid = "fake"
    root_path = "."


class FakeWindow:
    """Enough of a curses window for the explorer: a size, written text and a queue of keys."""

    def __init__(self, rows=30, cols=120, keys=()):
        self.rows, self.cols = rows, cols
        self.keys = list(keys)
        self.written = []
        self.timeouts = []

    def getmaxyx(self):
        return self.rows, self.cols

    def addnstr(self, row, col, text, n, attr=0):
        if (row, col + len(text[:n])) == (self.rows - 1, self.cols):
            raise curses.error  # curses raises when the bottom-right cell is written
        self.written.append((row, col, text[:n], attr))

    def getch(self):
        return self.keys.pop(0)

    def text(self):
        return "\n".join(text for _, _, text, _ in self.written)

    def erase(self):
        self.written.clear()

    def refresh(self):
        pass

    def keypad(self, flag):
        pass

    def timeout(self, ms):
        self.timeouts.append(ms)


@pytest.fixture
def app(monkeypatch, fake_field, fake_series):
    monkeypatch.setattr(tui, "fields_of", lambda sim: [fake_field, fake_field])
    monkeypatch.setattr(tui, "build_series", lambda sim: fake_series)
    return Interactive(FakeSim())


def keys(*names):
    return [ord(k) if isinstance(k, str) else k for k in names]


def test_key_names():
    assert tui._key_name(curses.KEY_LEFT) == "left"
    assert tui._key_name(curses.KEY_NPAGE) == "pgdn"
    assert tui._key_name(10) == tui._key_name(13) == "enter"
    assert tui._key_name(27) == "escape"
    assert tui._key_name(ord("q")) == "q"
    assert tui._key_name(100000) == ""


def test_put_clips_to_the_window():
    window = FakeWindow(rows=5, cols=10)
    tui._put(window, 9, 0, "below")
    tui._put(window, 0, 10, "right of it")
    tui._put(window, 1, 4, "far too long a line")
    tui._put(window, 2, 0, "abcdef", width=3)
    tui._put(window, 4, 9, "x")  # the bottom-right cell raises inside curses
    assert [text for _, _, text, _ in window.written] == ["far to", "abc"]


def test_ramped_lines_colour_each_cell():
    window = FakeWindow()
    attrs = list(range(100, 100 + len(RAMP)))
    tui._draw_ramped(window, 0, 0, RAMP[0] + RAMP[-1], attrs, 10)
    assert [attr for *_, attr in window.written] == [0, attrs[-1]]  # blank cells stay uncoloured
    tui._draw_ramped(window, 1, 0, "plain", None, 10)
    assert window.written[-1][2] == "plain"


@pytest.mark.parametrize("n_colours", [8, 256])
def test_colour_pairs(monkeypatch, n_colours):
    monkeypatch.setattr(curses, "has_colors", lambda: True)
    monkeypatch.setattr(curses, "start_color", lambda: None)
    monkeypatch.setattr(curses, "use_default_colors", lambda: (_ for _ in ()).throw(curses.error()))
    monkeypatch.setattr(curses, "COLORS", n_colours, raising=False)
    pairs = []

    def init_pair(i, fg, bg):
        if i == 2:
            raise curses.error  # a pair the terminal refuses
        pairs.append(fg)

    monkeypatch.setattr(curses, "init_pair", init_pair)
    monkeypatch.setattr(curses, "color_pair", lambda i: 1000 + i)
    attrs = tui._init_colors()
    assert len(attrs) == len(RAMP) and attrs[1] == 0
    assert pairs[-1] == (tui._HEAT_256 if n_colours >= 256 else tui._HEAT_8)[len(RAMP) - 1]


def test_no_colours(monkeypatch):
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    assert tui._init_colors() is None


def test_pick_one():
    window = FakeWindow(keys=keys("j", curses.KEY_DOWN, "k", 10))
    assert tui._pick(window, "Slice", ["a", "b", "c"], [0], False) == [1]
    assert "enter to choose" in window.text()


def test_pick_several_and_cancel():
    window = FakeWindow(keys=keys(" ", curses.KEY_END, " ", curses.KEY_HOME, " ", 10))
    # untick a, tick c, tick a again
    assert tui._pick(window, "Curves", ["a", "b", "c"], [0], True) == [0, 2]
    assert tui._pick(FakeWindow(keys=[27]), "Curves", ["a"], [], True) is None
    assert tui._pick(FakeWindow(keys=keys("q")), "Curves", ["a"], [], True) is None
    assert tui._pick(FakeWindow(), "Curves", [], [], True) is None


def test_pick_scrolls_a_long_list():
    window = FakeWindow(rows=8, keys=[curses.KEY_NPAGE, curses.KEY_NPAGE, curses.KEY_PPAGE, 10])
    choices = [f"item {i}" for i in range(30)]
    assert tui._pick(window, "Slice", choices, [], False) == [4]
    assert "5/30" in window.text()


def test_draw_the_screen_and_the_help(app):
    window = FakeWindow()
    tui._draw(window, app, None)
    assert "fake" in window.text() and "? for help" in window.text()
    app.handle_key("?")
    tui._draw(window, app, None)
    assert "any key to go back" in window.text()


def test_run_until_quit(app, monkeypatch):
    window = FakeWindow(
        keys=keys(
            "f",
            curses.KEY_DOWN,
            10,  # pick the second slice
            "c",
            " ",
            10,  # untick the first curve
            " ",
            -1,  # play: the wait runs out and the next frame is drawn
            curses.KEY_RESIZE,
            "q",
        )
    )
    monkeypatch.setattr(curses, "wrapper", lambda main: main(window))
    monkeypatch.setattr(curses, "curs_set", lambda visibility: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    monkeypatch.setattr(tui, "Interactive", lambda sim, state=None: app)
    assert tui.run(FakeSim()) is app
    assert app.state.field_index == 1
    assert app.state.curves == ()
    assert app.state.timestep == 1
    assert app.interval in window.timeouts
