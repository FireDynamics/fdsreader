"""Interactive terminal front end.

A keyboard-driven view of one simulation: a slice on one side, curves on the other, and
a time bar driving both. What is shown is chosen inside the application rather than on
the command line.

Built on :mod:`curses` from the standard library, so it adds no dependency. That module
is not part of the standard library on Windows, where ``pip install windows-curses``
provides it.

The drawing and the key handling are kept apart from curses itself -- :func:`layout` and
:meth:`Interactive.handle_key` are ordinary functions -- so the behaviour can be tested
without a terminal.
"""

from .data import build_series
from .fields import fields_of
from .render import RAMP, series_lines, slice_lines
from .state import ExplorerState

#: Width at which the slice and the curves stop fitting side by side and stack instead.
SIDE_BY_SIDE_COLUMNS = 100

#: How long a frame is held while playing, in milliseconds, slowest to fastest.
PLAY_INTERVALS = (500, 250, 120, 60, 30)

HELP = [
    "",
    "  Time",
    "    left / right        one step        h / l      one step",
    "    page up / page down ten steps       < / >      one per cent",
    "    home / end          first / last    space      play / pause",
    "    + / -               play speed",
    "",
    "  Choosing what to show",
    "    f                   pick the slice from a list",
    "    c                   pick the curves from a list",
    "    [ / ]               previous / next slice",
    "",
    "  Colour scale",
    "    g                   the whole run",
    "    s                   the step on screen",
    "",
    "  ?                     this help          q      quit",
    "",
]


def layout(rows, cols):
    """Where everything goes, for a terminal of ``rows`` by ``cols`` characters.

    Returns a dict of ``(top, left, height, width)`` boxes. The slice and the curves sit
    side by side when there is room and stack when there is not, which mirrors what the
    notebook front end does with its panels.
    """
    rows, cols = max(rows, 6), max(cols, 20)
    header, footer = 2, 2
    body_top = header
    body_height = max(1, rows - header - footer)

    if cols >= SIDE_BY_SIDE_COLUMNS:
        left_width = max(20, cols // 2)
        boxes = {
            "slice": (body_top, 0, body_height, left_width),
            "curves": (body_top, left_width + 1, body_height, cols - left_width - 1),
        }
    else:
        slice_height = max(1, (body_height * 3) // 5)
        boxes = {
            "slice": (body_top, 0, slice_height, cols),
            "curves": (body_top + slice_height, 0, body_height - slice_height, cols),
        }

    boxes["header"] = (0, 0, header, cols)
    boxes["footer"] = (rows - footer, 0, footer, cols)
    return boxes


class Interactive:
    """The application: what is selected, and what each key does to it.

    Holds no curses objects, so it can be driven directly in a test.
    """

    def __init__(self, sim, state=None):
        self.sim = sim
        self.fields = fields_of(sim)
        self.series = build_series(sim)
        self.state = state or ExplorerState()
        self.state.curves = (0,) if self.series else ()
        self.playing = False
        self.speed = 2  # index into PLAY_INTERVALS
        self.message = ""
        self.showing_help = False

    # -- what is on screen -----------------------------------------------
    @property
    def field(self):
        return self.state.field(self.fields)

    @property
    def interval(self):
        return PLAY_INTERVALS[self.speed]

    def title(self):
        field = self.field
        if field is None:
            return f"{self.sim.chid}   (no slice output)"
        step = self.state.clamp(self.fields)
        return f"{self.sim.chid}   {field.label}   t = {field.times[step]:.3f} s   step {step + 1}/{len(field.times)}"

    def status(self):
        if self.message:
            return self.message
        scale = {"global": "whole run", "step": "this step", "manual": "manual"}
        bits = [f"scale: {scale.get(self.state.scale_mode, self.state.scale_mode)}"]
        bits.append(f"curves: {len(self.state.curves)}")
        if self.playing:
            bits.append(f"playing ({self.interval} ms)")
        bits.append("? for help, q to quit")
        return "   ".join(bits)

    # -- drawing ----------------------------------------------------------
    def slice_text(self, height, width):
        field = self.field
        if field is None:
            return ["", "  This simulation has no slice (SLCF) output."]
        step = self.state.clamp(self.fields)
        frame = field.frame(step)
        vmin, vmax, warning = self.state.limits(field, frame=frame)
        lines = slice_lines(field, step, vmin, vmax, height=max(4, height - 3), width=max(10, width - 2), frame=frame)
        if warning:
            lines.append(f"  note: {warning}")
        return lines

    def curves_text(self, height, width):
        chosen = self.state.selected_series(self.series)
        if not chosen:
            return ["", "  No curves selected — press c to choose some."]
        return series_lines(
            chosen,
            height=max(4, height - 4),
            width=max(10, width - 13),
            cursor_time=self.state.time(self.fields),
        )

    # -- keys --------------------------------------------------------------
    def step_by(self, delta):
        field = self.field
        if field is None:
            return
        last = len(field.times) - 1
        self.state.timestep = max(0, min(self.state.timestep + delta, last))

    def handle_key(self, key):
        """Act on one key. Returns ``"quit"``, ``"fields"``, ``"curves"`` or ``None``.

        ``key`` is a character, or one of the ``"left"``, ``"right"``, ``"pgup"``,
        ``"pgdn"``, ``"home"``, ``"end"`` names, so tests need no curses constants.
        """
        self.message = ""
        if self.showing_help and key not in ("?",):
            self.showing_help = False
            return None

        field = self.field
        per_cent = max(1, len(field.times) // 100) if field is not None else 1

        if key in ("q", "Q"):
            return "quit"
        if key == "?":
            self.showing_help = not self.showing_help
            return None
        if key in ("left", "h"):
            self.step_by(-1)
        elif key in ("right", "l"):
            self.step_by(1)
        elif key == "pgup":
            self.step_by(-10)
        elif key == "pgdn":
            self.step_by(10)
        elif key == "<":
            self.step_by(-per_cent)
        elif key == ">":
            self.step_by(per_cent)
        elif key == "home":
            self.state.timestep = 0
        elif key == "end":
            self.state.timestep = (len(field.times) - 1) if field is not None else 0
        elif key == " ":
            self.playing = not self.playing
        elif key == "+":
            self.speed = min(self.speed + 1, len(PLAY_INTERVALS) - 1)
        elif key == "-":
            self.speed = max(self.speed - 1, 0)
        elif key == "g":
            self.state.scale_mode = "global"
        elif key == "s":
            self.state.scale_mode = "step"
        elif key == "[":
            self.select_field(self.state.field_index - 1)
        elif key == "]":
            self.select_field(self.state.field_index + 1)
        elif key == "f":
            return "fields"
        elif key == "c":
            return "curves"
        return None

    def advance(self):
        """One frame of playback; stops at the end rather than looping."""
        field = self.field
        if field is None or not self.playing:
            return
        if self.state.timestep >= len(field.times) - 1:
            self.playing = False
        else:
            self.state.timestep += 1

    def select_field(self, index):
        if not self.fields:
            return
        previous = self.field
        index = max(0, min(index, len(self.fields) - 1))
        if previous is not None and index != self.state.field_index:
            previous.clear_cache()  # keep one field's data in memory at a time
        self.state.field_index = index
        self.state.clamp(self.fields)

    def set_curves(self, indices):
        self.state.curves = tuple(sorted(set(indices)))

    # -- list contents for the pickers --------------------------------------
    def field_choices(self):
        return [f.label for f in self.fields]

    def curve_choices(self):
        return [entry["label"] for entry in self.series]


# ---------------------------------------------------------------- curses layer

#: 256-colour approximation of the sequential map the other front ends use, from the
#: coldest ramp character to the hottest.
_HEAT_256 = (233, 17, 54, 90, 126, 160, 196, 202, 208, 220, 227)
#: The same idea for a terminal with only the eight basic colours.
_HEAT_8 = (0, 4, 4, 5, 5, 1, 1, 3, 3, 7, 7)


def _key_name(key):
    """Turn a curses key code into the names :meth:`Interactive.handle_key` expects."""
    import curses

    names = {
        curses.KEY_LEFT: "left",
        curses.KEY_RIGHT: "right",
        curses.KEY_UP: "up",
        curses.KEY_DOWN: "down",
        curses.KEY_PPAGE: "pgup",
        curses.KEY_NPAGE: "pgdn",
        curses.KEY_HOME: "home",
        curses.KEY_END: "end",
        10: "enter",
        13: "enter",
        27: "escape",
    }
    if key in names:
        return names[key]
    if 0 <= key < 256:
        return chr(key)
    return ""


def _init_colors():
    """Colour pairs for the ramp, or ``None`` when the terminal has no colour."""
    import curses

    if not curses.has_colors():
        return None
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    palette = _HEAT_256 if curses.COLORS >= 256 else _HEAT_8
    attrs = []
    for i, colour in enumerate(palette[: len(RAMP)], start=1):
        try:
            curses.init_pair(i, colour, -1)
            attrs.append(curses.color_pair(i))
        except curses.error:
            attrs.append(0)
    return attrs


def _put(window, row, col, text, attr=0, width=None):
    """Write text, clipped to the window; curses errors on the last cell are ignored."""
    import curses

    height, cols = window.getmaxyx()
    if not (0 <= row < height) or col >= cols:
        return
    room = (cols - col) if width is None else min(width, cols - col)
    if room <= 0:
        return
    try:
        window.addnstr(row, col, text, room, attr)
    except curses.error:
        pass  # writing the bottom-right cell always raises


def _draw_ramped(window, row, col, text, ramp_attrs, width):
    """Draw a line of slice art, colouring each cell by how far up the ramp it is."""
    if ramp_attrs is None:
        _put(window, row, col, text, 0, width)
        return
    for offset, char in enumerate(text[:width]):
        index = RAMP.find(char)
        attr = ramp_attrs[min(index, len(ramp_attrs) - 1)] if index > 0 else 0
        _put(window, row, col + offset, char, attr, 1)


def _pick(stdscr, title, choices, selected, multiple):
    """A modal list. Returns the chosen indices, or ``None`` when cancelled."""
    import curses

    if not choices:
        return None
    chosen = set(selected)
    cursor = min(selected) if selected else 0

    while True:
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()
        _put(stdscr, 0, 0, title, curses.A_BOLD)
        hint = "space to tick, enter to accept, esc to cancel" if multiple else "enter to choose, esc to cancel"
        _put(stdscr, 1, 0, hint, curses.A_DIM)

        room = max(1, rows - 4)
        first = max(0, min(cursor - room // 2, len(choices) - room))
        for offset in range(min(room, len(choices) - first)):
            index = first + offset
            mark = "[x] " if index in chosen else "[ ] " if multiple else "    "
            attr = curses.A_REVERSE if index == cursor else 0
            _put(stdscr, 2 + offset, 0, f" {mark}{choices[index]}", attr, cols - 1)

        if len(choices) > room:
            _put(stdscr, rows - 1, 0, f"{cursor + 1}/{len(choices)}", curses.A_DIM)
        stdscr.refresh()

        key = _key_name(stdscr.getch())
        if key == "escape" or key in ("q", "Q"):
            return None
        if key in ("up", "k"):
            cursor = max(0, cursor - 1)
        elif key in ("down", "j"):
            cursor = min(len(choices) - 1, cursor + 1)
        elif key == "pgup":
            cursor = max(0, cursor - room)
        elif key == "pgdn":
            cursor = min(len(choices) - 1, cursor + room)
        elif key == "home":
            cursor = 0
        elif key == "end":
            cursor = len(choices) - 1
        elif key == " " and multiple:
            chosen.symmetric_difference_update({cursor})
        elif key == "enter":
            if multiple:
                return sorted(chosen)
            return [cursor]


def _draw(stdscr, app, ramp_attrs):
    import curses

    stdscr.erase()
    rows, cols = stdscr.getmaxyx()

    if app.showing_help:
        for row, line in enumerate(HELP[: rows - 1]):
            heading = line.strip() and not line.startswith("    ")
            _put(stdscr, row, 0, line, curses.A_BOLD if heading else 0)
        _put(stdscr, rows - 1, 0, " any key to go back", curses.A_DIM)
        stdscr.refresh()
        return

    boxes = layout(rows, cols)
    top, left, _, width = boxes["header"]
    _put(stdscr, top, left, app.title(), curses.A_BOLD, width)
    _put(stdscr, top + 1, left, "─" * width, curses.A_DIM, width)

    top, left, height, width = boxes["slice"]
    for offset, line in enumerate(app.slice_text(height, width)[:height]):
        _draw_ramped(stdscr, top + offset, left, line, ramp_attrs, width)

    top, left, height, width = boxes["curves"]
    for offset, line in enumerate(app.curves_text(height, width)[:height]):
        _put(stdscr, top + offset, left, line, 0, width)

    top, left, _, width = boxes["footer"]
    _put(stdscr, top, left, "─" * width, curses.A_DIM, width)
    _put(stdscr, top + 1, left, " " + app.status(), curses.A_DIM, width)
    stdscr.refresh()


def run(sim, state=None):
    """Show the interactive explorer for ``sim`` until the user quits."""
    try:
        import curses
    except ModuleNotFoundError as exc:  # pragma: no cover - Windows without the shim
        raise ModuleNotFoundError(
            "the interactive explorer needs the curses module, which is not part of the "
            "standard library on Windows. Install it with: pip install windows-curses"
        ) from exc

    app = Interactive(sim, state=state)

    def main(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        ramp_attrs = _init_colors()

        while True:
            _draw(stdscr, app, ramp_attrs)
            stdscr.timeout(app.interval if app.playing else -1)
            key = stdscr.getch()

            if key == -1:  # the wait ran out, so play the next frame
                app.advance()
                continue
            if key == curses.KEY_RESIZE:
                continue

            action = app.handle_key(_key_name(key))
            if action == "quit":
                return
            if action == "fields":
                picked = _pick(stdscr, " Slice", app.field_choices(), [app.state.field_index], False)
                if picked:
                    app.select_field(picked[0])
            elif action == "curves":
                picked = _pick(stdscr, " Curves", app.curve_choices(), list(app.state.curves), True)
                if picked is not None:
                    app.set_curves(picked)

    curses.wrapper(main)
    return app
