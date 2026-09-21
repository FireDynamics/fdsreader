"""Terminal rendering backend: drawing slices and time series with characters.

Needs nothing but numpy, so the command line front end stays usable in an installation
that has neither matplotlib nor ipywidgets.
"""

import shutil

import numpy as np

#: Characters from empty to full. The default ramp is deliberately short: a terminal
#: without colour can only distinguish a handful of levels anyway.
RAMP = " .:-=+*#%@"

#: A character cell is about twice as tall as it is wide. Horizontal counts are scaled by
#: this so that a slice keeps the aspect ratio it has in metres.
CELL_ASPECT = 2.0


def terminal_size(fallback=(100, 30)):
    """Width and height of the terminal, falling back when it is not a tty."""
    size = shutil.get_terminal_size(fallback)
    return size.columns, size.lines


def block_mean(values, rows, cols):
    """Area-average a 2D array down to a character grid.

    Averaging rather than sampling keeps thin hot structures visible when a slice is much
    finer than the terminal.
    """
    row_edges = np.linspace(0, values.shape[0], rows + 1).astype(int)
    col_edges = np.linspace(0, values.shape[1], cols + 1).astype(int)
    out = np.empty((rows, cols))
    for i in range(rows):
        r0, r1 = row_edges[i], max(row_edges[i] + 1, row_edges[i + 1])
        for j in range(cols):
            c0, c1 = col_edges[j], max(col_edges[j] + 1, col_edges[j + 1])
            out[i, j] = np.nanmean(values[r0:r1, c0:c1])
    return out


def shape_for(field, height, width):
    """Character grid for a field that keeps its physical aspect ratio.

    Returns ``(rows, cols)`` fitting inside ``height`` x ``width`` characters.
    """
    horizontal, vertical = field.axes
    xs, ys = field.coords[horizontal], field.coords[vertical]
    span_h = float(xs[-1] - xs[0]) or 1.0
    span_v = float(ys[-1] - ys[0]) or 1.0

    rows = max(4, height)
    cols = int(round(span_h / span_v * rows * CELL_ASPECT))
    if cols > width:  # too wide for the terminal, so scale the other way round
        cols = max(4, width)
        rows = max(4, int(round(span_v / span_h * cols / CELL_ASPECT)))
    return rows, max(4, cols)


def slice_lines(field, timestep, vmin, vmax, height=26, width=None, ramp=RAMP, frame=None):
    """Render one time step of a 2D field as text lines, axes included."""
    if width is None:
        width = max(20, terminal_size()[0] - 10)

    horizontal, vertical = field.axes
    xs, ys = field.coords[horizontal], field.coords[vertical]
    rows, cols = shape_for(field, height, width)

    if frame is None:
        frame = field.frame(timestep)
    # data is (dim1, dim2); transpose so that rows run along the upright axis
    grid = block_mean(frame.T, rows, cols)

    span = (vmax - vmin) or 1.0
    index = np.clip(((grid - vmin) / span * (len(ramp) - 1)).round().astype(int), 0, len(ramp) - 1)

    lines = []
    for i in range(rows - 1, -1, -1):  # origin at the bottom
        if i % 5 == 0 or i == rows - 1:
            coord = ys[0] + (ys[-1] - ys[0]) * i / max(1, rows - 1)
            label = f"{coord:7.2f} "
        else:
            label = " " * 8
        lines.append(label + "|" + "".join(ramp[j] for j in index[i]))

    lines.append(" " * 8 + "+" + "-" * cols)
    left, right = f"{xs[0]:g}", f"{xs[-1]:g}"
    pad = max(1, cols - len(left) - len(right))
    lines.append(" " * 9 + left + " " * pad + right + f"   {horizontal} [m]")
    lines.append(" " * 9 + f"scale {vmin:.4g} … {vmax:.4g}   ramp '{ramp}'   grid {cols}x{rows}")
    return lines


def series_lines(series, height=16, width=None, cursor_time=None):
    """Render one or more time series as text lines.

    Several series share the plot; each gets its own marker, and the y-axis is only
    labelled with real numbers when they share a unit.
    """
    if not series:
        return ["(nothing selected)"]
    if width is None:
        width = max(30, terminal_size()[0] - 12)

    markers = "*o+x#~"
    t_start = min(float(s["times"][0]) for s in series)
    t_end = max(float(s["times"][-1]) for s in series)
    t_span = (t_end - t_start) or 1.0

    units = {s["unit"] for s in series}
    shared_scale = len(units) == 1
    if shared_scale:
        low = min(float(np.nanmin(s["values"])) for s in series)
        high = max(float(np.nanmax(s["values"])) for s in series)
    else:  # mixed units: each series is scaled to its own range, so only shapes compare
        low, high = 0.0, 1.0

    canvas = [[" "] * width for _ in range(height)]
    for n, entry in enumerate(series):
        values = np.asarray(entry["values"], dtype=float)
        if shared_scale:
            lo, hi = low, high
        else:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
        span = (hi - lo) or 1.0

        xs = np.clip(
            ((np.asarray(entry["times"], dtype=float) - t_start) / t_span * (width - 1)).round().astype(int),
            0,
            width - 1,
        )
        ys = np.clip(((values - lo) / span * (height - 1)).round().astype(int), 0, height - 1)
        mark = markers[n % len(markers)]
        for x, y in zip(xs, ys):
            canvas[y][x] = mark

    if cursor_time is not None:
        column = int(round((float(cursor_time) - t_start) / t_span * (width - 1)))
        if 0 <= column < width:
            for row in canvas:
                if row[column] == " ":
                    row[column] = "|"

    lines = []
    for r in range(height - 1, -1, -1):
        if shared_scale:
            label = f"{low + (high - low) * r / max(1, height - 1):10.4g} |"
        else:
            label = f"{r / max(1, height - 1):10.2f} |"
        lines.append(label + "".join(canvas[r]))

    lines.append(" " * 10 + "+" + "-" * width)
    left, right = f"{t_start:g}", f"{t_end:g}"
    pad = max(1, width - len(left) - len(right))
    lines.append(" " * 11 + left + " " * pad + right + "   t [s]")

    for n, entry in enumerate(series):
        note = "" if shared_scale else "  (scaled to its own range)"
        lines.append(f"  {markers[n % len(markers)]}  {entry['label']}{note}")
    if not shared_scale:
        lines.append("  y-axis is relative: the selected series do not share a unit")
    return lines
