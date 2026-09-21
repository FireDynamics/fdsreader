"""The terminal rendering is pinned to its exact output.

Text is easy to compare, so these read as pictures of what the CLI prints; a change in
layout shows up as a readable diff rather than a number that no longer matches.
"""

import numpy as np

from fdsreader.explorer.render import block_mean, series_lines, shape_for, slice_lines


def ink(lines):
    """How many characters of a drawing are not blank."""
    return sum(ch not in " |" for line in lines for ch in line)


def test_block_mean_averages_rather_than_samples():
    values = np.arange(16, dtype=float).reshape(4, 4)
    assert block_mean(values, 2, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]


def test_block_mean_handles_upsampling_without_empty_blocks():
    out = block_mean(np.array([[1.0, 2.0]]), 3, 4)
    assert out.shape == (3, 4)
    assert np.isfinite(out).all()


def test_shape_for_keeps_the_physical_aspect_ratio(fake_field):
    # the field is 4 m wide and 8 m tall, and a character cell is twice as tall as wide
    assert shape_for(fake_field, 20, 200) == (20, 20)


def test_shape_for_fits_a_narrow_terminal(fake_field):
    rows, cols = shape_for(fake_field, 40, 12)
    assert cols <= 12
    assert rows >= 4


def test_slice_lines(fake_field):
    low, high, _ = fake_field.value_range()
    assert slice_lines(fake_field, 1, low, high, height=10, width=40) == [
        "   8.00 |          ",
        "        |          ",
        "        |          ",
        "        |          ",
        "   4.44 |   .....  ",
        "        |  .:+++:. ",
        "        |  .:===:. ",
        "        |    ...   ",
        "        |          ",
        "   0.00 |          ",
        "        +----------",
        "         -2       2   y [m]",
        "         scale 20 … 471.6   ramp ' .:-=+*#%@'   grid 10x10",
    ]


def test_slice_lines_moves_with_time(fake_field):
    low, high, _ = fake_field.value_range()
    first = slice_lines(fake_field, 0, low, high, height=10, width=40)[:10]
    last = slice_lines(fake_field, 2, low, high, height=10, width=40)[:10]
    assert first != last
    # the blob rises, so its ink sits lower in the first frame than in the last
    weight = lambda rows: sum(i for i, row in enumerate(rows) if row.strip("| ")) 
    assert weight(first) > weight(last)  # row 0 is the top of the picture


def test_slice_lines_follows_the_colour_limits(fake_field):
    wide = slice_lines(fake_field, 1, 0.0, 5000.0, height=6, width=20)[:6]
    tight = slice_lines(fake_field, 1, 20.0, 60.0, height=6, width=20)[:6]
    assert ink(wide) < ink(tight)
    assert "@" in "".join(tight)
    assert "@" not in "".join(wide)


def test_series_lines_shared_unit(fake_series):
    assert series_lines(fake_series, height=6, width=24, cursor_time=2.0) == [
        "        84 |            |         oo",
        "      71.2 |            |       oo  ",
        "      58.4 |            |    oo ****",
        "      45.6 |            *oooo**     ",
        "      32.8 |     ***ooo o           ",
        "        20 |oooo ooo    |           ",
        "          +------------------------",
        "           0                      4   t [s]",
        "  *  DEVC  A — TEMPERATURE [C]",
        "  o  DEVC  B — TEMPERATURE [C]",
    ]


def test_series_lines_without_a_cursor_has_no_rule(fake_series):
    assert "|" not in "".join(
        line.split("|", 1)[1] for line in
        series_lines(fake_series, height=6, width=24)[:6]
    )


def test_series_lines_mixed_units_say_so(fake_series):
    mixed = [fake_series[0], dict(fake_series[1], unit="kW", quantity="HRR")]
    lines = series_lines(mixed, height=5, width=20)
    assert "scaled to its own range" in lines[-3]
    assert lines[-1].startswith("  y-axis is relative")


def test_series_lines_with_nothing_selected():
    assert series_lines([]) == ["(nothing selected)"]
