"""FDS 6.8.0 to 6.10.1 write the hundredths of the Plot3D time in the .smv file without a leading zero, so
0.07 s is labelled 0.7 like 0.70 s. The time is taken from the file name instead (GitHub issue #99)."""

import pytest

from fdsreader.simulation import _plot3d_time


@pytest.mark.parametrize(
    "label, filename, expected",
    [
        ("0.7", "chid_1_0p07.q", 0.07),  # label written by FDS 6.8.0 to 6.10.1
        ("0.07", "chid_1_0p07.q", 0.07),  # label written by FDS 6.11
        ("0.70", "chid_1_0p70.q", 0.70),
        ("1.6", "chid_12_1p06.q", 1.06),
        ("120.0", "results/chid_3_120p00.q", 120.0),
        ("-1.5", "chid_1_-1p05.q", -1.05),
    ],
)
def test_plot3d_time_from_file_name(label, filename, expected):
    assert _plot3d_time(label, filename) == pytest.approx(expected)


def test_plot3d_time_falls_back_to_label():
    assert _plot3d_time("2.5", "renamed.q") == pytest.approx(2.5)
