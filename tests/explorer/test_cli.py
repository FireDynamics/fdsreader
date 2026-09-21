"""The command line front end, driven the way a user drives it."""

import json

import pytest

from fdsreader.explorer.cli import main


def run(capsys, *argv):
    """Run the CLI and return (exit status, stdout, stderr)."""
    status = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_overview(tiny_case, capsys):
    status, out, _ = run(capsys, tiny_case)
    assert status == 0
    assert "tiny" in out
    assert "devices (DEVC)    2" in out
    assert "HRR quantities    1" in out
    assert "slices (SLCF)     0" in out
    assert "time              0 … 10 s" in out
    assert "TEMPERATURE                     2  C" in out


def test_list_names_what_can_be_asked_for(tiny_case, capsys):
    _, out, _ = run(capsys, tiny_case, "--list")
    assert "slices:\n  (none)" in out
    assert "--curve TC_1" in out
    assert "--curve HRR" in out


def test_curve_is_drawn_with_axes_and_a_legend(tiny_case, capsys):
    _, out, _ = run(capsys, tiny_case, "--curve", "TC_1", "--width", "20", "--height", "12")
    lines = out.splitlines()
    # TC_1 rises linearly from 20 to 100 C, so the top row is labelled with the maximum
    # and carries ink only at the right-hand end
    assert lines[0].startswith("       100 |")
    assert lines[0].rstrip().endswith("*")
    assert lines[0].split("|", 1)[1].startswith(" ")
    assert lines[-4].startswith("        20 |*")   # and the bottom row starts at t = 0
    assert lines[-3].startswith("          +---")  # then the axis rule
    assert "t [s]" in lines[-2]
    assert lines[-1] == "  *  DEVC  TC_1 — TEMPERATURE [C]"


def test_two_curves_share_one_scale_when_units_match(tiny_case, capsys):
    _, out, _ = run(capsys, tiny_case, "--curve", "TC_1", "--curve", "TC_2")
    assert "scaled to its own range" not in out
    assert out.rstrip().endswith("o  DEVC  TC_2 — TEMPERATURE [C]")


def test_mixed_units_are_flagged(tiny_case, capsys):
    _, out, _ = run(capsys, tiny_case, "--curve", "TC_1", "--curve", "HRR")
    assert "y-axis is relative" in out


def test_cursor_is_drawn_at_the_requested_time(tiny_case, capsys):
    _, without, _ = run(capsys, tiny_case, "--curve", "TC_1", "--width", "20")
    _, with_cursor, _ = run(capsys, tiny_case, "--curve", "TC_1", "--width", "20",
                            "--time", "5")
    assert with_cursor.count("|") > without.count("|")


def test_json_is_machine_readable(tiny_case, capsys):
    status, out, _ = run(capsys, tiny_case, "--json")
    assert status == 0
    payload = json.loads(out)
    assert payload["chid"] == "tiny"
    assert payload["devices"] == 2
    assert payload["slices"] == []
    names = {curve["name"] for curve in payload["curves"]}
    assert names == {"TC_1", "TC_2", "HRR"}
    tc1 = next(c for c in payload["curves"] if c["name"] == "TC_1")
    assert tc1["min"] == pytest.approx(20.0)
    assert tc1["max"] == pytest.approx(100.0)
    assert tc1["unit"] == "C"


def test_json_includes_the_selected_curves(tiny_case, capsys):
    _, out, _ = run(capsys, tiny_case, "--json", "--curve", "HRR")
    payload = json.loads(out)
    assert [s["name"] for s in payload["selected"]] == ["HRR"]
    assert len(payload["selected"][0]["values"]) == 21


def test_unknown_curve_is_reported(tiny_case, capsys):
    with pytest.raises(SystemExit) as raised:
        run(capsys, tiny_case, "--curve", "NOPE")
    assert "no curve called 'NOPE'" in str(raised.value)


def test_asking_for_a_slice_when_there_are_none(tiny_case, capsys):
    status, _, err = run(capsys, tiny_case, "--slice", "0")
    assert status == 1
    assert "no slice output" in err


def test_bad_path_is_reported_on_stderr(tmp_path, capsys):
    status, out, err = run(capsys, tmp_path / "nowhere")
    assert status == 1
    assert out == ""
    assert "could not load" in err


@pytest.mark.parametrize("scale", ["sideways", "5", "10,5", "a,b"])
def test_bad_scale_is_rejected(tiny_case, capsys, scale):
    with pytest.raises(SystemExit):
        run(capsys, tiny_case, "--curve", "TC_1", "--scale", scale)


def test_caching_is_off_unless_asked_for(tiny_case, capsys):
    run(capsys, tiny_case)
    assert not list(tiny_case.glob("*.pickle"))
