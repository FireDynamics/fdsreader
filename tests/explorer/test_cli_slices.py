"""The command line front end on a case with a slice: drawing it as text, saving pictures and animations,
handing over to the interactive view, and running as ``python -m fdsreader.explorer``.

Uses a copy of the stored case steckler_data (devices, an HRR file and one slice).
"""

import gc
import json
import runpy
import shutil
import sys

import pytest
from fds_cases import CASES_DIR

import fdsreader.explorer
from fdsreader.explorer import cli, tui
from fdsreader.explorer.cli import main


@pytest.fixture
def steckler(tmp_path):
    return shutil.copytree(CASES_DIR / "steckler_data", tmp_path / "steckler")


def run(capsys, *argv):
    """Run the CLI and return (exit status, stdout, stderr)."""
    status = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_overview_and_list_count_the_slice(steckler, capsys):
    _, out, _ = run(capsys, steckler)
    assert "slices (SLCF)     1" in out
    _, out, _ = run(capsys, steckler, "--list")
    assert "--slice 0   TEMPERATURE [C]" in out


def test_a_slice_drawn_as_text(steckler, capsys):
    status, out, _ = run(capsys, steckler, "--slice", "0", "--time", "5", "--height", "10", "--width", "40")
    assert status == 0
    title = out.splitlines()[0]
    assert title.startswith("TEMPERATURE [C]") and "(step 6 of 101)" in title
    assert len(out.splitlines()) > 10


def test_render_slice_notes_limits_it_cannot_use(steckler):
    # The command line rejects such limits already, so the note is reached through the function
    fields = cli.fields_of(cli._load(steckler))
    state = cli.ExplorerState(scale_mode="manual", manual_limits=(5.0, 1.0))
    assert any(
        line.startswith("  note: min must be below max") for line in cli.render_slice(fields, state, 8, 30, " .#")
    )
    with pytest.raises(SystemExit, match="no slice output"):
        cli.render_slice([], state, 8, 30, " .#")


def test_a_slice_out_of_range(steckler, capsys):
    status, _, err = run(capsys, steckler, "--slice", "3")
    assert status == 1 and "out of range (0 … 0)" in err


def test_a_slice_and_a_curve_together(steckler, capsys):
    device = next(entry["name"] for entry in cli.build_series(cli._load(steckler)) if entry["source"] == "device")
    _, out, _ = run(capsys, steckler, "--slice", "0", "--curve", device, "--height", "12", "--width", "40")
    assert out.startswith("TEMPERATURE [C]") and device in out


def test_json_with_a_slice(steckler, capsys):
    _, out, _ = run(capsys, steckler, "--json", "--slice", "0", "--time", "2")
    data = json.loads(out)
    assert data["chid"] == "Steckler_010"


@pytest.mark.parametrize("suffix", [".png", ".pdf"])
def test_save_a_picture_of_slice_and_curve(steckler, tmp_path, capsys, suffix):
    pytest.importorskip("matplotlib")
    target = tmp_path / f"view{suffix}"
    status, out, _ = run(capsys, steckler, "--slice", "0", "--curve", "HRR", "--save", target, "--render", "filled")
    assert status == 0 and out.strip() == f"wrote {target}"
    assert target.stat().st_size > 0


def test_save_a_picture_of_curves_only(tiny_case, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    target = tmp_path / "curves.png"
    run(capsys, tiny_case, "--curve", "TC_1", "--save", target)
    assert target.stat().st_size > 0


def test_save_an_animation(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    masks = shutil.copytree(CASES_DIR / "slcf_masks_fds611", tmp_path / "masks")  # three frames, so it is quick
    target = tmp_path / "slice.gif"
    _, out, _ = run(capsys, masks, "--slice", "0", "--save", target)
    assert "(3 frames)" in out and target.stat().st_size > 0


def test_an_animation_needs_a_slice(tiny_case, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    with pytest.raises(SystemExit, match="needs a slice"):
        run(capsys, tiny_case, "--save", tmp_path / "x.gif")


@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")
def test_an_animation_that_cannot_be_written(tmp_path, capsys, monkeypatch):
    animation = pytest.importorskip("matplotlib.animation")
    plt = pytest.importorskip("matplotlib.pyplot")

    def fail(*args, **kwargs):
        raise RuntimeError("no ffmpeg")

    monkeypatch.setattr(animation.Animation, "save", fail)
    masks = shutil.copytree(CASES_DIR / "slcf_masks_fds611", tmp_path / "masks")
    with pytest.raises(SystemExit, match="could not write"):
        run(capsys, masks, "--slice", "0", "--save", tmp_path / "x.mp4")
    assert not plt.get_fignums()  # the figure is closed although writing failed
    gc.collect()  # the unsaved animation warns when it is collected, which belongs to this test


def test_interactive_hands_the_state_to_the_terminal_view(steckler, capsys, monkeypatch):
    seen = {}
    monkeypatch.setattr(tui, "run", lambda sim, state: seen.update(sim=sim, state=state))
    status, _, _ = run(capsys, steckler, "-i", "--curve", "HRR", "--curve", "Q_RADI")
    assert status == 0
    assert seen["state"].scale_mode == "global"  # frames stay comparable while stepping
    assert seen["state"].curves == (0, 1)


def test_interactive_without_curses(steckler, capsys, monkeypatch):
    def missing(sim, state):
        raise ModuleNotFoundError("the interactive explorer needs the curses module")

    monkeypatch.setattr(tui, "run", missing)
    status, _, err = run(capsys, steckler, "-i")
    assert status == 1 and "curses" in err


def test_save_without_matplotlib(steckler, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # makes the import fail
    with pytest.raises(SystemExit, match="needs matplotlib"):
        cli.save_figure([], cli.ExplorerState(), [], tmp_path / "x.png", 10, 10)


def test_running_the_package(steckler, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fdsreader-explore", str(steckler)])
    monkeypatch.delitem(sys.modules, "fdsreader.explorer.__main__", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("fdsreader.explorer", run_name="__main__")
    assert exit_info.value.code == 0
    assert "Steckler_010" in capsys.readouterr().out


def test_the_package_exports_the_front_ends_lazily():
    assert "NotebookExplorer" in dir(fdsreader.explorer)
    assert fdsreader.explorer.RAMP
    with pytest.raises(AttributeError):
        _ = fdsreader.explorer.no_such_thing
