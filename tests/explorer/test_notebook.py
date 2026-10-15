"""The Jupyter front end, driven through its widgets without a browser or kernel.

Plots go to static Output widgets (interactive_canvas=False) on the Agg backend. Uses the stored case steckler_data
(devices, an HRR file and one slice) and slcf_masks_fds611 (several slices, no devices).
"""

import shutil

import pytest
from fds_cases import CASES_DIR

pytest.importorskip("ipywidgets")
plt = pytest.importorskip("matplotlib.pyplot")

from fdsreader.explorer import notebook  # noqa: E402
from fdsreader.explorer.notebook import NotebookExplorer, SimulationBrowser, explore  # noqa: E402

# plt.show() into an Output widget has nothing to show on the Agg backend
pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")


@pytest.fixture(autouse=True)
def _agg_and_close():
    plt.switch_backend("Agg")
    yield
    plt.close("all")


@pytest.fixture
def steckler(tmp_path):
    # A copy, as the explorer may write or remove a pickle cache next to the data
    return shutil.copytree(CASES_DIR / "steckler_data", tmp_path / "steckler")


@pytest.fixture
def explorer(steckler):
    return NotebookExplorer(path=steckler, interactive_canvas=False)


def test_loading_fills_curves_slices_and_time_bar(explorer):
    assert explorer.sim is not None
    assert "Steckler" in explorer.summary.value and "1 slices" in explorer.summary.value
    assert explorer.series_select.value == (0,)
    assert len(explorer.selection) == 1
    assert [label for label, _ in explorer.slice_dropdown.options] == [explorer.field.label]
    assert explorer.time_slider.max == len(explorer.field.times) - 1
    assert not explorer.time_slider.disabled
    assert "step 1 of" in explorer.time_readout.value


def test_the_time_bar_moves_slice_and_curves(explorer):
    explorer.time_slider.value = 3
    assert "step 4 of" in explorer.time_readout.value
    assert explorer.current_time == pytest.approx(explorer.field.times[3])


def test_render_modes_show_the_level_slider_for_contours(explorer):
    explorer.render_mode.value = "filled"
    assert explorer.level_slider.layout.display == "flex"
    explorer.level_slider.value = 5
    explorer.render_mode.value = "lines"
    explorer.render_mode.value = "image"
    assert explorer.level_slider.layout.display == "none"


def test_manual_scale_starts_from_the_global_limits(explorer):
    explorer.scale_mode.value = "manual"
    assert explorer.scale_manual.layout.display == "flex"
    low, high, _ = explorer.field.value_range()
    assert (explorer.scale_min.value, explorer.scale_max.value) == (round(low, 4), round(high, 4))
    explorer.scale_max.value = explorer.scale_min.value - 1  # an inverted range is reported, not drawn as is
    assert explorer.scale_note.value
    explorer.scale_mode.value = "step"
    assert explorer.scale_manual.layout.display == "none"


def test_colour_map_and_several_curves(explorer):
    explorer.cmap_select.value = "viridis"
    explorer.series_select.value = tuple(range(min(3, len(explorer.series))))
    assert len(explorer.selection) == min(3, len(explorer.series))
    explorer.series_select.value = ()
    assert explorer.selection == []


def test_switching_slices(tmp_path):
    case = shutil.copytree(CASES_DIR / "slcf_masks_fds611", tmp_path / "masks")
    explorer = NotebookExplorer(path=case, interactive_canvas=False)
    assert len(explorer.slice_dropdown.options) == 8  # the planar slices; the 3D ones are left out
    explorer.slice_dropdown.value = 3
    assert explorer.field is explorer.fields[3]
    assert explorer.time_slider.max == len(explorer.fields[3].times) - 1


def test_a_simulation_with_nothing_to_show(simulation_with_nothing):
    explorer = NotebookExplorer(path=simulation_with_nothing, interactive_canvas=False)
    assert "no device" in explorer.summary.value
    assert "no slice" in explorer.time_readout.value
    assert explorer.field is None and explorer.current_time is None
    assert explorer.slice_dropdown.disabled and explorer.series_select.disabled


def test_explore_returns_the_explorer(steckler):
    assert isinstance(explore(steckler, interactive_canvas=False), NotebookExplorer)


def test_the_browser_finds_one_or_several_simulations(tmp_path, steckler):
    browser = SimulationBrowser(start_dir=tmp_path)
    assert browser.load_button.disabled and "No" in browser.status.value
    assert [label for label, _ in browser.folders.options] == ["steckler/"]

    browser.folders.value = str(steckler)  # selecting a folder opens it
    assert browser.current == steckler.resolve()
    assert not browser.load_button.disabled and browser.smv_picker.layout.display == "none"

    shutil.copy(steckler / "Steckler_010.smv", steckler / "copy.smv")
    browser.go_to(steckler)
    assert browser.smv_picker.layout.display == "flex" and "2</b> simulations" in browser.status.value


def test_the_browser_navigates_and_reports_problems(tmp_path, steckler):
    loaded = []
    browser = SimulationBrowser(start_dir=steckler, on_load=loaded.append)
    browser.path_box.value = ".."  # relative to the folder on screen
    assert browser.current == tmp_path.resolve()
    browser.go_to(tmp_path / "missing")
    assert "Not a directory" in browser.status.value
    browser.go_to(steckler)
    browser.load()
    assert loaded == [browser.sim] and "Loaded" in browser.status.value

    browser.go_to(tmp_path)  # the picker still points into the previous folder
    browser.smv_picker.options = [("old", str(steckler / "Steckler_010.smv"))]
    browser.load()
    assert "No simulation selected" in browser.status.value


def test_the_browser_reports_a_simulation_it_cannot_read(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "b.smv").write_text("GRID\n not a grid\n")
    browser = SimulationBrowser(start_dir=broken)
    browser.load()
    assert browser.sim is None and "Could not load" in browser.status.value


def test_backend_selection_without_a_shell():
    assert notebook._select_backend(False) is False
    assert notebook._select_backend(None) is False
    with pytest.raises(RuntimeError):
        notebook._select_backend(True)


def test_drawing_on_a_long_lived_figure(fake_field):
    # The ipympl path redraws one figure in place instead of rendering into an Output widget
    fig = plt.figure()
    with notebook._figure_surface(fig, None, (4, 3)) as surface:
        assert surface is fig
        surface.add_subplot(111)
    notebook._clear_surface(fig, None)
    assert not fig.axes
