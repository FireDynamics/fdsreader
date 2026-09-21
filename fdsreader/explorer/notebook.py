"""Jupyter front end: a directory browser, a time bar, a slice and a curve plot.

Importing this module requires the notebook extras::

    pip install "fdsreader[notebook]"
"""

try:
    import ipywidgets as widgets
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the installation
    raise ModuleNotFoundError(
        "fdsreader.explorer.notebook needs ipywidgets and matplotlib. Install them with:\n"
        '    pip install "fdsreader[notebook]"\n'
        "ipympl is optional and adds zooming and panning."
    ) from exc

from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import display

import fdsreader
from fdsreader import settings

from .data import build_series, list_devices, quantity_of, unit_of
from .fields import fields_of
from .plots import DIVERGING_CMAPS, SEQUENTIAL_CMAPS, plot_series, plot_slice
from .state import ExplorerState


def _select_backend(preference=None):
    """Switch matplotlib to ipympl when it is usable, otherwise to inline images.

    ``preference`` is ``None`` to auto-detect, ``True`` to insist on ipympl and ``False``
    to force inline. Auto-detection cannot see whether the *browser* has the matching
    front-end extension, so pass ``False`` when the plots come up as
    "Failed to load model class 'MPLCanvasModel'".
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:
        shell = None

    if preference is not False:
        # Probe without importing: importing ipympl switches the matplotlib backend as a
        # side effect, and that would stick even when the inline fall-back is taken.
        available = find_spec("ipympl") is not None
        if available and shell is not None:
            try:
                shell.run_line_magic("matplotlib", "widget")
                return True
            except Exception:
                if preference is True:
                    raise
        elif preference is True:
            raise RuntimeError(
                "interactive_canvas=True, but "
                + ("ipympl is not installed" if not available else "no IPython shell is running")
            )

    if shell is not None:
        try:
            shell.run_line_magic("matplotlib", "inline")
        except Exception:
            pass
    return False


@contextmanager
def _figure_surface(fig, view, figsize):
    """Hand out a blank figure to draw on, whichever backend is in use.

    With ipympl the one long-lived figure is cleared and redrawn in place, because its
    canvas is a widget that already sits in the layout. Without it, a throw-away figure is
    rendered into an ``Output`` widget and closed again -- inline figures made inside a
    widget callback are not cleaned up on their own and would otherwise pile up.
    """
    if fig is not None:
        fig.clear()
        yield fig
        fig.canvas.draw_idle()
    else:
        view.clear_output(wait=True)
        with view:
            throwaway = plt.figure(figsize=figsize)
            try:
                yield throwaway
                plt.show()
            finally:
                plt.close(throwaway)


def _clear_surface(fig, view):
    """Blank the plot area without drawing anything into it."""
    if fig is not None:
        fig.clear()
        fig.canvas.draw_idle()
    else:
        view.clear_output()


class SimulationBrowser:
    """Directory browser that loads the FDS simulation found in the selected folder."""

    def __init__(self, start_dir=None, on_load=None):
        self.on_load = on_load
        self.sim = None
        self.current = None

        self.path_box = widgets.Text(
            description="Path:",
            continuous_update=False,
            layout=widgets.Layout(flex="1 1 auto", width="auto", min_width="240px"),
            style={"description_width": "50px"},
        )
        self.up_button = widgets.Button(description="⬆ Up", layout=widgets.Layout(width="90px"))
        self.folders = widgets.Select(
            options=[],
            rows=10,
            layout=widgets.Layout(width="auto", min_width="220px"),
        )
        self.smv_picker = widgets.Dropdown(
            description=".smv:",
            layout=widgets.Layout(width="340px", display="none"),
            style={"description_width": "50px"},
        )
        self.load_button = widgets.Button(
            description="Load simulation",
            button_style="primary",
            disabled=True,
            layout=widgets.Layout(width="160px"),
        )
        self.status = widgets.HTML()

        self.up_button.on_click(lambda _: self.go_to(self.current.parent))
        self.load_button.on_click(lambda _: self.load())
        self.folders.observe(self._on_folder_selected, names="value")
        self.path_box.observe(self._on_path_typed, names="value")

        # The folder list and the status share a row so the browser stays shallow and
        # leaves the vertical space to the plots.
        self.widget = widgets.VBox(
            [
                widgets.HBox([self.path_box], layout=widgets.Layout(width="100%")),
                widgets.HBox(
                    [self.up_button, self.load_button, self.smv_picker],
                    layout=widgets.Layout(flex_flow="row wrap", align_items="center"),
                ),
                widgets.HBox(
                    [
                        widgets.Box([self.folders], layout=widgets.Layout(flex="1 1 280px", min_width="240px")),
                        widgets.Box(
                            [self.status],
                            layout=widgets.Layout(flex="1 1 300px", min_width="240px", padding="4px 0 0 14px"),
                        ),
                    ],
                    layout=widgets.Layout(width="100%", flex_flow="row wrap", align_items="flex-start"),
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )

        self.go_to(Path(start_dir) if start_dir is not None else Path.cwd())

    # -- navigation ------------------------------------------------------
    def _on_path_typed(self, change):
        if self.current is not None and change["new"] != str(self.current):
            self.go_to(change["new"])

    def _on_folder_selected(self, change):
        if change["new"]:
            self.go_to(Path(change["new"]))

    def go_to(self, directory):
        directory = Path(directory).expanduser()
        # A typed relative path should be relative to the folder on screen, not to the
        # directory the Jupyter kernel happens to have been started in.
        if not directory.is_absolute() and self.current is not None:
            directory = self.current / directory
        if not directory.is_dir():
            self.status.value = f"<span style='color:#c00'>Not a directory: {directory}</span>"
            return

        self.current = directory.resolve()
        self.path_box.value = str(self.current)

        try:
            subdirs = sorted(
                (p for p in self.current.iterdir() if p.is_dir() and not p.name.startswith(".")),
                key=lambda p: p.name.lower(),
            )
        except PermissionError:
            subdirs = []
            self.status.value = "<span style='color:#c00'>Permission denied.</span>"

        # Rebuild the list without re-triggering navigation.
        self.folders.unobserve(self._on_folder_selected, names="value")
        self.folders.options = [(p.name + "/", str(p)) for p in subdirs]
        self.folders.value = None
        self.folders.observe(self._on_folder_selected, names="value")

        self._scan_for_smv()

    def _scan_for_smv(self):
        smv_files = sorted(self.current.glob("*.smv"))

        if not smv_files:
            self.smv_picker.layout.display = "none"
            self.load_button.disabled = True
            self.status.value = (
                f"<span style='color:#888'>No <code>.smv</code> file in "
                f"<code>{self.current.name or self.current}</code> — keep browsing.</span>"
            )
            return

        self.load_button.disabled = False
        self.smv_picker.options = [(p.name, str(p)) for p in smv_files]
        # Assigning `options` does not select anything while the index is None.
        self.smv_picker.index = 0
        if len(smv_files) > 1:
            self.smv_picker.layout.display = "flex"
            self.status.value = (
                f"<b>{len(smv_files)}</b> simulations found — choose one, then click <i>Load simulation</i>."
            )
        else:
            self.smv_picker.layout.display = "none"
            self.status.value = f"Found <code>{smv_files[0].name}</code> — click <i>Load simulation</i>."

    # -- loading ---------------------------------------------------------
    def load(self):
        smv_path = self.smv_picker.value
        if smv_path is None or Path(smv_path).parent != self.current:
            self.status.value = "<span style='color:#c00'>No simulation selected in this folder.</span>"
            return

        self.status.value = f"Loading <code>{Path(smv_path).name}</code> …"
        try:
            self.sim = fdsreader.Simulation(smv_path)
        except Exception as exc:
            self.sim = None
            self.status.value = f"<span style='color:#c00'>Could not load: {exc}</span>"
            return

        devices = list_devices(self.sim)
        self.status.value = (
            f"<span style='color:#080'>Loaded <b>{self.sim.chid}</b> — "
            f"{len(devices)} device(s), {len(self.sim.meshes)} mesh(es).</span>"
        )
        if self.on_load is not None:
            self.on_load(self.sim)


class NotebookExplorer:
    """Interactive view of one FDS simulation at a time.

    Shows a directory browser, a time bar, a 2D slice and any number of device or HRR
    curves. The slice and the curves sit side by side on a wide window and wrap onto
    their own lines on a narrow one.

    :param path: Simulation to load straight away; if omitted, browse to one.
    :param start_dir: Where the directory browser opens. Defaults to the working directory.
    :param interactive_canvas: ``None`` to use ipympl when available, ``True`` to insist,
        ``False`` to force static inline images. Pass ``False`` if the plots come up as
        "Failed to load model class 'MPLCanvasModel'", which means the browser-side
        extension is missing.
    :param caching: Sets :data:`fdsreader.settings.ENABLE_CACHING`. Off by default:
        fdsreader caches a parsed simulation as a ``.pickle`` next to the data, which
        fails on a read-only directory such as a shared course folder. Note that such a
        directory must also be free of leftover ``.pickle`` files, since switching the
        cache off makes fdsreader try to delete them.
    """

    def __init__(
        self,
        path=None,
        start_dir=None,
        interactive_canvas=None,
        caching=False,
        slice_figsize=(6.5, 5.5),
        curve_figsize=(7.0, 5.5),
        panel_basis="560px",
        panel_min="360px",
    ):
        settings.ENABLE_CACHING = caching
        self.interactive_canvas = _select_backend(interactive_canvas)
        self.slice_figsize = slice_figsize
        self.curve_figsize = curve_figsize

        self.sim = None
        self.series = []
        self.devices = []
        self.fields = []
        self.state = ExplorerState()
        self._updating = False

        self._build_widgets(panel_basis, panel_min)
        if start_dir is None and path is not None:
            start_dir = Path(path).expanduser().parent if Path(path).is_file() else path
        self.browser = SimulationBrowser(start_dir=start_dir, on_load=self._on_simulation_loaded)

        self.widget = widgets.VBox(
            [
                self.browser.widget,
                self._rule(),
                self.summary,
                self._rule(),
                self.time_bar,
                self._rule(),
                widgets.HBox(
                    [self.slice_panel, self.curve_panel],
                    layout=widgets.Layout(width="100%", flex_flow="row wrap", align_items="flex-start"),
                ),
            ],
            layout=widgets.Layout(width="100%"),
        )

        if path is not None:
            self.load(path)

    # -- construction ----------------------------------------------------
    @staticmethod
    def _rule():
        return widgets.HTML("<hr style='margin:10px 0'>")

    def _build_widgets(self, panel_basis, panel_min):
        self.summary = widgets.HTML()

        if self.interactive_canvas:
            # With ipympl a figure is a live widget: create it once, put it in the layout
            # and redraw it in place. A new one per update would stack up canvases.
            with plt.ioff():
                self.slice_fig = plt.figure(figsize=self.slice_figsize)
                self.curve_fig = plt.figure(figsize=self.curve_figsize)
            for fig in (self.slice_fig, self.curve_fig):
                canvas = fig.canvas
                canvas.header_visible = False
                canvas.footer_visible = False
                canvas.layout.width = "100%"
                canvas.layout.height = "auto"
            self.slice_view, self.curve_view = self.slice_fig.canvas, self.curve_fig.canvas
        else:
            self.slice_fig = self.curve_fig = None
            self.slice_view = widgets.Output()
            self.curve_view = widgets.Output()

        self.series_select = widgets.SelectMultiple(
            description="Curves:",
            options=[],
            disabled=True,
            rows=9,
            layout=widgets.Layout(width="auto"),
            style={"description_width": "60px"},
        )
        self.series_hint = widgets.HTML(
            "<span style='color:#888;font-size:90%'>Devices (DEVC) and the quantities of "
            "the HRR file. Hold ⌘/Ctrl to add single entries, ⇧ for a run of them.</span>"
        )

        self.slice_dropdown = widgets.Dropdown(
            description="Slice:",
            options=[],
            disabled=True,
            layout=widgets.Layout(width="auto"),
            style={"description_width": "60px"},
        )
        self.scale_mode = widgets.ToggleButtons(
            options=[("Global", "global"), ("Per step", "step"), ("Manual", "manual")],
            value="global",
            tooltips=[
                "One scale for the whole time series",
                "Rescale to the time step on screen",
                "Type the limits yourself",
            ],
            layout=widgets.Layout(width="auto"),
            style={"button_width": "auto"},
        )
        self.scale_min = widgets.FloatText(
            description="min", layout=widgets.Layout(width="140px"), style={"description_width": "30px"}
        )
        self.scale_max = widgets.FloatText(
            description="max", layout=widgets.Layout(width="140px"), style={"description_width": "30px"}
        )
        self.scale_manual = widgets.HBox(
            [self.scale_min, self.scale_max], layout=widgets.Layout(display="none", flex_flow="row wrap")
        )
        self.scale_note = widgets.HTML()
        self.slice_note = widgets.HTML()
        self.cmap_select = widgets.Dropdown(
            description="Map:",
            value="auto",
            options=[("Auto", "auto")] + [(n, n) for n in SEQUENTIAL_CMAPS + DIVERGING_CMAPS],
            layout=widgets.Layout(width="190px"),
            style={"description_width": "36px"},
        )
        self.render_mode = widgets.ToggleButtons(
            options=[("Image", "image"), ("Filled", "filled"), ("Lines", "lines")],
            value="image",
            tooltips=["One pixel per cell, as computed", "Filled contours", "Labelled contour lines"],
            layout=widgets.Layout(width="auto"),
            style={"button_width": "auto"},
        )
        self.level_slider = widgets.IntSlider(
            description="Levels:",
            min=2,
            max=30,
            value=12,
            continuous_update=False,
            layout=widgets.Layout(width="230px", display="none"),
            style={"description_width": "50px"},
        )

        self.time_slider = widgets.IntSlider(
            description="Time:",
            min=0,
            max=0,
            value=0,
            disabled=True,
            continuous_update=False,
            readout=False,
            style={"description_width": "60px"},
            layout=widgets.Layout(flex="1 1 auto", width="auto", min_width="200px"),
        )
        self.time_play = widgets.Play(min=0, max=0, value=0, interval=120, disabled=True)
        self.time_readout = widgets.HTML(layout=widgets.Layout(padding="0 0 0 14px"))
        # The Play button drives the slider in the browser; the slider's observer draws.
        widgets.jslink((self.time_play, "value"), (self.time_slider, "value"))

        self.time_bar = widgets.HBox(
            [self.time_play, self.time_slider, self.time_readout],
            layout=widgets.Layout(width="100%", flex_flow="row wrap", align_items="center"),
        )

        def row(children):
            """A line of controls that wraps when the window is narrow."""
            return widgets.HBox(
                children,
                layout=widgets.Layout(width="100%", flex_flow="row wrap", align_items="center"),
            )

        def panel(children):
            """A column that grows into a wide window and wraps on a narrow one."""
            return widgets.VBox(
                children,
                layout=widgets.Layout(flex=f"1 1 {panel_basis}", min_width=panel_min, padding="0 8px 0 0"),
            )

        self.slice_panel = panel(
            [
                self.slice_dropdown,
                row(
                    [
                        widgets.HTML("<span style='padding-right:8px'><b>Colour</b></span>"),
                        self.scale_mode,
                        self.scale_manual,
                        self.cmap_select,
                    ]
                ),
                row(
                    [
                        widgets.HTML("<span style='padding-right:8px'><b>Draw</b></span>"),
                        self.render_mode,
                        self.level_slider,
                    ]
                ),
                self.scale_note,
                self.slice_note,
                self.slice_view,
            ]
        )
        self.curve_panel = panel([self.series_select, self.series_hint, self.curve_view])

        self.series_select.observe(self._refresh_curves, names="value")
        self.slice_dropdown.observe(self._on_slice_change, names="value")
        self.time_slider.observe(self._on_time_change, names="value")
        self.scale_mode.observe(self._on_scale_mode_change, names="value")
        self.scale_min.observe(self._refresh_slice, names="value")
        self.scale_max.observe(self._refresh_slice, names="value")
        self.cmap_select.observe(self._refresh_slice, names="value")
        self.render_mode.observe(self._on_render_change, names="value")
        self.level_slider.observe(self._refresh_slice, names="value")

    # -- public ----------------------------------------------------------
    def load(self, path):
        """Load a simulation directly, as clicking *Load simulation* would."""
        self.browser.go_to(Path(path).parent if Path(path).is_file() else path)
        self.browser.load()
        return self.sim

    @property
    def field(self):
        """The 2D field currently selected, or ``None``."""
        return self.state.field(self.fields)

    @property
    def current_time(self):
        """Simulation time currently shown by the time bar, or ``None`` without a field."""
        return self.state.time(self.fields)

    @property
    def selection(self):
        """The series currently ticked in the curve list."""
        return self.state.selected_series(self.series)

    def _ipython_display_(self):
        display(self.widget)

    # -- drawing ---------------------------------------------------------
    def _sync_state(self):
        """Copy the widget values into the shared state."""
        self.state.timestep = self.time_slider.value
        self.state.scale_mode = self.scale_mode.value
        self.state.manual_limits = (float(self.scale_min.value), float(self.scale_max.value))
        self.state.cmap = self.cmap_select.value
        self.state.render = self.render_mode.value
        self.state.n_levels = self.level_slider.value
        self.state.curves = tuple(self.series_select.value)

    def _refresh_slice(self, _change=None):
        if self._updating:
            return
        self._sync_state()
        field = self.field
        if field is None:
            self.time_readout.value = ""
            _clear_surface(self.slice_fig, self.slice_view)
            return

        timestep = self.state.clamp(self.fields)
        self.time_readout.value = (
            f"<code>t = {field.times[timestep]:.2f} s</code> "
            f"<span style='color:#888'>(step {timestep + 1} of {len(field.times)})</span>"
        )

        frame = field.frame(timestep)  # read once, drawn and possibly scaled from
        vmin, vmax, warning = self.state.limits(field, frame=frame)
        self.scale_note.value = f"<span style='color:#c00;font-size:90%'>{warning}</span>" if warning else ""

        with _figure_surface(self.slice_fig, self.slice_view, self.slice_figsize) as fig:
            ax = fig.add_subplot(111)
            plot_slice(
                field,
                timestep,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap=self.state.colormap(field),
                render=self.state.render,
                n_levels=self.state.n_levels,
                frame=frame,
            )
            fig.tight_layout()

    def _refresh_curves(self, _change=None):
        if self._updating:
            return
        self._sync_state()
        selected = self.selection
        if not selected:
            _clear_surface(self.curve_fig, self.curve_view)
            return
        with _figure_surface(self.curve_fig, self.curve_view, self.curve_figsize) as fig:
            ax = fig.add_subplot(111)
            plot_series(selected, ax=ax, cursor_time=self.current_time)
            fig.tight_layout()

    # -- callbacks -------------------------------------------------------
    def _on_time_change(self, _change=None):
        """The time bar moves the slice and the marker on the curve plot together."""
        if self._updating:
            return
        self._refresh_slice()
        self._refresh_curves()

    def _on_render_change(self, _change=None):
        """The level count only matters for the contour modes."""
        self.level_slider.layout.display = "none" if self.render_mode.value == "image" else "flex"
        self._refresh_slice()

    def _on_scale_mode_change(self, _change=None):
        manual = self.scale_mode.value == "manual"
        self.scale_manual.layout.display = "flex" if manual else "none"
        field = self.field
        if manual and field is not None and not self._updating:
            # Start from what is on screen, so the fields are a useful starting point.
            self._sync_state()
            self.state.scale_mode = "global"
            low, high, _ = self.state.limits(field)
            self._updating = True
            self.scale_min.value, self.scale_max.value = round(low, 4), round(high, 4)
            self._updating = False
        self._refresh_slice()

    def _on_slice_change(self, _change=None):
        if self._updating:
            return
        index = self.slice_dropdown.value
        if index is None:
            return

        previous = self.field
        if previous is not None:
            previous.clear_cache()  # keep only one field's data in memory

        self.state.field_index = index
        field = self.field
        if field is None:
            self.slice_note.value = "<i>Nothing to show.</i>"
            _clear_surface(self.slice_fig, self.slice_view)
            return
        self.slice_note.value = ""

        self._updating = True
        last = len(field.times) - 1
        self.time_slider.max = self.time_play.max = last
        self.time_slider.value = self.time_play.value = min(self.time_slider.value, last)
        self.time_slider.disabled = self.time_play.disabled = False
        self._updating = False

        self._on_time_change()

    # -- loading ---------------------------------------------------------
    def _setup_fields(self, sim):
        """Fill the field dropdown and the time bar for a freshly loaded simulation."""
        self.fields = fields_of(sim)
        self.state.field_index = 0
        self.state.timestep = 0

        self._updating = True
        self.slice_dropdown.options = [(f.label, i) for i, f in enumerate(self.fields)]
        self.slice_dropdown.disabled = not self.fields
        self.time_slider.disabled = self.time_play.disabled = not self.fields
        if not self.fields:
            self.time_slider.max = self.time_play.max = 0
        self._updating = False

        _clear_surface(self.slice_fig, self.slice_view)
        if not self.fields:
            self.slice_note.value = ""
            self.time_readout.value = "<i>This simulation has no slice (SLCF) output.</i>"
            return

        self._updating = True
        self.slice_dropdown.index = 0
        self._updating = False
        self._on_slice_change()

    def _on_simulation_loaded(self, sim):
        """Populate the curve list, the slices and the overview once a simulation is read."""
        self.sim = sim
        self.devices = list_devices(sim)
        self.series = build_series(sim)
        self.state.curves = (0,) if self.series else ()

        self._updating = True
        self.series_select.options = [(e["label"], i) for i, e in enumerate(self.series)]
        self.series_select.disabled = not self.series
        self.series_select.value = (0,) if self.series else ()
        self._updating = False

        if not self.series:
            self.summary.value = "<i>This simulation has no device (DEVC) or HRR output.</i>"
            _clear_surface(self.curve_fig, self.curve_view)
            self._setup_fields(sim)
            return

        self.summary.value = self._overview()
        self._setup_fields(sim)  # also positions the time bar
        self._refresh_curves()

    def _overview(self):
        """Short HTML summary: what is in this simulation, at a glance."""
        n_hrr = sum(1 for e in self.series if e["source"] == "hrr")
        counts = []
        if self.devices:
            counts.append(f"{len(self.devices)} devices")
        if n_hrr:
            counts.append(f"{n_hrr} HRR quantities")
        if self.fields:
            counts.append(f"{len(self.fields)} slices")

        # Devices and the HRR file are not necessarily written at the same interval, so
        # report the span covered by everything rather than one series' sample count.
        starts = [e["times"][0] for e in self.series if len(e["times"])]
        ends = [e["times"][-1] for e in self.series if len(e["times"])]
        if starts:
            counts.append(f"t = {min(starts):g} … {max(ends):g} s")

        groups = {}
        for device in self.devices:
            groups.setdefault(quantity_of(device) or "—", []).append(unit_of(device))
        if n_hrr:
            groups["HRR file"] = [e["unit"] for e in self.series if e["source"] == "hrr"]

        rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0'>{name}</td>"
            f"<td style='padding:2px 14px 2px 0'>{len(units)}</td>"
            f"<td style='padding:2px 0;color:#666'>"
            f"{', '.join(sorted(set(units) - {''})) or '—'}</td></tr>"
            for name, units in sorted(groups.items())
        )
        table = (
            f"""
          <table style="margin-top:6px;border-collapse:collapse">
            <tr style="text-align:left;color:#666">
              <th style='padding-right:14px'>Quantity</th>
              <th style='padding-right:14px'>#</th><th>Unit</th>
            </tr>{rows}
          </table>"""
            if rows
            else ""
        )

        return f"""
        <div style="font-size:90%">
          <b>{self.sim.chid}</b> &nbsp;·&nbsp; {" &nbsp;·&nbsp; ".join(counts)}{table}
        </div>"""


def explore(path=None, **kwargs):
    """Show the explorer and return it.

    ``explore()`` opens a directory browser; ``explore("path/to/case")`` loads that
    simulation straight away. Keyword arguments are passed to :class:`NotebookExplorer`.
    """
    explorer = NotebookExplorer(path=path, **kwargs)
    display(explorer.widget)
    return explorer
