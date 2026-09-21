"""Front ends for looking at an FDS simulation.

Three of them share one core:

* :mod:`~fdsreader.explorer.notebook` -- widgets for Jupyter, ``explore()``
* :mod:`~fdsreader.explorer.cli` -- a terminal application, ``fdsreader-explorer-cli``
* a desktop front end may follow; it would sit beside these two

The shared parts are :mod:`~fdsreader.explorer.data` (reading devices, HRR and slices,
numpy only), :mod:`~fdsreader.explorer.plots` (matplotlib) and
:mod:`~fdsreader.explorer.render` (characters in a terminal).

Importing this package pulls in nothing beyond numpy. The names that need matplotlib or
ipywidgets are resolved on first use, so::

    from fdsreader.explorer.cli import main        # numpy only
    from fdsreader.explorer import explore         # needs the notebook extras
"""

from .data import (
    axis_label as axis_label,
)
from .data import (
    build_series as build_series,
)
from .data import (
    color_limits as color_limits,
)
from .data import (
    device_label as device_label,
)
from .data import (
    device_time as device_time,
)
from .data import (
    hrr_units as hrr_units,
)
from .data import (
    list_devices as list_devices,
)
from .data import (
    load_device_data as load_device_data,
)
from .data import (
    prepare_slice as prepare_slice,
)
from .data import (
    quantity_of as quantity_of,
)
from .data import (
    slice_label as slice_label,
)
from .data import (
    unit_of as unit_of,
)

#: Names that live in a module with heavier requirements, resolved on first access.
_LAZY = {
    "NotebookExplorer": "notebook",
    "SimulationBrowser": "notebook",
    "explore": "notebook",
    "DIVERGING_CMAPS": "plots",
    "SEQUENTIAL_CMAPS": "plots",
    "plot_device": "plots",
    "plot_series": "plots",
    "plot_slice": "plots",
    "RAMP": "render",
    "series_lines": "render",
    "slice_lines": "render",
}


def __getattr__(name):
    """Import the front-end modules only when something from them is asked for."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = sorted(_LAZY) + [
    "axis_label",
    "build_series",
    "color_limits",
    "device_label",
    "device_time",
    "hrr_units",
    "list_devices",
    "load_device_data",
    "prepare_slice",
    "quantity_of",
    "slice_label",
    "unit_of",
]
