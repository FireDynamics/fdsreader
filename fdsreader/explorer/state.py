"""What the user is currently looking at, independent of which front end shows it.

Keeping the selection *and the rules that act on it* here is what stops the front ends
drifting apart: the notebook and the command line ask the same object for the colour
limits, so "per step" means the same thing in both, and a third front end inherits the
behaviour rather than reimplementing it.
"""

from dataclasses import dataclass
from dataclasses import field as _dataclass_field
from typing import Tuple

import numpy as np

from .data import color_limits

SCALE_MODES = ("global", "step", "manual")
RENDER_MODES = ("image", "filled", "lines")


@dataclass
class ExplorerState:
    """The current selection. Front ends read and write it; nothing here draws."""

    field_index: int = 0
    timestep: int = 0
    scale_mode: str = "global"
    manual_limits: Tuple[float, float] | None = None
    cmap: str = "auto"
    render: str = "image"
    n_levels: int = 12
    curves: Tuple[int, ...] = _dataclass_field(default_factory=tuple)

    # -- selection -------------------------------------------------------
    def field(self, fields):
        """The selected field, or ``None`` when there is nothing to show."""
        if not fields:
            return None
        self.field_index = max(0, min(self.field_index, len(fields) - 1))
        return fields[self.field_index]

    def clamp(self, fields):
        """Keep the time step inside the selected field."""
        current = self.field(fields)
        if current is None:
            self.timestep = 0
        else:
            self.timestep = max(0, min(self.timestep, len(current.times) - 1))
        return self.timestep

    def time(self, fields):
        """Simulation time currently selected, in seconds, or ``None``."""
        current = self.field(fields)
        if current is None or not len(current.times):
            return None
        return float(current.times[self.clamp(fields)])

    def seek(self, fields, seconds):
        """Move to the time step nearest to ``seconds``."""
        current = self.field(fields)
        if current is None or not len(current.times):
            return 0
        self.timestep = int(np.argmin(np.abs(current.times - float(seconds))))
        return self.timestep

    # -- drawing rules ---------------------------------------------------
    def limits(self, field, frame=None):
        """Colour limits for the frame on screen.

        Returns ``(vmin, vmax, warning)``; *warning* is a message when the request could
        not be honoured and the global scale was used instead.
        """
        low, high, diverging = field.value_range()

        if self.scale_mode == "manual":
            if self.manual_limits is not None:
                lo, hi = (float(v) for v in self.manual_limits)
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    return lo, hi, None
            return low, high, "min must be below max — showing the global scale"

        if self.scale_mode == "step":
            if frame is None:
                frame = field.frame(self.timestep)
            lo, hi, _ = color_limits(frame, diverging)
            return lo, hi, None

        return low, high, None

    def colormap(self, field):
        """The colour map to draw with, resolving ``"auto"`` against the field."""
        return field.cmap if self.cmap == "auto" else self.cmap

    def selected_series(self, series):
        """The series ticked in the curve list, dropping any stale indices."""
        return [series[i] for i in self.curves if 0 <= i < len(series)]
