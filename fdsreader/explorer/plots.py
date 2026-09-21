"""Matplotlib drawing for the explorer.

These are plain functions taking an axes, so they can be used on their own to build
figures for a report without any of the widget machinery.
"""

import matplotlib.pyplot as plt
import numpy as np

from .data import axis_label, device_label, device_time, quantity_of, unit_of

#: Colour maps offered for slices. "Auto" picks a one-sided map for quantities such as
#: temperature and a zero-centred one for signed quantities such as velocities.
SEQUENTIAL_CMAPS = [
    "inferno",
    "magma",
    "plasma",
    "viridis",
    "cividis",
    "turbo",
    "hot",
    "afmhot",
    "gist_heat",
    "Greys_r",
]
DIVERGING_CMAPS = ["RdBu_r", "coolwarm", "bwr", "seismic", "PuOr_r", "PRGn_r", "Spectral_r"]


def plot_series(series, ax=None, cursor_time=None):
    """Plot one or more time series against simulation time.

    Series whose unit differs from the first one are drawn against a second y-axis on the
    right, so a heat release rate in kW and a temperature in C can share a figure.
    ``cursor_time`` draws a marker at one instant, used to tie the curves to a slice.
    """
    if ax is None:
        _, ax = plt.subplots()

    if not series:
        ax.set_title("Nothing selected")
        ax.set_xlabel("Time [s]")
        return ax

    units = []
    for entry in series:
        if entry["unit"] not in units:
            units.append(entry["unit"])

    # One extra axis is readable; beyond that the remaining units share the right-hand one
    # and the axis label names them all.
    right = ax.twinx() if len(units) > 1 else None
    if right is not None:
        right.grid(False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    left_group, right_group, handles = [], [], []

    for i, entry in enumerate(series):
        on_left = right is None or entry["unit"] == units[0]
        target = ax if on_left else right
        (left_group if on_left else right_group).append(entry)
        (line,) = target.plot(
            entry["times"],
            entry["values"],
            lw=1.5,
            color=colors[i % len(colors)],
            label=entry["label"],
        )
        handles.append(line)

    ax.set_ylabel(axis_label(left_group))
    if right_group:
        right.set_ylabel(axis_label(right_group))

    single = series[0] if len(series) == 1 else None

    if single is not None and single["device"] is not None:
        device, times = single["device"], single["times"]
        # FDS also writes the initial state of every device at t=0; that is not an event,
        # so it would otherwise put a spurious marker on every single plot.
        for act_time, state in getattr(device, "activation_times", []):
            if not len(times) or act_time <= times[0]:
                continue
            ax.axvline(act_time, color="tab:red", ls="--", lw=1, alpha=0.8)
            ax.annotate(
                "activated" if state else "deactivated",
                xy=(act_time, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )

    if cursor_time is not None:
        ax.axvline(cursor_time, color="0.35", lw=1.2, alpha=0.7)

    ax.set_xlabel("Time [s]")
    ax.margins(x=0.01)

    if single is not None:
        title = single["label"]
        if single["device"] is not None:
            x, y, z = single["device"].position
            title += f"   ·   position ({x:g}, {y:g}, {z:g}) m"
        # The range goes on a second title line rather than floating inside the axes,
        # where it would sit on top of either the title or the data.
        finite = single["values"][np.isfinite(single["values"])]
        if finite.size:
            title += f"\nmin {finite.min():.4g}   ·   max {finite.max():.4g}   ·   final {finite[-1]:.4g}"
        ax.set_title(title, fontsize=10)
    else:
        ax.set_title(f"{len(series)} series", fontsize=10)
        ax.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="best", framealpha=0.85)

    return ax


def plot_device(sim, device, ax=None, cursor_time=None):
    """Plot a single device against simulation time."""
    return plot_series(
        [
            {
                "label": device_label(device),
                "name": device.id,
                "times": device_time(sim, device),
                "values": np.asarray(device.data),
                "quantity": quantity_of(device),
                "unit": unit_of(device),
                "source": "device",
                "device": device,
            }
        ],
        ax=ax,
        cursor_time=cursor_time,
    )


def plot_slice(field, timestep, ax=None, vmin=None, vmax=None, cmap=None, render="image", n_levels=12, frame=None):
    """Draw one time step of a 2D field.

    ``vmin``/``vmax`` override the field's own range, which is what the per-step and
    manual scaling modes use. ``render`` is ``"image"`` (one pixel per cell), ``"filled"``
    (filled contours) or ``"lines"`` (labelled contour lines). Pass ``frame`` to reuse an
    array that has already been read.
    """
    if ax is None:
        _, ax = plt.subplots()
    low, high, _ = field.value_range()
    vmin = low if vmin is None else vmin
    vmax = high if vmax is None else vmax
    cmap = field.cmap if cmap is None else cmap
    if frame is None:
        frame = field.frame(timestep)

    horizontal, vertical = field.axes
    x, y = field.coords[horizontal], field.coords[vertical]

    if render == "image":
        mappable = ax.imshow(
            frame.T,  # (dim1, dim2) -> rows must run along the upright axis
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",  # show the cells as computed, do not smooth them away
        )
    else:
        # Contours are drawn on fixed levels spanning the colour limits, so that changing
        # the scaling mode moves the contours with it. "both" keeps values outside the
        # range coloured rather than blank.
        levels = np.linspace(vmin, vmax, max(2, n_levels) + 1)
        if render == "filled":
            mappable = ax.contourf(x, y, frame.T, levels=levels, cmap=cmap, extend="both")
        else:
            mappable = ax.contour(x, y, frame.T, levels=levels, cmap=cmap, linewidths=1.0, vmin=vmin, vmax=vmax)
            ax.clabel(mappable, inline=True, fontsize=7, fmt="%.4g")
        ax.set_aspect("equal")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])

    bar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label(f"{field.quantity} [{field.unit}]" if field.unit else field.quantity)

    ax.set_xlabel(f"{horizontal} [m]")
    ax.set_ylabel(f"{vertical} [m]")
    ax.set_title(f"{field.label}   ·   t = {field.times[timestep]:.2f} s", fontsize=10)
    ax.grid(False)
    return ax
