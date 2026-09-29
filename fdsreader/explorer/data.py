"""Reading the parts of a simulation the explorer shows.

Only depends on numpy, so it can be used without the plotting and widget extras.
"""

import csv
from pathlib import Path

import numpy as np

TIME_DEVICE_ID = "Time"


def list_devices(sim):
    """All real devices of a simulation as a flat list.

    The ``Time`` pseudo-device that fdsreader adds for the device time column is dropped.
    Devices sharing an ID are stored as a list, and are flattened here.
    """
    devices = []
    for entry in sim.devices:
        group = entry if isinstance(entry, list) else [entry]
        for device in group:
            if device.id != TIME_DEVICE_ID:
                devices.append(device)
    return devices


def device_values(device):
    """A device's samples, or ``None`` when FDS wrote no usable data for it.

    A DEVC section can be named in the .smv while the CSV beside it is missing, empty or
    just a header -- a run that was cut short, or one whose devices never wrote. fdsreader
    raises a different error for each of those cases, and none of them should take a front
    end down, so they are all treated as "nothing to plot", as is a device holding a
    single value rather than a series.
    """
    try:
        values = np.asarray(device.data)
    except Exception:
        return None

    if values.ndim == 0:
        # A DEVC *line* -- FDS writes <chid>_line.csv and fdsreader stores one scalar per
        # point along the line -- is a profile through space, not a series through time,
        # so there is nothing here to plot against a time axis.
        return None
    return values


def load_device_data(devices):
    """Force the device data to be read.

    fdsreader reads the ``_devc.csv`` lazily and only fills in ``device.unit`` at that
    point, so units would be missing from labels otherwise. Reading one device loads the
    whole file, hence the single access.
    """
    for device in devices:
        device_values(device)
        break


def device_time(sim, device, values=None):
    """Time axis in seconds matching the device's samples.

    ``values`` is the already-read data, if the caller has it. Returns an empty array for
    a device with nothing to plot.
    """
    if values is None:
        values = device_values(device)
    if values is None:
        return np.array([])

    if TIME_DEVICE_ID in sim.devices:
        times = device_values(sim.devices[TIME_DEVICE_ID])
        if times is not None and len(times) == len(values):
            return times
    # Fall back to the sample index if the simulation has no device time column.
    return np.arange(len(values))


def unit_of(device):
    """Unit string, cleaned up: fdsreader keeps the trailing newline of the CSV header."""
    return (device.unit or "").strip().strip('"')


def quantity_of(device):
    quantity = device.quantity
    return (quantity.name if quantity is not None else "").strip()


def device_label(device, duplicate_ids=()):
    """Human-readable entry, e.g. ``TC_1 - TEMPERATURE [C]``.

    Devices whose ID is not unique are additionally identified by their position.
    """
    parts = [device.id]
    quantity = quantity_of(device)
    if quantity:
        parts.append(f"— {quantity}")
    unit = unit_of(device)
    if unit:
        parts.append(f"[{unit}]")
    if device.id in duplicate_ids:
        x, y, z = device.position
        parts.append(f"@ ({x:g}, {y:g}, {z:g})")
    return " ".join(parts)


def slice_label(slc, index=None):
    """Entry for a slice, e.g. ``TEMPERATURE [C] - x = 0 m``.

    Slices usually have an empty ``id`` -- FDS only names them when ``SLCF ID`` is given --
    so they are identified by quantity and by the plane they cut.
    """
    quantity = slc.quantity.name.strip()
    unit = (slc.quantity.unit or "").strip()
    head = f"{quantity} [{unit}]" if unit else quantity
    if slc.orientation in (1, 2, 3):
        dim = "xyz"[slc.orientation - 1]
        head += f" — {dim} = {slc.extent[dim][0]:g} m"
    else:
        head += " — 3D"
    if slc.id.strip():
        head += f" ({slc.id.strip()})"
    elif index is not None:
        head += f"  #{index}"
    return head


def color_limits(values, diverging=None):
    """Colour scale for an array: its real minimum and maximum.

    ``diverging`` is decided once for a whole slice and handed back in for single frames,
    so the scale cannot flip between centred and one-sided while stepping through time.
    """
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high):
        low, high = 0.0, 1.0
    if diverging is None:
        diverging = low < 0 < high
    if diverging:  # velocities read better on a scale centred at zero
        bound = max(abs(low), abs(high)) or 1.0
        low, high = -bound, bound
    if low == high:  # a frame of constant value still needs a finite range
        low, high = low - 0.5, high + 0.5
    return low, high, diverging


def prepare_slice(slc):
    """Read a slice into one global array and work out how to draw it.

    ``to_global`` stitches the per-mesh subslices together and returns the data with the
    slice-normal axis squeezed out, so the array is ``(n_t, dim1, dim2)`` with dim1/dim2 in
    x, y, z order. Returns ``None`` for anything that is not a drawable 2D slice.
    """
    data, coords = slc.to_global(return_coordinates=True)

    # A cell-centered slice sitting exactly on a mesh border has a valid representation on
    # either side, and fdsreader hands back both. The first one is used here.
    if isinstance(data, tuple):
        data, coords = data[0], coords[0]

    spanned = [d for d in ("x", "y", "z") if len(coords[d]) > 1]
    if data.size == 0 or len(spanned) != 2:
        return None
    horizontal, vertical = spanned  # x, y, z order puts the upright axis second

    low, high, diverging = color_limits(data)
    return {
        "slice": slc,
        "data": data,
        "coords": coords,
        "axes": (horizontal, vertical),
        "vmin": low,
        "vmax": high,
        "diverging": diverging,
        "cmap": "RdBu_r" if diverging else "inferno",
    }


def hrr_units(sim):
    """Units of the HRR columns.

    ``sim.hrr`` is a plain dict of name -> values; fdsreader skips the unit line of the
    ``_hrr.csv`` header when reading it, so it is read again here.
    """
    for path in sorted(Path(sim.root_path).glob("*_hrr.csv")):
        try:
            with open(path) as infile:
                units = next(csv.reader([infile.readline()]))
                names = next(csv.reader([infile.readline()]))
        except (OSError, StopIteration):
            return {}
        return {n.strip(): u.strip() for n, u in zip(names, units)}
    return {}


def build_series(sim):
    """Every plottable time series of a simulation: the devices and the HRR quantities.

    Both are reduced to the same shape so they can be picked from one list and drawn on one
    figure. Each series keeps its own time axis, because the HRR file and the device file
    are not necessarily written at the same interval.
    """
    series = []

    devices = list_devices(sim)
    load_device_data(devices)  # so that units are known when building the labels
    seen, duplicate_ids = set(), set()
    for device in devices:
        (duplicate_ids if device.id in seen else seen).add(device.id)

    for device in devices:
        values = device_values(device)
        if values is None or not len(values):
            continue  # named in the .smv, but nothing usable was written for it
        series.append(
            {
                "label": "DEVC  " + device_label(device, duplicate_ids),
                "name": device.id,
                "times": device_time(sim, device, values),
                "values": values,
                "quantity": quantity_of(device),
                "unit": unit_of(device),
                "source": "device",
                "device": device,
            }
        )

    # `sim.hrr` only exists when the simulation wrote an HRR file, and reading it can
    # fail the same way the device CSV can.
    try:
        hrr = getattr(sim, "hrr", None)
    except Exception:
        hrr = None
    if hrr and "Time" in hrr:
        units = hrr_units(sim)
        times = np.asarray(hrr["Time"])
        for name, values in hrr.items():
            if name == "Time":
                continue
            values = np.asarray(values)
            if not len(values) or len(values) != len(times):
                continue
            unit = units.get(name, "").strip()
            series.append(
                {
                    "label": f"HRR   {name}" + (f" [{unit}]" if unit else ""),
                    "name": name,
                    "times": times,
                    "values": np.asarray(values),
                    "quantity": name,
                    "unit": unit,
                    "source": "hrr",
                    "device": None,
                }
            )

    return series


def axis_label(group):
    """Y-axis label for a group of series sharing an axis."""
    quantities = {s["quantity"] for s in group}
    units = sorted({s["unit"] for s in group if s["unit"]})
    head = quantities.pop() if len(quantities) == 1 else "Value"
    return f"{head} [{', '.join(units)}]" if units else head
