"""A 2D field that can be drawn at one instant in time.

Every front end asks a field the same three things -- what are your coordinates, what did
you look like at time step *t*, and what is your value range -- so slices, and later
boundary data or a plane cut from 3D output, can all be drawn by the same code.

Frames are read one at a time. A slice file is a sequence of fixed-size records, so the
frame wanted can be seeked to directly instead of materialising the whole time series,
which for a production run would be gigabytes.
"""

import os

import numpy as np

import fdsreader.utils.fortran_data as fdtype

from .data import color_limits, slice_label


def _read_subslice_frame(subslice, timestep):
    """One time step of one sub-slice, without reading the rest of the file.

    Mirrors what fdsreader does when it loads the whole array, but seeks to a single
    record. Belongs in :mod:`fdsreader.slcf` as ``SubSlice.frame(t)`` eventually.
    """
    count = subslice.dimension.size(cell_centered=False)
    record = fdtype.combine(fdtype.FLOAT, fdtype.new((("f", count),)))
    path = os.path.join(subslice._parent_slice._root_path, subslice.filename)

    with open(path, "rb") as infile:
        infile.seek(subslice._offset + timestep * record.itemsize)
        data = fdtype.read(infile, record, 1)[0]

    frame = data[1].reshape(subslice.dimension.shape(cell_centered=False), order="F")
    if subslice.cell_centered:
        # Ignore ghost points on every axis, as the eager path does.
        frame = frame[(slice(1, None),) * frame.ndim]
    return frame


def _bnd_range(slc):
    """Value range of a slice from the ``.bnd`` files FDS writes beside the slice files.

    Saves reading any data at all for the global colour scale. The range covers every
    sub-slice, so where a cell-centered slice straddles a mesh border it can be slightly
    wider than the half that ends up on screen. Returns ``None`` when a file is missing.
    """
    low, high = np.inf, -np.inf
    for subslice in slc.subslices:
        path = os.path.join(slc._root_path, subslice.filename + ".bnd")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as infile:
                for line in infile:
                    parts = line.split()
                    if len(parts) >= 3:
                        low = min(low, float(parts[1]))
                        high = max(high, float(parts[2]))
        except (OSError, ValueError):
            return None
    return (float(low), float(high)) if np.isfinite(low) and np.isfinite(high) else None


def _dedup(values):
    """Sort and drop near-duplicates: FDS writes coordinates with limited precision."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.size:
        values = values[np.concatenate(([True], np.diff(values) > 2e-6))]
    return values


def _sides(slc, normal):
    """Sub-slices grouped by where they sit along the slice normal.

    A cell-centered slice lying exactly on a mesh border has no cells *at* the border, so
    FDS writes the cells on either side of it. Each side is a complete, equally valid
    picture of the slice; the eager path returns both and this keeps them apart.
    """
    groups = {}
    for subslice in slc.subslices:
        coord = subslice.get_coordinates()[normal]
        if not len(coord):
            continue
        groups.setdefault(round(float(coord[0]), 6), []).append(subslice)
    return [groups[key] for key in sorted(groups)]


class SliceField:
    """A 2D ``SLCF`` slice, stitched across meshes one time step at a time."""

    kind = "slice"

    def __init__(self, slc, index=None, side=0):
        self.source = slc
        self.index = index
        self.label = slice_label(slc, index)
        self.quantity = slc.quantity.name.strip()
        self.unit = (slc.quantity.unit or "").strip()
        self.times = np.asarray(slc.times, dtype=float)

        self.drawable = slc.orientation in (1, 2, 3)
        if not self.drawable:  # a 3D slice has no single plane to draw
            self.axes, self.coords, self._placement, self.shape = (), {}, [], (0, 0)
            return

        normal = "xyz"[slc.orientation - 1]
        horizontal, vertical = [d for d in ("x", "y", "z") if d != normal]
        self.axes = (horizontal, vertical)  # x, y, z order puts the upright axis second

        sides = _sides(slc, normal)
        subslices = sides[min(side, len(sides) - 1)] if sides else []

        # Cell coordinates decide where each sub-slice goes in the stitched grid ...
        cell = (
            {d: _dedup(np.concatenate([sub.get_coordinates()[d] for sub in subslices])) for d in (horizontal, vertical)}
            if subslices
            else {horizontal: np.array([]), vertical: np.array([])}
        )
        self.shape = (len(cell[horizontal]), len(cell[vertical]))

        # ... while the drawn axes span the extent the slice actually covers, so that a
        # cell-centered field is not drawn half a cell adrift.
        self.coords = {
            normal: np.array([float(slc.extent[normal][0])]),
            horizontal: np.linspace(*slc.extent[horizontal], self.shape[0]),
            vertical: np.linspace(*slc.extent[vertical], self.shape[1]),
        }

        self._placement = []
        for subslice in subslices:
            sub_coords = subslice.get_coordinates()
            sub_h, sub_v = sub_coords[horizontal], sub_coords[vertical]
            if not len(sub_h) or not len(sub_v):
                continue
            i0 = int(np.argmin(np.abs(cell[horizontal] - sub_h[0])))
            j0 = int(np.argmin(np.abs(cell[vertical] - sub_v[0])))
            self._placement.append((subslice, i0, len(sub_h), j0, len(sub_v)))

        self.drawable = self.shape[0] > 1 and self.shape[1] > 1
        self._range = None

    # -- data ------------------------------------------------------------
    def frame(self, timestep):
        """The stitched field at one time step, as ``(n_horizontal, n_vertical)``."""
        out = np.full(self.shape, np.nan, dtype=np.float32)
        written = np.zeros(self.shape, dtype=bool)

        for subslice, i0, ni, j0, nj in self._placement:
            data = _read_subslice_frame(subslice, timestep)
            i1, j1 = min(i0 + ni, self.shape[0]), min(j0 + nj, self.shape[1])
            patch = data[: i1 - i0, : j1 - j0]
            region = (slice(i0, i1), slice(j0, j1))
            # Where a cell-centered slice straddles a mesh border two sub-slices claim the
            # same cells; the first one wins, the way the eager path keeps one of them.
            out[region] = np.where(written[region], out[region], patch)
            written[region] = True
        return out

    def value_range(self):
        """``(vmin, vmax, diverging)`` over the whole time series.

        Read from the ``.bnd`` files when FDS wrote them, so no field data is touched.
        """
        if self._range is None:
            bounds = _bnd_range(self.source)
            if bounds is None:  # no .bnd files: fall back to scanning the frames
                low, high = np.inf, -np.inf
                for t in range(len(self.times)):
                    frame = self.frame(t)
                    low = min(low, float(np.nanmin(frame)))
                    high = max(high, float(np.nanmax(frame)))
                bounds = (low, high)
            self._range = color_limits(np.asarray(bounds, dtype=float))
        return self._range

    @property
    def cmap(self):
        return "RdBu_r" if self.value_range()[2] else "inferno"

    def clear_cache(self):
        self.source.clear_cache()


def fields_of(sim):
    """Every drawable 2D field of a simulation.

    Slices today. Boundary data and planes cut from 3D output would be appended here, and
    every front end would pick them up without changing.
    """
    fields = []
    for i, slc in enumerate(sim.slices):
        field = SliceField(slc, i)
        if field.drawable:
            fields.append(field)
    return fields
