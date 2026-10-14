import math
import os
import sys
import warnings
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np

import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings
from fdsreader.fds_classes import Mesh
from fdsreader.utils import NumpyArrayMixin, Quantity
from fdsreader.utils.data import resample_mesh_axis

_HANDLED_FUNCTIONS = {}

# Defaults FDS scales HRRPUV and temperature Smoke3D output to (HRRPUV_MAX_SMV, TEMP_MIN_SMV,
# TEMP_MAX_SMV on &DUMP). FDS 6.10 and later also write the values used to the .smv file.
DEFAULT_HRRPUV_MINMAX = (0.0, 1200.0)  # kW/m3
DEFAULT_TEMP_MINMAX = (20.0, 2000.0)  # C
KINDS = ("gas", "hrrpuv", "temperature")

_raw_codes_warned = False
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _caller_stacklevel() -> int:
    """The stacklevel for warnings.warn that points at the first caller outside fdsreader, seen
    from the function that calls this one."""
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR + os.sep):
        frame, level = frame.f_back, level + 1
    return level


def _warn_raw_codes():
    """Warn once per session that Smoke3D values are display codes."""
    global _raw_codes_warned
    if not _raw_codes_warned:
        _raw_codes_warned = True
        warnings.warn(
            "Smoke3D data are Smokeview display codes between 0 and 254, not physical values. Use "
            "to_physical() or to_global(physical=True) to convert them, see "
            "https://fdsreader.readthedocs.io/en/latest/api/smoke3d.html",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )


def _decode_rle_scalar(rle_data: np.ndarray, nchars_out: int, total_size: int) -> np.ndarray:
    """Reference (slow) run-length decoder, used as a fallback for encodings the vectorized
    decoder can't safely handle (see :func:`_decode_rle`)."""
    decoded_data = np.empty(total_size)
    i = 0
    mark = np.uint8(255)
    out_pos = 0
    while i < nchars_out:
        if rle_data[i] == mark:
            value = rle_data[i + 1]
            repeats = rle_data[i + 2]
            i += 3
        else:
            value = rle_data[i]
            repeats = 1
            i += 1
        decoded_data[out_pos : out_pos + repeats] = value
        out_pos += repeats
    return decoded_data


def _decode_rle(rle_data: np.ndarray, nchars_out: int, total_size: int) -> np.ndarray:
    """Decode run-length-encoded smoke3d data (see "RLE" subroutine in smvv.f90).

    The format escapes runs as a (255, value, repeat_count) triple; every other byte is a
    single literal value. In practice, encoded values and repeat counts never reach 255
    themselves (255 is reserved as the escape marker), which means every occurrence of a 255
    byte in the stream unambiguously starts a run and the whole stream can be decoded with
    vectorized numpy operations instead of a per-token Python loop. If that assumption doesn't
    hold for some file (i.e. two escape sequences would overlap), fall back to the slow but
    always-correct scalar decoder.
    """
    mark = np.uint8(255)
    data = rle_data[:nchars_out]
    mark_positions = np.flatnonzero(data == mark)
    if mark_positions.size > 0:
        if mark_positions[-1] + 3 > nchars_out or (mark_positions.size > 1 and np.any(np.diff(mark_positions) < 3)):
            return _decode_rle_scalar(rle_data, nchars_out, total_size)

    # Every byte is either a literal (kept as-is, 1 repeat) or the first byte of an escaped run
    # (replaced by the run's value/repeat-count, with the following 2 payload bytes dropped).
    consumed = np.zeros(nchars_out, dtype=bool)
    if mark_positions.size > 0:
        consumed[mark_positions + 1] = True
        consumed[mark_positions + 2] = True
    keep = ~consumed

    values = data.copy()
    repeats = np.ones(nchars_out, dtype=np.int64)
    if mark_positions.size > 0:
        values[mark_positions] = data[mark_positions + 1]
        repeats[mark_positions] = data[mark_positions + 2].astype(np.int64)

    decoded = np.repeat(values[keep], repeats[keep])
    return decoded[:total_size].astype(np.float64)


class SubSmoke3D:
    """Part of a smoke3d output for a single mesh.

    :ivar mesh: The mesh containing the data.
    :ivar upper_bounds: Numpy ndarray containing the last column of the .s3d.sz file for each
        timestep. That is MAX_VAL (see :attr:`max_vals`), except for soot written by FDS 6.10,
        whose last column is the maximum soot density of the frame.
    :ivar times: Numpy ndarray containing all time steps for which data has been written out.
    :ivar version: Smoke3D format version. 0 for FDS < 6.11 (soot stored as obscuration), 1 for
        FDS >= 6.11 (soot stored as density normalized by the maximum of each frame).
    :ivar mass_extinction_coefficient: Mass extinction coefficient in m2/kg as written to the .smv
        file. Only positive for soot ("GAS") quantities, 0 for HRRPUV and temperature.
    :ivar max_vals: The MAX_VAL column (4th column) of the .s3d.sz file for each timestep. For
        soot written by FDS >= 6.11 this is the maximum density of the frame, which the codes are
        normalised by.
    """

    def __init__(
        self,
        file_path: str,
        mesh: Mesh,
        upper_bounds: np.ndarray,
        times: np.ndarray,
        version: int = 0,
        mass_extinction_coefficient: float = 0.0,
        max_vals: np.ndarray | None = None,
        smoke3d: Optional["Smoke3D"] = None,
    ):
        self.mesh = mesh
        self.upper_bounds = upper_bounds
        self.times = times
        self.version = version
        self.mass_extinction_coefficient = mass_extinction_coefficient
        self.max_vals = upper_bounds if max_vals is None else max_vals
        self._smoke3d = smoke3d  # The Smoke3D this subsmoke belongs to, for its quantity and limits

        self._file_path = file_path  # Path to the binary data file

    @property
    def data(self) -> np.ndarray:
        """The raw Smoke3D values of this mesh: Smokeview display codes between 0 and 254, not
        physical values. See :meth:`to_physical` for the conversion. Lazily loaded."""
        _warn_raw_codes()
        return self._codes()

    def _codes(self) -> np.ndarray:
        """Load the display codes from the .s3d file, without warning."""
        if not hasattr(self, "_data"):
            with open(self._file_path, "rb") as infile:
                dtype_header = fdtype.new((("i", 8),))
                dtype_nchars = fdtype.new((("i", 2),))

                header = fdtype.read(infile, dtype_header, 1)[0][0]
                nx, ny, nz = int(header[3]), int(header[5]), int(header[7])
                data_shape = (nx + 1, ny + 1, nz + 1)

                self._data = np.empty((self.times.size,) + data_shape, dtype=np.float32)

                for t in range(self.times.size):
                    fdtype.read(infile, fdtype.FLOAT, 1)  # Skip time value
                    nchars_out = int(fdtype.read(infile, dtype_nchars, 1)[0][0][1])

                    if nchars_out > 0:
                        dtype_data = fdtype.new((("u", nchars_out),))

                        rle_data = fdtype.read(infile, dtype_data, 1)[0][0]
                        decoded_data = _decode_rle(rle_data, nchars_out, (nx + 1) * (ny + 1) * (nz + 1))

                        self._data[t, :, :, :] = decoded_data.reshape(data_shape, order="F")

        return self._data

    def to_physical(
        self,
        return_saturated: bool = False,
        hrrpuv_max: float | None = None,
        temp_min: float | None = None,
        temp_max: float | None = None,
        kind: Literal["gas", "hrrpuv", "temperature"] | None = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Convert the display codes of this mesh back to physical values.

        How FDS encodes the values (``SMOKE3D_TO_FILE`` in ``Source/smvv.f90``):

        - Gas density, e.g. soot (kg/m3): before FDS 6.11 (format version 0) the obscuration
          ``254 * (1 - exp(-K_m * dx * rho))`` over the size ``dx`` of the first cell along x;
          from FDS 6.11 on (version 1) ``254 * rho / MAX_VAL``, normalised by the maximum
          density of each frame.
        - HRRPUV (kW/m3): ``254 * q / q_max``, clipped to ``[0, q_max]``.
        - Temperature and every other quantity (C for temperature):
          ``254 * (T - T_min) / (T_max - T_min)``, clipped to ``[T_min, T_max]``. The default
          Smoke3D temperature is the effective flame temperature (corrected in burning cells),
          which FDS before 6.11 labels ``TEMPERATURE`` and FDS 6.11 ``EFFECTIVE FLAME TEMPERATURE``.

        Before FDS 6.11, HRRPUV and temperature were truncated to the code below, so the middle
        of the step is returned; from FDS 6.11 on they are rounded. The limits come from the .smv
        file (FDS 6.10 and later); for older output the FDS defaults (1200 kW/m3, 20 and 2000 C)
        are assumed with a warning, unless given here.

        The result keeps the 8-bit resolution and FDS's averaging onto cell corners. Code 254
        means at or above the upper limit (within half a step from FDS 6.11 on, as it rounds), so
        those values, and soot that saturates the obscuration before FDS 6.11, are returned at the
        limit and marked in the array ``return_saturated`` adds.

        :param return_saturated: Also return a boolean array marking values that were cut off.
        :param hrrpuv_max: ``HRRPUV_MAX_SMV`` in kW/m3, overrides the value from the .smv file.
        :param temp_min: ``TEMP_MIN_SMV`` in C, overrides the value from the .smv file.
        :param temp_max: ``TEMP_MAX_SMV`` in C, overrides the value from the .smv file.
        :param kind: How FDS encoded the quantity, see :attr:`Smoke3D.kind`, which is used by
            default. Only needed when it is recognised wrongly, e.g. for a quantity chosen
            with ``&SM3D``.
        :returns: Array of shape (time, x, y, z), and the saturation mask if requested.
        """
        if self._smoke3d is None:
            raise ValueError("to_physical needs the Smoke3D this subsmoke belongs to, for its quantity and limits")
        if kind is None:
            kind = self._smoke3d.kind
        elif kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        codes = self._codes()
        rounded = self.version >= 1  # FDS 6.11 rounds, earlier versions truncated HRRPUV and temperature
        saturated = codes >= 254

        if kind == "gas":
            if self.version >= 1:
                max_vals = np.asarray(self.max_vals, dtype=np.float32)[: codes.shape[0]]
                values = codes * (max_vals / 254).reshape((-1, 1, 1, 1))
                saturated = np.zeros_like(saturated)  # 254 is the maximum of the frame, not a cut-off
            else:
                if self.mass_extinction_coefficient <= 0:
                    raise ValueError(
                        "Before FDS 6.11 gas densities were stored as obscuration, which is 0 everywhere "
                        "for a species without a mass extinction coefficient, so they cannot be recovered."
                    )
                x = self.mesh.coordinates["x"]
                dx = x[1] - x[0]
                obscuration = np.minimum(codes, 253) / 254  # 254 would be infinite density
                values = -np.log1p(-obscuration) / (self.mass_extinction_coefficient * dx)
        else:
            if kind == "hrrpuv":
                low, high = self._smoke3d._limits("HRRPUV_MINMAX", 0.0, hrrpuv_max)  # FDS always starts at 0
            else:
                low, high = self._smoke3d._limits("TEMP_MINMAX", temp_min, temp_max)
            step = (high - low) / 254
            if rounded:
                values = low + codes * step
            else:
                values = np.where(codes > 0, low + (codes + 0.5) * step, low)
            values = np.where(saturated, high, values)

        values = np.asarray(values, dtype=np.float32)
        return (values, saturated) if return_saturated else values

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        if hasattr(self, "_data"):
            del self._data


class Smoke3D(NumpyArrayMixin):
    """Smoke3D file data container including metadata. Consists of multiple subsmokes, one for each
        mesh.

    :ivar times: Numpy ndarray containing all time steps for which data has been written out.
    :ivar quantity: :class:`Quantity` object containing information about the recorded quantity and its unit.
    """

    _handled_functions = _HANDLED_FUNCTIONS
    _HANDLED_FUNCTIONS[np.mean] = NumpyArrayMixin.mean
    _HANDLED_FUNCTIONS[np.std] = NumpyArrayMixin.std

    def _array_subitems(self):
        return self._subsmokes.values()

    def __init__(
        self,
        root_path: str,
        times: np.ndarray,
        quantity: Quantity,
        limits: Dict[str, Tuple[float, float]] | None = None,
    ):
        self._root_path = root_path
        self.times = times
        self.quantity = quantity
        # HRRPUV_MINMAX and TEMP_MINMAX from the .smv file; filled while it is parsed
        self._smv_limits = {} if limits is None else limits
        self._warned_default_limits = set()  # keywords the default-limits warning was shown for

        # List of all subsmokes this Smoke3D consists of (one per mesh).
        self._subsmokes: Dict[str, SubSmoke3D] = dict()

    @property
    def kind(self) -> Literal["gas", "hrrpuv", "temperature"]:
        """How FDS scaled this quantity to display codes: ``"gas"`` for the density of a species
        such as soot, ``"hrrpuv"`` for the heat release rate per unit volume, and
        ``"temperature"`` for temperature and every other quantity, which FDS scales with the
        temperature limits.

        A species density is recognised by its mass extinction coefficient or, for species
        without one, by a name ending in "DENSITY" with the unit kg/m3."""
        is_species_density = self.quantity.name.upper().endswith(" DENSITY") and self.quantity.unit == "kg/m3"
        if self.mass_extinction_coefficient > 0 or is_species_density:
            return "gas"
        if self.quantity.name.upper() == "HRRPUV" or self.quantity.short_name.lower() == "hrrpuv":
            return "hrrpuv"
        return "temperature"

    def _limits(self, keyword: str, low: float | None, high: float | None) -> Tuple[float, float]:
        """The (min, max) FDS scaled to, from the arguments, the .smv file or the FDS defaults.

        Before FDS 6.10 the limits are not in the .smv file. FDS 6.8 and 6.9 started the
        temperature scale at the ambient temperature, which the Simulation reads from the .out
        file; FDS 6.7 started it at 20 C."""
        smv = self._smv_limits.get(keyword)
        if keyword == "HRRPUV_MINMAX":
            default, source = DEFAULT_HRRPUV_MINMAX, "the FDS default"
        elif "AMBIENT_TEMPERATURE" in self._smv_limits:
            default = (self._smv_limits["AMBIENT_TEMPERATURE"][0], DEFAULT_TEMP_MINMAX[1])
            source = "the ambient temperature from the .out file and the FDS default maximum"
        else:
            default, source = DEFAULT_TEMP_MINMAX, "the FDS default"
        if smv is None and (low is None or high is None) and keyword not in self._warned_default_limits:
            self._warned_default_limits.add(keyword)
            warnings.warn(
                f"The .smv file has no {keyword} (written from FDS 6.10 on), assuming {default} for "
                f"{self.quantity.name}, {source}. Pass the values set on &DUMP if they were changed.",
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
        base = smv if smv is not None else default
        return (base[0] if low is None else low, base[1] if high is None else high)

    def _add_subsmoke(
        self,
        filename: str,
        mesh: Mesh,
        upper_bounds: np.ndarray,
        version: int = 0,
        mass_extinction_coefficient: float = 0.0,
        max_vals: np.ndarray | None = None,
    ) -> SubSmoke3D:
        subsmoke = SubSmoke3D(
            os.path.join(self._root_path, filename),
            mesh,
            upper_bounds,
            self.times,
            version,
            mass_extinction_coefficient,
            max_vals,
            smoke3d=self,
        )
        self._subsmokes[mesh.id] = subsmoke

        # If lazy loading has been disabled by the user, load the data instantaneously instead
        if not settings.LAZY_LOAD:
            subsmoke._codes()

        return subsmoke

    def to_global(
        self,
        masked: bool = False,
        fill: float = 0,
        return_coordinates: bool = False,
        physical: bool = False,
        return_saturated: bool = False,
        hrrpuv_max: float | None = None,
        temp_min: float | None = None,
        temp_max: float | None = None,
        kind: Literal["gas", "hrrpuv", "temperature"] | None = None,
    ) -> Union[np.ndarray, tuple]:
        """Creates a global numpy ndarray from all subsmokes.

        :param masked: Whether to apply the obstruction mask to the data or not.
        :param fill: The fill value to use for masked entries. Only used when masked=True.
        :param return_coordinates: If true, return the matching coordinate for each value on the generated grid.
        :param physical: If true, convert the display codes to physical values, see
            :meth:`SubSmoke3D.to_physical`. Otherwise the raw codes between 0 and 254 are returned.
        :param return_saturated: With ``physical=True``, also return a boolean array marking values
            FDS cut off at the upper limit.
        :param hrrpuv_max: Overrides ``HRRPUV_MAX_SMV``, see :meth:`SubSmoke3D.to_physical`.
        :param temp_min: Overrides ``TEMP_MIN_SMV``, see :meth:`SubSmoke3D.to_physical`.
        :param temp_max: Overrides ``TEMP_MAX_SMV``, see :meth:`SubSmoke3D.to_physical`.
        :param kind: Overrides :attr:`kind`, see :meth:`SubSmoke3D.to_physical`.
        :returns: The global array, followed by the saturation mask if ``return_saturated`` and the
            coordinates if ``return_coordinates``.
        """
        if return_saturated and not physical:
            raise ValueError("return_saturated needs physical=True")
        if not physical:
            _warn_raw_codes()

        if len(self._subsmokes) == 0:
            result = [np.array([])]
            if return_saturated:
                result.append(np.array([], dtype=bool))
            if return_coordinates:
                result.append({d: np.array([]) for d in ("x", "y", "z")})
            return tuple(result) if len(result) > 1 else result[0]

        coord_min = {"x": math.inf, "y": math.inf, "z": math.inf}
        coord_max = {"x": -math.inf, "y": -math.inf, "z": -math.inf}
        for dim in ("x", "y", "z"):
            for subsmoke in self._subsmokes.values():
                co = subsmoke.mesh.coordinates[dim]
                coord_min[dim] = min(co[0], coord_min[dim])
                coord_max[dim] = max(co[-1], coord_max[dim])

        # The global grid will use the finest mesh as base and duplicate values of the coarser
        # meshes. Therefore, we first find the finest mesh and calculate the step size in each
        # dimension.
        step_sizes_min = {
            "x": coord_max["x"] - coord_min["x"],
            "y": coord_max["y"] - coord_min["y"],
            "z": coord_max["z"] - coord_min["z"],
        }
        step_sizes_max = {"x": 0, "y": 0, "z": 0}
        steps = dict()
        global_max = {"x": -math.inf, "y": -math.inf, "z": -math.inf}

        for dim in ("x", "y", "z"):
            for subsmoke in self._subsmokes.values():
                step_size = subsmoke.mesh.coordinates[dim][1] - subsmoke.mesh.coordinates[dim][0]
                step_sizes_min[dim] = min(step_size, step_sizes_min[dim])
                step_sizes_max[dim] = max(step_size, step_sizes_max[dim])
                global_max[dim] = max(subsmoke.mesh.coordinates[dim][-1], global_max[dim])

        for dim in ("x", "y", "z"):
            if step_sizes_min[dim] == 0:
                step_sizes_min[dim] = math.inf
                steps[dim] = 1
            else:
                steps[dim] = (
                    max(int(round((coord_max[dim] - coord_min[dim]) / step_sizes_min[dim])), 1) + 1
                )  # + step_sizes_max[dim] / step_sizes_min[dim]

        grid = np.full((self.n_t, steps["x"], steps["y"], steps["z"]), np.nan, dtype=np.float32)
        saturated_grid = np.zeros(grid.shape, dtype=bool) if return_saturated else None

        for subsmoke in self._subsmokes.values():
            subsmoke_saturated = None
            if physical:
                subsmoke_data, subsmoke_saturated = subsmoke.to_physical(
                    return_saturated=True, hrrpuv_max=hrrpuv_max, temp_min=temp_min, temp_max=temp_max, kind=kind
                )
            else:
                subsmoke_data = subsmoke._codes().copy()
            if masked:
                mask = subsmoke.mesh.get_obstruction_mask(self.times)

            start_idx = {
                dim: int(round((subsmoke.mesh.coordinates[dim][0] - coord_min[dim]) / step_sizes_min[dim]))
                for dim in ("x", "y", "z")
            }
            end_idx = {
                dim: int(round((subsmoke.mesh.coordinates[dim][-1] - coord_min[dim]) / step_sizes_min[dim]))
                for dim in ("x", "y", "z")
            }

            # Map each axis onto the global grid in turn. Border points are dropped unless they lie on the
            # border of the simulation space, as all other border points appear twice where meshes overlap.
            # The global grid uses the finest mesh as base, so coarser meshes need their values repeated
            # along every axis to match its resolution.
            for axis, dim in enumerate(("x", "y", "z")):
                co = subsmoke.mesh.coordinates[dim]
                n_repeat = max(int(round((co[1] - co[0]) / step_sizes_min[dim])), 1)
                keep_last = bool(np.isclose(co[-1], global_max[dim]))
                if keep_last:
                    end_idx[dim] += 1
                subsmoke_data = resample_mesh_axis(subsmoke_data, axis + 1, n_repeat, keep_last)
                if return_saturated:
                    subsmoke_saturated = resample_mesh_axis(subsmoke_saturated, axis + 1, n_repeat, keep_last)
                if masked:
                    mask = resample_mesh_axis(mask, axis + 1, n_repeat, keep_last)

            # If the slice should be masked, we set all cells at which an obstruction is in the
            # simulation space to the fill value set by the user
            if masked:
                subsmoke_data = np.where(mask, subsmoke_data, fill)
                if return_saturated:
                    subsmoke_saturated = subsmoke_saturated & mask

            region = (
                slice(None),
                slice(start_idx["x"], end_idx["x"]),
                slice(start_idx["y"], end_idx["y"]),
                slice(start_idx["z"], end_idx["z"]),
            )
            shape = (
                self.n_t,
                end_idx["x"] - start_idx["x"],
                end_idx["y"] - start_idx["y"],
                end_idx["z"] - start_idx["z"],
            )
            grid[region] = subsmoke_data.reshape(shape)
            if return_saturated:
                saturated_grid[region] = subsmoke_saturated.reshape(shape)

        if return_coordinates:
            coordinates = dict()
            for dim_index, dim in enumerate(("x", "y", "z")):
                coordinates[dim] = np.linspace(coord_min[dim], coord_max[dim], grid.shape[dim_index + 1])

        result = [grid]
        if return_saturated:
            result.append(saturated_grid)
        if return_coordinates:
            result.append(coordinates)
        return tuple(result) if len(result) > 1 else result[0]

    def to_physical(
        self,
        masked: bool = False,
        fill: float = 0,
        return_coordinates: bool = False,
        return_saturated: bool = False,
        hrrpuv_max: float | None = None,
        temp_min: float | None = None,
        temp_max: float | None = None,
        kind: Literal["gas", "hrrpuv", "temperature"] | None = None,
    ):
        """Physical values of all meshes on one grid; the same as ``to_global(physical=True)``.
        See :meth:`SubSmoke3D.to_physical` for the conversion and its limits."""
        return self.to_global(
            masked=masked,
            fill=fill,
            return_coordinates=return_coordinates,
            physical=True,
            return_saturated=return_saturated,
            hrrpuv_max=hrrpuv_max,
            temp_min=temp_min,
            temp_max=temp_max,
            kind=kind,
        )

    def __getitem__(self, mesh: Mesh):
        """Returns the :class:`SubSmoke` that contains data for the given mesh."""
        return self.get_subsmoke(mesh)

    @property
    def n_t(self) -> int:
        """Get the number of timesteps for which data was output."""
        return len(self.times)

    def get_subsmoke(self, mesh: Mesh):
        """Returns the :class:`SubSmoke` that contains data for the given mesh."""
        return self._subsmokes[mesh.id]

    def __repr__(self):
        return f"Smoke3D(quantity={self.quantity}, meshes={len(self._subsmokes)})"

    @property
    def subsmokes(self):
        """Returns a list with one SubSmoke3D object per mesh."""
        return list(self._subsmokes.values())

    @property
    def version(self) -> int:
        """Smoke3D format version, see :attr:`SubSmoke3D.version`."""
        return max((subsmoke.version for subsmoke in self._subsmokes.values()), default=0)

    @property
    def mass_extinction_coefficient(self) -> float:
        """Mass extinction coefficient in m2/kg, see :attr:`SubSmoke3D.mass_extinction_coefficient`."""
        return max((subsmoke.mass_extinction_coefficient for subsmoke in self._subsmokes.values()), default=0.0)

    @property
    def vmax(self):
        """Maximum of the MAX_VAL column of the .s3d.sz files at any time. Note that this is not
        necessarily on the same scale as the raw values in :attr:`SubSmoke3D.data`: for soot
        written by FDS >= 6.11 it is the maximum soot density in kg/m3."""
        curr_max = max(np.max(subsmoke3d.upper_bounds) for subsmoke3d in self._subsmokes.values())
        if curr_max == np.float32(-1e33):
            return max(np.max(subsmoke3d._codes()) for subsmoke3d in self._subsmokes.values())
        return curr_max

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory."""
        for subsmoke in self._subsmokes.values():
            subsmoke.clear_cache()


# __array_function__ implementations
# ...
