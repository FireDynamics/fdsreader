import logging
import os
import pickle
import warnings
from datetime import datetime, timezone
from typing import AnyStr, Dict, List, Sequence, TextIO, Tuple, Union

import numpy as np

from fdsreader import __version__, settings
from fdsreader.bndf import ObstructionCollection, SubObstruction
from fdsreader.devc import Device, DeviceCollection
from fdsreader.evac import EvacCollection
from fdsreader.fds_classes import Mesh, MeshCollection, Surface, Ventilation
from fdsreader.geom import Geometry, GeometryCollection
from fdsreader.isof import IsosurfaceCollection
from fdsreader.part import ParticleCollection
from fdsreader.pl3d import Plot3D, Plot3DCollection
from fdsreader.simulation_loaders import SimulationLoaderMixin
from fdsreader.simulation_loaders import _plot3d_time as _plot3d_time  # noqa: F401 - imported from here before
from fdsreader.slcf import GeomSlice, GeomSliceCollection, Slice, SliceCollection
from fdsreader.smoke3d import Smoke3DCollection
from fdsreader.utils import Quantity, log_error
from fdsreader.utils.data import Profile, create_hash, get_smv_file


def _wall_times_to_utc(stamps: Sequence[str]) -> np.ndarray:
    """Parse the wall-clock time stamps of CHID_steps.csv as UTC, as numpy did before it
    deprecated time zone offsets (and warned on every load)."""
    utc = []
    for stamp in stamps:
        time = datetime.fromisoformat(stamp.strip())
        # Without an offset the stamp stays as written; astimezone would assume the local time zone
        utc.append(time.astimezone(timezone.utc).replace(tzinfo=None) if time.tzinfo else time)
    return np.array(utc, dtype="datetime64[ms]")


class StepData(dict):
    """The columns of CHID_steps.csv by name, see :attr:`Simulation.steps`.

    The key ``"Time Step"`` holds the wall-clock times for compatibility with earlier versions
    and warns once when used: in the FDS User Guide the column ``Time Step`` is the iteration
    number, available here as ``"Step"``, and the wall-clock time is ``"Wall Time"``.
    """

    _warned = False

    def __getitem__(self, key):
        if key == "Time Step" and not StepData._warned:
            StepData._warned = True
            warnings.warn(
                'steps["Time Step"] holds the wall-clock time, not the time step number. Use '
                'steps["Wall Time"] for the time and steps["Step"] for the number; "Time Step" will '
                "become the number, as in the FDS User Guide, in a future release.",
                FutureWarning,
                stacklevel=2,
            )
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


class Simulation(SimulationLoaderMixin):
    """Master class managing all data for a given simulation.

    :ivar reader_version: The version of the fdsreader used to load the Simulation.
    :ivar smv_file_path: Path to the .smv file of the simulation.
    :ivar root_path: Path to the root directory of the simulation.
    :ivar fds_version: Version of FDS the simulation was performed with.
    :ivar chid: Name (ID) of the simulation.
    :ivar hrrpuv_cutoff: The hrrpuv_cutoff value.
    :ivar default_texture_origin: The default origin used for textures with no explicit origin.
    :ivar out_file_path: Path to the .out file of the simulation.
    :ivar surfaces: List containing all surfaces defined in this simulation.
    :ivar meshes: List containing all meshes (grids) defined in this simulation.
    :ivar ventilations: List containing all ventilations defined in this simulation.
    :ivar obstructions: All defined obstructions combined into a :class:`ObstructionCollection`.
    :ivar slices: All defined slices combined into a :class:`SliceCollection`.
    :ivar data_3d: All defined 3D plotting data combined into a :class:`Plot3DCollection`.
    :ivar smoke_3d: All defined 3D smoke data combine into a :class:`Smoke3DCollecction`
    :ivar isosurfaces: All defined isosurfaces combined into a :class:`IsosurfaceCollection`.
    :ivar particles: All defined particles combined into a :class:`ParticleCollection`.
    :ivar evacs: All defined evacuations combined into a :class:`EvacCollection`.
    :ivar devices: List containing all :class:`Device` s defined in this simulation.
    :ivar profiles: Dictionary mapping profile ids to the corresponding :class:`Profile` s defined in this simulation.
    :ivar geoms: List containing all geometries (:class:`Geometry`) defined in this simulation.
    :ivar geom_data: All geometry data by quantity combined into a :class:`GeometryCollection`.
    :ivar cpu: Dictionary mapping .csv header keys to numpy arrays containing cpu data.
    :ivar hrr: Dictionary mapping .csv header keys to numpy arrays containing hrr data.
    :ivar steps: The time step data of CHID_steps.csv as a :class:`StepData` dictionary:
        ``"Step"`` (iteration number), ``"Wall Time"`` (date and time, UTC), ``"Step Size"`` and
        ``"Simulation Time"`` (s), and ``"CPU Time"`` (s since FDS started). ``"Time Step"`` still
        holds the wall-clock time, but warns: it will become the iteration number.
    :ivar ctrl: Dictionary mapping control (&CTRL) names to numpy arrays of their state over time.
        Only set if the simulation has a CTRL csv export.
    :ivar mass: Dictionary mapping species names to numpy arrays of their mass history. Only set
        if the simulation has a mass csv export.
    :ivar load_errors: List of (module, exception) tuples for loaders that failed and were
        swallowed during parsing. Empty when nothing went wrong.
    """

    # Bump when the pickled structure of a Simulation or its data classes changes, so caches
    # written by an older layout are parsed again instead of being loaded with missing attributes.
    _CACHE_FORMAT = 4
    _loading = False

    def __new__(cls, path: str):
        smv_file_path = get_smv_file(path)
        root_path = os.path.dirname(smv_file_path)

        with open(smv_file_path) as infile:
            if not infile.read(1):
                raise ValueError(f"SMV file is empty: '{smv_file_path}'")

        chid = None
        with open(smv_file_path) as infile:
            for line in infile:
                if line.strip() == "CHID":
                    chid = infile.readline().strip()
                    break

        if chid is None:
            raise ValueError(f"Could not find CHID in '{smv_file_path}'")

        pickle_file_path = Simulation._get_pickle_filename(root_path, chid)
        if settings.ENABLE_CACHING:
            if not Simulation._loading and os.path.isfile(pickle_file_path):
                Simulation._loading = True
                try:
                    with open(pickle_file_path, "rb") as f:
                        sim = pickle.load(f)
                except Exception as e:
                    sim = None
                    if settings.DEBUG:
                        logging.exception(e)
                finally:
                    # Reset immediately after unpickling so the cache can be used again by any
                    # later Simulation(...) call in this process, not just the first one.
                    Simulation._loading = False

                if sim is not None:
                    # Short-circuiting matters here: a pickle file holding some other/older
                    # object shape must not reach sim.reader_version/sim._hash at all.
                    valid = (
                        isinstance(sim, cls)  # Check if pickle file stores a Simulation
                        and sim.reader_version == __version__  # Check if the fdsreader version still matches
                        # Editable installs keep a frozen __version__, so also compare the cache format
                        and getattr(sim, "_cache_format", None) == Simulation._CACHE_FORMAT
                        and sim._hash == create_hash(smv_file_path)  # Check if the smv_file did not change
                    )

                    if valid:
                        # Older pickle caches may predate the load_errors attribute
                        if not hasattr(sim, "load_errors"):
                            sim.load_errors = list()
                        # Return cached sim if it turned out to be valid
                        return sim

                os.remove(pickle_file_path)
        else:
            if os.path.isfile(pickle_file_path):
                os.remove(pickle_file_path)
        return super().__new__(cls)

    def __getnewargs__(self):
        return (self.smv_file_path,)

    def __repr__(self):
        r = (
            f"Simulation(chid={self.chid},\n"
            + f"           meshes={len(self.meshes)},\n"
            + (f"           obstructions={len(self.obstructions)},\n" if len(self.obstructions) > 0 else "")
            + (f"           geoms={len(self.geoms)},\n" if len(self.geoms) > 0 else "")
            + (f"           slices={len(self.slices)},\n" if len(self.slices) > 0 else "")
            + (f"           geomslices={len(self.geomslices)},\n" if len(self.geomslices) > 0 else "")
            + (f"           data_3d={len(self.data_3d)},\n" if len(self.data_3d) > 0 else "")
            + (f"           smoke_3d={len(self.smoke_3d)},\n" if len(self.smoke_3d) > 0 else "")
            + (f"           isosurfaces={len(self.isosurfaces)},\n" if len(self.isosurfaces) > 0 else "")
            + (f"           particles={len(self.particles)},\n" if len(self.particles) > 0 else "")
            + (f"           evacs={len(self.evacs)},\n" if len(self.evacs) > 0 else "")
            + (f"           devices={len(self.devices)},\n" if len(self.devices) > 0 else "")
        )
        return r[:-2] + ")"

    def __init__(self, path: str):
        """
        :param path: Either the path to the directory containing the simulation data or direct path
            to the .smv file for the simulation in case that multiple simulation output was written to
            the same directory.
        """
        if settings.IGNORE_ERRORS:
            warnings.filterwarnings("ignore")

        # Check if the file has already been instantiated via a cached pickle file
        if not hasattr(self, "_hash"):
            self.reader_version = __version__
            self._cache_format = Simulation._CACHE_FORMAT

            self.smv_file_path = get_smv_file(path)

            self.root_path = os.path.dirname(self.smv_file_path)

            self.geoms: List[Geometry] = list()
            self.surfaces: List[Surface] = list()
            self.ventilations = dict()

            # Will only be used during the loading process to map boundary data to the correct
            # obstruction
            self._subobstructions: Dict[str, List[SubObstruction]] = dict()

            # First collect all meta-information for any FDS data to later combine the gathered
            # information into data collections. While collecting the meta-data simple python
            # containers are used.
            self._obstructions = list()
            self._slices = dict()
            self._geomslices = dict()
            self.data_3d = Plot3DCollection([Plot3D(self.root_path) for _ in range(5)])
            self._smoke_3d = dict()
            # Value ranges FDS scaled Smoke3D output to, from the HRRPUV_MINMAX and TEMP_MINMAX
            # keywords of the .smv file (FDS 6.10 and later). Shared with every Smoke3D.
            self._smoke3d_limits: Dict[str, Tuple[float, float]] = dict()
            self._isosurfaces = dict()
            self._particles = list()
            self._evacs = list()
            self._geom_data = list()
            self._meshes: List[Mesh] = list()
            self._devices = dict()

            self.profiles: Dict[str, Profile] = dict()
            self.load_errors: List[Tuple[str, Exception]] = list()

            self.parse_smv_file()

            self.cpu = self._load_CPU_data()
            self._load_profiles()

            # POST INIT (post read)
            self.out_file_path = os.path.join(self.root_path, self.chid + ".out")
            self._read_smoke3d_ambient_temperature()
            self.ventilations: List[Ventilation] = list(self.ventilations.values())

            for device_id, device in self._devices.items():
                if isinstance(device, list):
                    for devc in device:
                        devc._data_callback = self._load_DEVC_data
                else:
                    device._data_callback = self._load_DEVC_data

            # Combine the gathered temporary information into data collections
            self.geom_data = GeometryCollection(self._geom_data)
            self.slices = SliceCollection(
                Slice(self.root_path, slice_data[0]["id"], slice_data[0]["cell_centered"], slice_data[1:])
                for slice_data in self._slices.values()
            )
            self.geomslices = GeomSliceCollection(
                GeomSlice(self.root_path, slice_data[0]["id"], slice_data[0]["times"], slice_data[1:])
                for slice_data in self._geomslices.values()
            )
            self.smoke_3d = Smoke3DCollection(self._smoke_3d.values())
            self.isosurfaces = IsosurfaceCollection(self._isosurfaces.values())
            self.devices = DeviceCollection(self._devices.values())
            self.obstructions = ObstructionCollection(self._obstructions)
            # If no particles are simulated, initialize empty data container for consistency
            if isinstance(self._particles, list):
                self.particles = ParticleCollection((), ())
            else:
                self.particles = self._particles
                self.particles._post_init()
            # If no evacs are simulates, initialize empty data container for consistency
            # (also when Evac classes were registered but their data could not be loaded)
            if not isinstance(self._evacs, EvacCollection):
                self.evacs = EvacCollection((), "", ())
            self.meshes = MeshCollection(self._meshes)
            del (
                self._geom_data,
                self._geomslices,
                self._slices,
                self._obstructions,
                self._smoke_3d,
                self._isosurfaces,
                self._devices,
                self._particles,
                self._evacs,
                self._meshes,
                self._subobstructions,
            )

            if settings.ENABLE_CACHING:
                # Hash will be saved to simulation pickle file and compared to new hash when loading
                # the pickled simulation again in the next run of the program.
                self._hash = create_hash(self.smv_file_path)
                with open(Simulation._get_pickle_filename(self.root_path, self.chid), "wb") as pickle_file:
                    pickle.dump(self, pickle_file, protocol=4)

        if len(self.evacs) > 0:
            warnings.warn(
                "FDS+Evac has been removed from FDS, and fdsreader's evacuation support is deprecated: "
                "it is no longer tested and may be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )

    # The .smv keywords and the methods that read what follows them, checked in this order: a keyword that is not
    # matched exactly is matched by the first entry it starts with ("prefix") or contains ("contains").
    _EXACT_KEYWORDS = {
        "VERSION": "_read_version",
        "FDSVERSION": "_read_version",
        "CHID": "_read_chid",
        "TITLE": "_read_title",
        "TIMES": "_read_times",
        "CSVF": "_read_csvf",
        "HRRPUV_MINMAX": "_read_smoke3d_limits",
        "TEMP_MINMAX": "_read_smoke3d_limits",
        "HRRPUVCUT": "_read_hrrpuv_cutoff",
        "TOFFSET": "_read_texture_origin",
        "CLASS_OF_PARTICLES": "_read_particle_class",
        "CLASS_OF_HUMANS": "_read_evac_class",
        "SURFACE": "_read_surface",
        "DEVICE": "_read_device",
    }
    _KEYWORD_PATTERNS = (
        ("prefix", "GEOM", "_load_geoms"),
        ("prefix", "GRID", "_read_mesh"),
        ("prefix", "DEVICE_ACT", "_read_device_activation"),
        ("prefix", "SLC", "_load_slice"),
        ("prefix", "BNDS", "_load_geomslice"),
        ("contains", "ISOG", "_load_isosurface"),
        ("prefix", "PL3D", "_load_plot_3d"),
        ("prefix", "SMOKG3D", "_load_smoke_3d"),
        ("prefix", "SMOKF3D", "_load_smoke_3d"),
        ("prefix", "BNDF", "_read_boundary_data"),
        ("prefix", "BNDC", "_read_boundary_data"),
        ("prefix", "BNDE", "_load_boundary_data_geom"),
        ("prefix", "PRT5", "_load_particle_data"),
        ("prefix", "EVA5", "_load_evac_data"),
        ("prefix", "SHOW_OBST", "_toggle_obst"),
        ("prefix", "HIDE_OBST", "_toggle_obst"),
    )

    @classmethod
    def _handler_name(cls, keyword: str) -> Union[str, None]:
        """The method that reads the lines after a keyword of the .smv file, or None for keywords not read."""
        name = cls._EXACT_KEYWORDS.get(keyword)
        if name is not None:
            return name
        for kind, pattern, name in cls._KEYWORD_PATTERNS:
            if keyword.startswith(pattern) if kind == "prefix" else pattern in keyword:
                return name
        return None

    def parse_smv_file(self):
        # Global device list in registration order, used to resolve DEVICE_ACT indices; only needed while parsing
        self._devices_by_global_index: List[Device] = []
        try:
            with open(self.smv_file_path) as smv_file:
                for line in smv_file:
                    keyword = line.strip()
                    name = self._handler_name(keyword)
                    if name is not None:
                        getattr(self, name)(smv_file, keyword)
        finally:
            del self._devices_by_global_index

    # -- readers of single keywords, called by parse_smv_file ------------------------------------------------------
    def _read_version(self, smv_file: TextIO, keyword: str):
        self.fds_version = smv_file.readline().strip()

    def _read_chid(self, smv_file: TextIO, keyword: str):
        self.chid = smv_file.readline().strip()

    def _read_title(self, smv_file: TextIO, keyword: str):
        self.title = smv_file.readline().strip()

    def _read_times(self, smv_file: TextIO, keyword: str):
        self.times = [float(t.strip()) for t in smv_file.readline().strip().split()]

    def _read_csvf(self, smv_file: TextIO, keyword: str):
        csv_type = smv_file.readline().strip()
        filename = smv_file.readline().strip()
        file_path = os.path.join(self.root_path, filename)
        if csv_type == "hrr":
            self.hrr = self._load_HRR_data(file_path)
        elif csv_type == "steps":
            self.steps = self._load_step_data(file_path)
        elif csv_type == "ctrl":
            self.ctrl = self._load_named_csv_data(file_path)
        elif csv_type == "mass":
            self.mass = self._load_named_csv_data(file_path)
        elif csv_type == "devc":
            self.devc_path = file_path
            self._devices["Time"] = Device("Time", Quantity("TIME", "TIME", "s"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def _read_smoke3d_limits(self, smv_file: TextIO, keyword: str):
        minimum, maximum = (float(v) for v in smv_file.readline().split()[:2])
        self._smoke3d_limits[keyword] = (minimum, maximum)

    def _read_hrrpuv_cutoff(self, smv_file: TextIO, keyword: str):
        self.hrrpuv_cutoff = float(smv_file.readline().strip())

    def _read_texture_origin(self, smv_file: TextIO, keyword: str):
        offsets = smv_file.readline().strip().split()
        self.default_texture_origin = tuple(float(offsets[i]) for i in range(3))

    def _read_particle_class(self, smv_file: TextIO, keyword: str):
        self._particles.append(self._register_particle(smv_file))

    def _read_evac_class(self, smv_file: TextIO, keyword: str):
        self._evacs.append(self._register_evac(smv_file))

    def _read_mesh(self, smv_file: TextIO, keyword: str):
        self._meshes.append(self._load_mesh(smv_file, keyword))

    def _read_surface(self, smv_file: TextIO, keyword: str):
        self.surfaces.append(self._load_surface(smv_file))

    def _read_device(self, smv_file: TextIO, keyword: str):
        device_id, device = self._register_device(smv_file)
        if device_id in self._devices:
            if isinstance(self._devices[device_id], list):
                self._devices[device_id].append(device)
            else:
                self._devices[device_id] = [self._devices[device_id], device]
        else:
            self._devices[device_id] = device
        self._devices_by_global_index.append(device)

    def _read_device_activation(self, smv_file: TextIO, keyword: str):
        idx_str, time_str, value_str = smv_file.readline().split()
        global_index = int(idx_str) - 1  # FDS uses 1-based global device index
        act_time = float(time_str)
        act_state = bool(int(value_str))
        if global_index < len(self._devices_by_global_index):
            self._devices_by_global_index[global_index].add_activation_time(act_time, act_state)

    def _read_boundary_data(self, smv_file: TextIO, keyword: str):
        self._load_boundary_data(smv_file, keyword, cell_centered=keyword.startswith("BNDC"))

    @classmethod
    def _get_pickle_filename(cls, root_path: str, chid: str) -> AnyStr:
        """Get the filename used to save the pickled simulation."""
        return os.path.join(root_path, chid + ".pickle")

    @log_error("csv")
    def _load_step_data(self, file_path: str) -> "StepData":
        with open(file_path) as infile:
            infile.readline()
            keys = [name.replace('"', "").replace("\n", "").strip() for name in infile.readline().split(",")][2:]
        steps = np.loadtxt(file_path, dtype=int, delimiter=",", ndmin=1, usecols=0, skiprows=2)
        wall_times = _wall_times_to_utc(np.loadtxt(file_path, dtype=str, delimiter=",", ndmin=1, usecols=1, skiprows=2))
        float_values = np.loadtxt(file_path, delimiter=",", usecols=range(2, len(keys) + 2), ndmin=2, skiprows=2)
        data = StepData(self._transform_csv_data(keys, float_values))
        data["Step"] = steps
        data["Wall Time"] = wall_times
        data["Time Step"] = wall_times  # kept for compatibility, see StepData
        return data

    def clear_cache(self, clear_persistent_cache=False):
        """Remove all data from the internal cache that has been loaded so far to free memory.

        :param clear_persistent_cache: Whether to clear the persistent simulation cache as well.
        """
        self.slices.clear_cache()
        self.data_3d.clear_cache()
        self.smoke_3d.clear_cache()
        self.isosurfaces.clear_cache()
        self.obstructions.clear_cache()
        self.devices.clear_cache()
        self.evacs.clear_cache()
        self.particles.clear_cache()

        if clear_persistent_cache:
            os.remove(Simulation._get_pickle_filename(self.root_path, self.chid))
