import logging
import os
import pickle
import warnings
from types import MappingProxyType
from typing import AnyStr, Dict, List, TextIO, Tuple

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
from fdsreader.slcf import GeomSlice, GeomSliceCollection, Slice, SliceCollection
from fdsreader.smoke3d import Smoke3DCollection
from fdsreader.utils import Quantity
from fdsreader.utils.data import Profile, create_hash, get_smv_file


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
    :ivar steps: Dictionary mapping .csv header keys to numpy arrays containing steps data.
    :ivar ctrl: Dictionary mapping control (&CTRL) names to numpy arrays of their state over time.
        Only set if the simulation has a CTRL csv export.
    :ivar mass: Dictionary mapping species names to numpy arrays of their mass history. Only set
        if the simulation has a mass csv export.
    :ivar load_errors: List of (module, exception) tuples for loaders that failed and were
        swallowed during parsing. Empty when nothing went wrong.
    """

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
            if len(self._evacs) == 0:
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

    # Keyword -> handler-method-name registry replacing a hand-written if/elif chain. Exact
    # matches are checked first (mirrors the original `keyword == "..."` branches), then prefix
    # matches in the same order the original `keyword.startswith("...")` branches appeared, with
    # the ISOG substring check split out into its own step so it runs at exactly the same relative
    # position the original if/elif chain checked it (after BNDS, before PL3D) rather than after
    # every prefix. Handler names point directly at the real loader methods (on
    # SimulationLoaderMixin) wherever a keyword needs nothing beyond a straight forward; keywords
    # needing extra bookkeeping (appending a return value, partial-applying an argument, DEVICE's
    # own registration logic) get a small `_handle_*` method instead. Frozen (MappingProxyType /
    # tuples) so nothing can accidentally mutate a registry shared by the class and every instance.
    _EXACT_KEYWORD_HANDLERS: MappingProxyType = MappingProxyType(
        {
            "VERSION": "_handle_version",
            "FDSVERSION": "_handle_version",
            "CHID": "_handle_chid",
            "TITLE": "_handle_title",
            "TIMES": "_handle_times",
            "CSVF": "_handle_csvf",
            "HRRPUVCUT": "_handle_hrrpuvcut",
            "TOFFSET": "_handle_toffset",
            "CLASS_OF_PARTICLES": "_handle_class_of_particles",
            "CLASS_OF_HUMANS": "_handle_class_of_humans",
            "SURFACE": "_handle_surface",
            "DEVICE": "_handle_device",
        }
    )

    _PREFIX_KEYWORD_HANDLERS_BEFORE_ISOG: Tuple[Tuple[str, str], ...] = (
        ("GEOM", "_load_geoms"),
        ("GRID", "_handle_grid"),
        ("DEVICE_ACT", "_handle_device_act"),
        ("SLC", "_load_slice"),
        ("BNDS", "_load_geomslice"),
    )

    _PREFIX_KEYWORD_HANDLERS_AFTER_ISOG: Tuple[Tuple[str, str], ...] = (
        ("PL3D", "_load_plot_3d"),
        ("SMOKG3D", "_load_smoke_3d"),
        ("SMOKF3D", "_load_smoke_3d"),
        ("BNDF", "_handle_bndf"),
        ("BNDC", "_handle_bndc"),
        ("BNDE", "_load_boundary_data_geom"),
        ("PRT5", "_load_particle_data"),
        ("EVA5", "_load_evac_data"),
        ("SHOW_OBST", "_toggle_obst"),
        ("HIDE_OBST", "_toggle_obst"),
    )

    def parse_smv_file(self):
        # Global device list in registration order — used to resolve DEVICE_ACT indices. Lives on
        # self (not a local var) so _handle_device/_handle_device_act can both reach it; the
        # try/finally guarantees it's cleaned up (same lifetime as the original local variable)
        # even if a handler raises partway through parsing.
        self._devices_by_global_index: List[Device] = []
        try:
            with open(self.smv_file_path) as smv_file:
                for line in smv_file:
                    keyword = line.strip()
                    handler_name = self._EXACT_KEYWORD_HANDLERS.get(keyword)
                    if handler_name is None:
                        for prefix, name in self._PREFIX_KEYWORD_HANDLERS_BEFORE_ISOG:
                            if keyword.startswith(prefix):
                                handler_name = name
                                break
                    if handler_name is None and "ISOG" in keyword:
                        handler_name = "_load_isosurface"
                    if handler_name is None:
                        for prefix, name in self._PREFIX_KEYWORD_HANDLERS_AFTER_ISOG:
                            if keyword.startswith(prefix):
                                handler_name = name
                                break
                    if handler_name is not None:
                        getattr(self, handler_name)(smv_file, keyword)
        finally:
            del self._devices_by_global_index

    def _handle_version(self, smv_file: TextIO, keyword: str):
        self.fds_version = smv_file.readline().strip()

    def _handle_chid(self, smv_file: TextIO, keyword: str):
        self.chid = smv_file.readline().strip()

    def _handle_title(self, smv_file: TextIO, keyword: str):
        self.title = smv_file.readline().strip()

    def _handle_times(self, smv_file: TextIO, keyword: str):
        self.times = [float(t.strip()) for t in smv_file.readline().strip().split()]

    def _handle_csvf(self, smv_file: TextIO, keyword: str):
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

    def _handle_hrrpuvcut(self, smv_file: TextIO, keyword: str):
        self.hrrpuv_cutoff = float(smv_file.readline().strip())

    def _handle_toffset(self, smv_file: TextIO, keyword: str):
        offsets = smv_file.readline().strip().split()
        self.default_texture_origin = tuple(float(offsets[i]) for i in range(3))

    def _handle_class_of_particles(self, smv_file: TextIO, keyword: str):
        self._particles.append(self._register_particle(smv_file))

    def _handle_class_of_humans(self, smv_file: TextIO, keyword: str):
        self._evacs.append(self._register_evac(smv_file))

    def _handle_grid(self, smv_file: TextIO, keyword: str):
        self._meshes.append(self._load_mesh(smv_file, keyword))

    def _handle_surface(self, smv_file: TextIO, keyword: str):
        self.surfaces.append(self._load_surface(smv_file))

    def _handle_device(self, smv_file: TextIO, keyword: str):
        device_id, device = self._register_device(smv_file)
        if device_id in self._devices:
            if isinstance(self._devices[device_id], list):
                self._devices[device_id].append(device)
            else:
                self._devices[device_id] = [self._devices[device_id], device]
        else:
            self._devices[device_id] = device
        self._devices_by_global_index.append(device)

    def _handle_device_act(self, smv_file: TextIO, keyword: str):
        idx_str, time_str, value_str = smv_file.readline().split()
        global_index = int(idx_str) - 1  # FDS uses 1-based global device index
        act_time = float(time_str)
        act_state = bool(int(value_str))
        if global_index < len(self._devices_by_global_index):
            self._devices_by_global_index[global_index].add_activation_time(act_time, act_state)

    def _handle_bndf(self, smv_file: TextIO, keyword: str):
        self._load_boundary_data(smv_file, keyword, cell_centered=False)

    def _handle_bndc(self, smv_file: TextIO, keyword: str):
        self._load_boundary_data(smv_file, keyword, cell_centered=True)

    @classmethod
    def _get_pickle_filename(cls, root_path: str, chid: str) -> AnyStr:
        """Get the filename used to save the pickled simulation."""
        return os.path.join(root_path, chid + ".pickle")

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
