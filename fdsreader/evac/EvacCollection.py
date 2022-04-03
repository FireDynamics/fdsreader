import logging
import os
from typing import Iterable, Dict, List, Union
import numpy as np

from fdsreader.evac import Evacuation
from fdsreader.fds_classes import Mesh
from fdsreader.utils.data import FDSDataCollection, Quantity
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings


class EvacCollection(FDSDataCollection):
    """Collection of :class:`Evacuation` objects. Next to agent-class specific data (such as trajectories) lots of
        other data such as FED-data is provided via this class.

        :ivar times: List of all time steps of the simulation.
        :ivar z_offsets: The offset in z-direction for each mesh where the evac plane lays.
        :ivar all_agents: Number of all agents per time step.
        :ivar agents_inside_mesh: Number of all agents per time step inside a specific mesh.
        :ivar number_of_deads: Number of dead agents per time step.
        :ivar fed_max: FED max per time step.
        :ivar fed_max_alive: FED max alive per time step.
        :ivar exit_counters: Exit counts per time step.
        :ivar target_exit_counters: Target exit counts per time step.
        :ivar door_counters: Door counts per time step.
        :ivar target_door_counters: Target door counts per time step.
    """

    def __init__(self, evacs: Iterable[Evacuation], base_path: str, times: Iterable[float]):
        super().__init__(evacs)

        self._file_paths: Dict[Mesh, str] = dict()
        self.z_offsets: Dict[Mesh, float] = dict()
        self._base_path = base_path

        self._load_csv_data()

        self.times = times

        for evac in evacs:
            evac.times = self.times

        for evac in self:
            if settings.LAZY_LOAD:
                evac._init_callback = self._load_prt_data
            else:
                self._load_xyz_data()
                self._load_eff_data()
                self._load_prt_data()

    @property
    def quantities(self) -> List[Quantity]:
        """Gives a list of all quantities that are written out for some (or sometimes all) human classes.
        """
        qs = set()
        for evac in self:
            for q in evac.quantities:
                qs.add(q)
        return list(qs)

    def _load_csv_data(self):
        file_path = self._base_path + ".csv"
        if not os.path.exists(file_path):
            return
        with open(file_path, 'r') as infile:
            units = [unit.replace('"', '').replace('\n', '').strip() for unit in infile.readline().split(',')]
            names = [name.replace('"', '').replace('\n', '').strip() for name in infile.readline().split(',')]
            values = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
            dtypes = [int] * len(names)
            dtypes[0] = float
            dtypes[-2:] = (float, float)

            size = values.shape[0]
            data = {names[i]: np.empty((size,), dtype=dtypes[i]) for i in range(len(names))}
            for k, arr in enumerate(data.values()):
                for i in range(size):
                    arr[i] = values[i][k]

            self.times = data["EVAC_Time"]
            self.all_agents = data["AllAgents"]
            self.agents_inside_mesh = {names[i]: data[names[i]] for i in range(len(names)) if
                                       units[i] == "AgentsInsideMesh"}
            self.number_of_deads = data["Number_of_Deads"]
            self.fed_max = data["FED_max"]
            self.fed_max_alive = data["FED_max_alive"]

            self.exit_counters = {names[i]: data[names[i]] for i in range(len(names)) if units[i] == "ExitCounter"}
            self.target_exit_counters = {names[i]: data[names[i]] for i in range(len(names)) if
                                         units[i] == "TargetExitCounter"}

            self.door_counters = {names[i]: data[names[i]] for i in range(len(names)) if units[i] == "DoorCounter"}
            self.target_door_counters = {names[i]: data[names[i]] for i in range(len(names)) if
                                         units[i] == "TargetDoorCounter"}

    @property
    def xyz(self) -> List[np.ndarray]:
        """List of xyz-data for each mesh.
        """
        if not hasattr(self, "_xyz"):
            self._load_xyz_data()
        return self._xyz

    def _load_xyz_data(self):
        file_path = self._base_path + ".xyz"

        if not os.path.exists(file_path):
            self._xyz = np.array([])
            self._load_fed_data(0)
            return

        with open(file_path, 'rb') as infile:
            dtype_meta = fdtype.new((('i', 6),))
            dtype_grid_meta = fdtype.new((('i', 4),))
            dtype_grid_data = fdtype.new((('f', 3),))

            file_format = fdtype.read(infile, fdtype.INT, 1)[0][0][0]
            if file_format != -4:
                logging.warning("The evac xyz file (" + file_path + ") was written in an unsupported file format. "
                                                                    "Please submit an issue on Github!")

            meta = fdtype.read(infile, dtype_meta, 1)[0][0]
            n_grids = meta[0]
            n_corrs = meta[2]
            n_devc = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            n_i = list()
            n_j = list()
            for g in range(n_grids):
                n_i_g, n_j_g, _, _ = fdtype.read(infile, dtype_grid_meta, 1)[0][0]
                n_i.append(n_i_g)
                n_j.append(n_j_g)
                for i in range(n_i_g):
                    for j in range(n_j_g):
                        _ = fdtype.read(infile, dtype_grid_data, 1)

            # Reset pointer (after file header)
            infile.seek(fdtype.INT.itemsize * 2 + dtype_meta.itemsize)

            self._xyz = [np.empty((n_i[g], n_j[g], 3)) for g in range(n_grids)]

            for g in range(n_grids):
                infile.seek(dtype_grid_meta.itemsize, 1)
                for i in range(n_i[g]):
                    for j in range(n_j[g]):
                        self._xyz[g][i, j] = fdtype.read(infile, dtype_grid_data, 1)[0][0]

        if os.path.exists(self._base_path + ".fed"):
            self._load_fed_data(n_corrs)

    @property
    def fed_grid(self) -> Dict[str, List[np.ndarray]]:
        """
        """
        if not hasattr(self, "_fed_grid"):
            self._load_xyz_data()
        return self._fed_grid

    @property
    def fed_corr(self) -> Dict[str, List[np.ndarray]]:
        """
        """
        if not hasattr(self, "_fed_corr"):
            self._load_xyz_data()
        return self._fed_corr

    @property
    def devc(self) -> Dict[str, Union[List[np.ndarray], np.ndarray]]:
        """
        """
        if not hasattr(self, "_devc"):
            self._load_xyz_data()
        return self._devc

    @property
    def fed_times(self) -> np.ndarray:
        """
        """
        if not hasattr(self, "_fed_times"):
            self._load_xyz_data()
        return self._fed_times

    def _load_fed_data(self, n_corrs: int):
        file_path = self._base_path + ".fed"

        if not os.path.exists(file_path):
            self._fed_grid = dict(co_co2_o2=[], soot_dens=[], tmp_g=[], radflux=[])
            self._fed_corr = dict(co_co2_o2=[], soot_dens=[], tmp_g=[], radflux=[])
            self._devc = dict(current=[], prior=[], t_change=[], times=np.array([]))
            self._fed_times = np.array([])
            return

        with open(file_path, 'rb') as infile:
            dtype_meta = fdtype.new((('i', 6),))
            dtype_time = fdtype.new((('f', 2),))
            dtype_grid_meta = fdtype.new((('i', 4),))
            dtype_corr = fdtype.new((('f', 8),))
            dtype_devs_meta = fdtype.new((('i', 2),))
            dtype_devs_data = fdtype.new((('i', 1), ('f', 1), ('i', 2), ('f', 1)))

            # File header
            file_format = fdtype.read(infile, fdtype.INT, 1)[0][0][0]
            if file_format != -4:
                logging.warning("The evac fed file (" + file_path + ") was written in an unsupported file format. "
                                                                    "Please submit an issue on Github!")

            n_grids = fdtype.read(infile, dtype_meta, 1)[0][0][0]
            n_devc = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            # Read gridsize from first timestep header
            infile.seek(dtype_time.itemsize, 1)

            n_i = list()
            n_j = list()
            n = list()
            dtype_grid_data = list()
            for g in range(n_grids):
                n_i_g, n_j_g, _, n_g = fdtype.read(infile, dtype_grid_meta, 1)[0][0]
                n_i.append(n_i_g)
                n_j.append(n_j_g)
                n.append(n_g)
                dtype_grid_data.append(fdtype.new((('f', n_g),)))
                for i in range(n_i_g):
                    for j in range(n_j_g):
                        _ = fdtype.read(infile, dtype_grid_data[-1], 1)

            # Reset pointer (after file header)
            infile.seek(dtype_meta.itemsize + fdtype.INT.itemsize * 2)

            n_t = (os.stat(file_path).st_size - (fdtype.INT.itemsize * 2 + dtype_meta.itemsize)) // (
                        dtype_time.itemsize + sum(
                    (dtype_grid_meta.itemsize + n_i[g] * n_j[g] * dtype_grid_data[g].itemsize) for g in
                    range(n_grids)) + n_corrs * dtype_corr.itemsize + fdtype.FLOAT.itemsize + n_devc * (
                                    dtype_devs_meta.itemsize + dtype_devs_data.itemsize))

            times = list()
            self._fed_grid = dict(co_co2_o2=[np.empty((n_t, n_i[g], n_j[g])) for g in range(n_grids)],
                                  soot_dens=[np.empty((n_t, n_i[g], n_j[g])) for g in range(n_grids)],
                                  tmp_g=[np.empty((n_t, n_i[g], n_j[g])) for g in range(n_grids)],
                                  radflux=[np.empty((n_t, n_i[g], n_j[g])) for g in range(n_grids)])
            self._fed_corr = dict(co_co2_o2=[np.empty((n_t, 2)) for _ in range(n_corrs)],
                                  soot_dens=[np.empty((n_t, 2)) for _ in range(n_corrs)],
                                  tmp_g=[np.empty((n_t, 2)) for _ in range(n_corrs)],
                                  radflux=[np.empty((n_t, 2)) for _ in range(n_corrs)])
            self._devc = dict(i_type=[0 for _ in range(n_devc)], devc_id=[0 for _ in range(n_devc)],
                              current=[np.empty((n_t,), dtype=int) for _ in range(n_devc)],
                              prior=[np.empty((n_t,), dtype=int) for _ in range(n_devc)],
                              t_change=[np.empty((n_t,)) for _ in range(n_devc)], times=np.empty((n_t,)))

            for t in range(n_t):
                times.append(fdtype.read(infile, dtype_time, 1)[0][0])  # t, dt_save

                for g in range(n_grids):
                    infile.seek(dtype_grid_meta.itemsize, 1)
                    for i in range(n_i[g]):
                        for j in range(n_j[g]):
                            co_co2_o2, soot_dens, tmp_g, radflux = fdtype.read(infile, dtype_grid_data[g], 1)[0][0][:4]
                            self._fed_grid["co_co2_o2"][g][t, i, j] = co_co2_o2
                            self._fed_grid["soot_dens"][g][t, i, j] = soot_dens
                            self._fed_grid["tmp_g"][g][t, i, j] = tmp_g
                            self._fed_grid["radflux"][g][t, i, j] = radflux

                # Corr
                for c in range(n_corrs):
                    co_co2_o2_1, soot_dens_1, tmp_g_1, radflux_1, co_co2_o2_2, soot_dens_2, tmp_g_2, radflux_2 = \
                        fdtype.read(infile, dtype_corr, 1)[0][0][:8]
                    self._fed_corr["co_co2_o2"][c][t] = (co_co2_o2_1, co_co2_o2_2)
                    self._fed_corr["soot_dens"][c][t] = (soot_dens_1, soot_dens_2)
                    self._fed_corr["tmp_g"][c][t] = (tmp_g_1, tmp_g_2)
                    self._fed_corr["radflux"][c][t] = (radflux_1, radflux_2)

                self._devc["times"][t] = fdtype.read(infile, fdtype.FLOAT, 1)[0][0]
                for d in range(n_devc):
                    i_type, devc_id = fdtype.read(infile, dtype_devs_meta, 1)[0][0]
                    self._devc["i_type"][d] = i_type
                    self._devc["devc_id"][d] = devc_id
                    _, _, current, prior, t_change = fdtype.read(infile, dtype_devs_data, 1)[0]
                    self._devc["current"][d][t] = current[0]
                    self._devc["prior"][d][t] = prior[0]
                    self._devc["t_change"][d][t] = t_change[0]

            self._fed_times = np.array(times)

    @property
    def eff(self):
        if not hasattr(self, "_eff"):
            self._load_eff_data()
        return self._eff

    def _load_eff_data(self):
        file_path = self._base_path + ".eff"
        if not os.path.exists(file_path):
            self._eff = None
            return

        with open(file_path, 'rb') as infile:
            dtype_grid_meta = fdtype.new((('i', 3),))
            dtype_grid_data = fdtype.new((('f', 2),))  # u, v

            n_grids = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            # n_fields = (os.stat(file_path).st_size - fdtype.INT.itemsize) // (
            #         sum(dtype_grid_meta.itemsize + n_i[g] * n_j[g] * dtype_grid_data.itemsize for g in range(n_grids)))
            #
            # self._eff = np.empty((n_grids, n_fields, n_i, n_j, 2))
            # for g in range(n_grids):
            #     for f in range(n_fields):
            #         infile.seek(dtype_grid_meta.itemsize, 1)
            #         for i in range(n_i[g]):
            #             for j in range(n_j[g]):
            #                 self._eff[g, f, i, j] = fdtype.read(infile, dtype_grid_data, 1)[0][0]

            self._eff = list()
            meta = fdtype.read(infile, dtype_grid_meta, 1)
            while len(meta) > 0:
                n_i, n_j, _ = meta[0][0]
                for i in range(n_i + 2):
                    for j in range(n_j + 2):
                        self._eff.append(fdtype.read(infile, dtype_grid_data, 1)[0][0])
                meta = fdtype.read(infile, dtype_grid_meta, 1)

    def get_unfiltered_positions(self, quantity: Union[Quantity, str] = None):
        """Convenience function that combines all trajectories into a single array.

        :param quantity: Optionally a quantity can be specified to filter all human classes by those who have written
            out data for the specific quantity. Can either be a string or :class:`Quantity` object.
        """
        filtered_evacs = [evac for evac in self if evac.has_quantity(quantity)] if quantity is not None else self

        combined_positions = list()

        for t in range(len(self.times)):
            size = 0
            for evac in filtered_evacs:
                size += evac.positions[t].shape[0]

            combined_positions.append(np.empty((size, 3)))
            if size != 0:
                size = 0
                for evac in filtered_evacs:
                    pos = evac.positions[t]
                    combined_positions[t][size:size + pos.shape[0]] = pos
                    size += pos.shape[0]

        return combined_positions

    def get_unfiltered_data(self, quantity: Union[Quantity, str]):
        """Convenience function that combines data for a specific quantity of all humans into a single array.

        :param quantity: The quantity to get data for. Can either be a string or :class:`Quantity` object.
        """
        filtered_evacs = [evac for evac in self if evac.has_quantity(quantity)]

        combined_data = list()
        data = list()
        for evac in filtered_evacs:
            data.append(evac.get_data(quantity))

        for t in range(len(self.times)):
            size = 0
            for d in data:
                size += d[t].shape[0]

            combined_data.append(np.empty((size,)))
            if size != 0:
                size = 0
                for d in data:
                    combined_data[t][size:size + d[t].shape[0]] = d[t]
                    size += d[t].shape[0]

        return combined_data

    def _load_prt_data(self):
        """Function to read in all evac data for a simulation.
        """
        evacs = self
        pointer_location = {evac: [0] * len(self.times) for evac in evacs}

        for evac in evacs:
            if len(evac._positions) == 0 and len(evac._tags) == 0:
                for t in range(len(self.times)):
                    size = 0
                    for mesh in self._file_paths.keys():
                        size += evac.n_humans[mesh][t]
                    for quantity in evac.quantities:
                        evac._data[quantity.name].append(np.empty((size,), dtype=np.float32))
                    evac._positions.append(np.empty((size, 3), dtype=np.float32))
                    evac._body_angles.append(np.empty((size,), dtype=np.float32))
                    evac._semi_major_axis.append(np.empty((size,), dtype=np.float32))
                    evac._semi_minor_axis.append(np.empty((size,), dtype=np.float32))
                    evac._agent_heights.append(np.empty((size,), dtype=np.float32))
                    evac._tags.append(np.empty((size,), dtype=int))

        for mesh, file_path in self._file_paths.items():
            with open(file_path, 'rb') as infile:
                # Initial offset (ONE, fds version and number of evac classes)
                offset = 3 * fdtype.INT.itemsize
                # Number of quantities for each evac class (plus an INTEGER_ZERO)
                offset += fdtype.new((('i', 2),)).itemsize * len(evacs)
                # 30-char long name and unit information for each quantity
                offset += fdtype.new((('c', 30),)).itemsize * 2 * sum(
                    [len(evac.quantities) for evac in evacs])
                infile.seek(offset)

                for t in range(len(self.times)):
                    # Skip time value and meta data
                    infile.seek(fdtype.FLOAT.itemsize, 1)

                    # Read data for each evac class
                    for evac in evacs:
                        # Read number of evacs in each class
                        n_humans = fdtype.read(infile, fdtype.INT, 1)[0][0][0]
                        offset = pointer_location[evac][t]
                        # Read positions
                        dtype_positions = fdtype.new((('f', 7 * n_humans),))
                        pos = fdtype.read(infile, dtype_positions, 1)[0][0].reshape((n_humans, 7),
                                                                                    order='F').astype(float)
                        evac._positions[t][offset: offset + n_humans] = pos[:, :3]
                        evac._body_angles[t][offset: offset + n_humans] = pos[:, 3]
                        evac._semi_major_axis[t][offset: offset + n_humans] = pos[:, 4]
                        evac._semi_minor_axis[t][offset: offset + n_humans] = pos[:, 5]
                        evac._agent_heights[t][offset: offset + n_humans] = pos[:, 6]
                        # Read tags
                        dtype_tags = fdtype.new((('i', n_humans),))
                        evac._tags[t][offset: offset + n_humans] = fdtype.read(infile, dtype_tags, 1)[0][0]

                        # Read actual quantity values
                        if len(evac.quantities) > 0:
                            dtype_data = fdtype.new(
                                (('f', str((n_humans, len(evac.quantities)))),))
                            data_raw = fdtype.read(infile, dtype_data, 1)[0][0].reshape(
                                (n_humans, len(evac.quantities)), order='F')

                            for q, quantity in enumerate(evac.quantities):
                                evac._data[quantity.name][t][
                                offset:offset + n_humans] = data_raw[:, q].astype(float)
                        pointer_location[evac][t] += evac.n_humans[mesh][t]

    def __getitem__(self, key):
        """Get the evac data for a specific agent/human class.
        """
        if type(key) == int:
            return self._elements[key]
        for evac in self:
            if evac.class_name == key:
                return evac

    def __contains__(self, value):
        if value in self._elements:
            return True
        for evac in self:
            if evac.class_name == value:
                return True
        return False

    def __repr__(self):
        return "EvacCollection(" + super(EvacCollection, self).__repr__() + ")"
