import logging
import os
from typing import Iterable, Dict, List
import numpy as np

from fdsreader.evac import Evacuation
from fdsreader.utils import Mesh
from fdsreader.utils.data import FDSDataCollection, Quantity
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings


class EvacCollection(FDSDataCollection):
    """Collection of :class:`Plot3D` objects. Offers extensive functionality for filtering and
        using plot3Ds as well as its subclasses such as :class:`SubPlot3D`.
    """

    def __init__(self, times: Iterable[float], evacs: Iterable[Evacuation], base_path: str):
        super().__init__(evacs)
        self.times = list(times)
        self._file_paths: Dict[Mesh, str] = dict()
        self.z_offsets: Dict[Mesh, float] = dict()
        self.base_path = base_path

        for evac in evacs:
            evac.times = times

        for evac in self:
            if settings.LAZY_LOAD:
                evac._init_callback = self._load_prt_data()
            else:
                self._load_csv_data()
                self._load_xyz_data()
                self._load_eff_data()
                self._load_prt_data()

    @property
    def quantities(self) -> List[Quantity]:
        qs = set()
        for evac in self:
            for q in evac.quantities:
                qs.add(q)
        return list(qs)

    @property
    def meta(self):
        if not hasattr(self, "_meta"):
            self._load_csv_data()
        return self._meta

    def _load_csv_data(self):
        file_path = self.base_path + ".csv"
        if not os.path.exists(file_path):
            return
        with open(file_path, 'r') as infile:
            units = [unit.replace('"', '').replace('\n', '').strip() for unit in infile.readline().split(',')]
            names = [name.replace('"', '').replace('\n', '').strip() for name in infile.readline().split(',')]
            values = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
            dtypes = [float, int, int, int, int, int, float, float]

            size = values.shape[0]
            self._meta = {
                Quantity(names[i], names[i], units[i]): np.empty((size,), dtype=dtypes[i] if i < 8 else float) for i
                in range(len(names))}
            for k, arr in enumerate(self._meta.values()):
                for i in range(size):
                    arr[i] = values[i][k]

    @property
    def xyz(self):
        if not hasattr(self, "_xyz"):
            self._load_xyz_data()
        return self._xyz

    def _load_xyz_data(self):
        file_path = self.base_path + ".xyz"

        if not os.path.exists(file_path):
            return

        with open(file_path, 'rb') as infile:
            dtype_meta = fdtype.new((('i', 6),))
            dtype_grid_meta = fdtype.new((('i', 4),))

            file_format = fdtype.read(infile, fdtype.INT, 1)[0][0][0]
            if file_format != -4:
                logging.warning("The evac xyz file (" + file_path + ") was written in an unsupported file format. "
                                                                    "Please submit an issue on Github!")

            meta = fdtype.read(infile, dtype_meta, 1)[0][0]
            n_grids = meta[0]
            n_corrs = meta[2]
            n_devc = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            n_i, n_j, n_k, n = fdtype.read(infile, dtype_grid_meta, 1)[0][0]  # n_k will always be 1

            # Reset pointer (after file header)
            infile.seek(fdtype.INT.itemsize * 2 + dtype_meta.itemsize)

            dtype_grid_data = fdtype.new((('f', str((n_i, n_j, 3))),))

            n_t = (os.stat(file_path).st_size - (fdtype.INT.itemsize * 2 + dtype_meta.itemsize)) // \
                  (n_grids * (dtype_grid_meta.itemsize + n_i * n_j * dtype_grid_data.itemsize))

            self._xyz = np.empty((n_t, n_grids, n_i, n_j, 3))
            for t in range(n_t):
                for g in range(n_grids):
                    infile.seek(dtype_grid_meta.itemsize, 1)
                    self._xyz[t, g] = fdtype.read(infile, dtype_grid_data, 1)[0][0].reshape((n_i, n_j, 3), order='F')

        if os.path.exists(self.base_path + ".fed"):
            self._load_fed_data(n_corrs)

    @property
    def fed(self):
        if not hasattr(self, "_fed"):
            self._load_xyz_data()
        return self._fed

    def _load_fed_data(self, n_corrs: int):
        file_path = self.base_path + ".fed"
        with open(file_path, 'rb') as infile:
            dtype_meta = fdtype.new((('i', 6),))
            dtype_time = fdtype.new((('f', 2),))
            dtype_grid_meta = fdtype.new((('i', 4),))

            # File header
            file_format = fdtype.read(infile, fdtype.INT, 1)[0][0][0]
            if file_format != -4:
                logging.warning("The evac fed file (" + file_path + ") was written in an unsupported file format. "
                                                                    "Please submit an issue on Github!")

            n_grids = fdtype.read(infile, dtype_meta, 1)[0][0][0]
            n_devc = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            # Read gridsize from first timestep header
            infile.seek(dtype_time.itemsize, 1)
            n_i, n_j, n_k, n = fdtype.read(infile, dtype_grid_meta, 1)[0][0]  # n_k will always be 1

            # Reset pointer (after file header)
            infile.seek(dtype_meta.itemsize + fdtype.INT.itemsize * 2)

            dtype_grid_data = fdtype.new((('f', n),))
            dtype_corr = fdtype.new((('f', 8),))
            dtype_devs_meta = fdtype.new((('i', 2),))
            dtype_devs_data = fdtype.new((('i', 1), ('f', 1), ('i', 2), ('f', 1)))

            n_t = (os.stat(file_path).st_size - (fdtype.INT.itemsize * 2 + dtype_meta.itemsize)) // \
                  (dtype_time.itemsize + n_grids * (dtype_grid_meta.itemsize + n_i * n_j * dtype_grid_data.itemsize) +
                   n_corrs * dtype_corr.itemsize + fdtype.FLOAT.itemsize + n_devc * (
                               dtype_devs_meta.itemsize + dtype_devs_data.itemsize))

            times = list()
            self._fed = {
                "grid": [dict(co_co2_o2=np.empty((n_t, n_i, n_j)), soot_dens=np.empty((n_t, n_i, n_j)),
                              tmp_g=np.empty((n_t, n_i, n_j)), radflux=np.empty((n_t, n_i, n_j))) for _ in
                         range(n_grids)],
                "corr": [dict(co_co2_o2=np.empty((n_t, 2)), soot_dens=np.empty((n_t, 2)), tmp_g=np.empty((n_t, 2)),
                              radflux=np.empty((n_t, 2))) for _ in range(n_corrs)],
                "devc": [dict(current=np.empty((n_t,), dtype=int), prior=np.empty((n_t,), dtype=int),
                              t_change=np.empty((n_t,))) for _ in range(n_devc)],
                "devc_time": np.empty((n_t,))
            }

            for t in range(n_t):
                times.append(fdtype.read(infile, dtype_time, 1)[0][0])  # t, dt_save

                for g in range(n_grids):
                    infile.seek(dtype_grid_meta.itemsize, 1)
                    # Todo: Make this faster
                    for i in range(n_i):
                        for j in range(n_j):
                            co_co2_o2, soot_dens, tmp_g, radflux = fdtype.read(infile, dtype_grid_data, 1)[0][0][:4]
                            self._fed["grid"][g]["co_co2_o2"][t, i, j] = co_co2_o2
                            self._fed["grid"][g]["soot_dens"][t, i, j] = soot_dens
                            self._fed["grid"][g]["tmp_g"][t, i, j] = tmp_g
                            self._fed["grid"][g]["radflux"][t, i, j] = radflux

                # Corr
                for c in range(n_corrs):
                    co_co2_o2_1, soot_dens_1, tmp_g_1, radflux_1, co_co2_o2_2, soot_dens_2, tmp_g_2, radflux_2 = \
                        fdtype.read(infile, dtype_corr, 1)[0][0][:8]
                    self._fed["corr"][c]["co_co2_o2"][t] = (co_co2_o2_1, co_co2_o2_2)
                    self._fed["corr"][c]["soot_dens"][t] = (soot_dens_1, soot_dens_2)
                    self._fed["corr"][c]["tmp_g"][t] = (tmp_g_1, tmp_g_2)
                    self._fed["corr"][c]["radflux"][t] = (radflux_1, radflux_2)

                self._fed["devc_time"][t] = fdtype.read(infile, fdtype.FLOAT, 1)[0][0]
                for d in range(n_devc):
                    i_type, devc_id = fdtype.read(infile, dtype_devs_meta, 1)[0][0]
                    self._fed["devc"][d]["i_type"] = i_type
                    self._fed["devc"][d]["devc_id"] = devc_id
                    _, _, current, prior, t_change = fdtype.read(infile, dtype_devs_data, 1)[0]
                    self._fed["devc"][d]["current"][t] = current[0]
                    self._fed["devc"][d]["prior"][t] = prior[0]
                    self._fed["devc"][d]["t_change"][t] = t_change[0]

            self._fed["times"] = np.array(times)

    @property
    def eff(self):
        if not hasattr(self, "_eff"):
            self._load_eff_data()
        return self._eff

    def _load_eff_data(self):
        file_path = self.base_path + ".eff"
        if not os.path.exists(file_path):
            return
        with open(file_path, 'rb') as infile:
            dtype_grid_meta = fdtype.new((('i', 3),))

            n_grids = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            n_i, n_j, n_k = fdtype.read(infile, dtype_grid_meta, 1)[0][0]  # n_k will always be 1

            dtype_grid_data = fdtype.new((('f', str((n_i, n_j, 2))),))

            # Reset pointer (after file header)
            infile.seek(fdtype.INT.itemsize)

            n_fields = (os.stat(file_path).st_size - fdtype.INT.itemsize) // (
                        n_grids * (dtype_grid_meta.itemsize + n_i * n_j * dtype_grid_data.itemsize))

            self._eff = np.empty((n_grids, n_fields, n_i, n_j, 2))

            for g in range(n_grids):
                for f in range(n_fields):
                    infile.seek(dtype_grid_meta.itemsize, 1)
                    self._eff[g, f] = fdtype.read(infile, dtype_grid_data, 1)[0][0].reshape((n_i, n_j, 2), order='F')

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
                        pos = fdtype.read(infile, dtype_positions, 1)[0][0]
                        evac._positions[t][offset: offset + n_humans] = pos[:3].reshape((n_humans, 3),
                                                                                        order='F').astype(
                            float)
                        ap = pos[3:]  # Todo: What might this be for?

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
