from typing import Tuple, Dict, Iterable

import numpy as np

from fdsreader.utils import Surface, Quantity
import fdsreader.utils.fortran_data as fdtype


class GeomBoundary:
    """Boundary data of a specific quantity for all geoms in the simulation.

    :ivar quantity: Quantity object containing information about the quantity calculated for geoms.
    :ivar times: Numpy array containing all times for which data has been recorded.
    :ivar n_t: Total number of time steps for which output data has been written.
    :ivar lower_bounds: Dictionary with lower bounds for each timestep per mesh.
    :ivar upper_bounds: Dictionary with upper bounds for each timestep per mesh.
    """

    def __init__(self, quantity: Quantity, times: np.ndarray, n_t: int):
        self.quantity = quantity
        self.times = times
        self.n_t = n_t

        self.lower_bounds: Dict[int, np.ndarray] = dict()
        self.upper_bounds: Dict[int, np.ndarray] = dict()

        self.file_paths_be: Dict[int, str] = dict()
        self.file_paths_gbf: Dict[int, str] = dict()

    def _add_data(self, mesh: int, file_path_be: str, file_path_gbf: str, lower_bounds: np.ndarray,
                  upper_bounds: np.ndarray):
        self.file_paths_be[mesh] = file_path_be
        self.file_paths_gbf[mesh] = file_path_gbf

        self.lower_bounds[mesh] = lower_bounds
        self.upper_bounds[mesh] = upper_bounds

    def _load_data(self):
        self._vertices: Dict[int, np.ndarray] = dict()
        self._faces: Dict[int, np.ndarray] = dict()
        self._data: Dict[int, np.ndarray] = dict()

        for mesh in self.file_paths_be.keys():
            file_path_be = self.file_paths_be[mesh]
            file_path_gbf = self.file_paths_gbf[mesh]
            # Load .gbf
            with open(file_path_gbf, 'rb') as infile:
                offset = fdtype.INT.itemsize * 2 + fdtype.new(
                    (('i', 3),)).itemsize + fdtype.FLOAT.itemsize
                infile.seek(offset)

                dtype_meta = fdtype.new((('i', 3),))
                n_vertices, n_faces, _ = np.fromfile(infile, dtype_meta, 1)[0][1]

                dtype_vertices = fdtype.new((('f', 3 * n_vertices),))
                vertices = np.fromfile(infile, dtype_vertices, 1)[0][1].reshape(
                    (n_vertices, 3)).astype(float)

                dtype_faces = fdtype.new((('i', 3 * n_faces),))
                faces = fdtype.read(infile, dtype_faces, 1)[0][0].reshape((n_faces, 3)).astype(
                    int) - 1

            # Load .be
            data = np.empty((self.n_t,), dtype=object)
            with open(file_path_be, 'rb') as infile:
                offset = fdtype.INT.itemsize * 2
                infile.seek(offset)

                for t in range(self.n_t):
                    # Skip time value
                    infile.seek(fdtype.FLOAT.itemsize, 1)

                    dtype_meta = fdtype.new((('i', 4),))
                    n_faces = fdtype.read(infile, dtype_meta, 1)[0][0][3]
                    if n_faces > 0:
                        dtype_faces = fdtype.new((('f', n_faces),))
                        data[t] = np.fromfile(infile, dtype_faces, 1)[0][1]

            self._vertices[mesh] = vertices
            self._faces[mesh] = faces
            self._data[mesh] = data

    @property
    def vertices(self) -> Iterable:
        """Returns a global array of the vertices from all meshes.
        """
        if not hasattr(self, "_vertices"):
            self._load_data()

        size = sum(v.shape[0] for v in self._vertices.values())
        ret = np.empty((size, 3), dtype=float)
        counter = 0

        for v in self._vertices.values():
            size = v.shape[0]
            ret[counter:counter + size, :] = v
            counter += size

        return ret

    @property
    def faces(self) -> np.ndarray:
        """Returns a global array of the faces from all meshes.
        """
        if not hasattr(self, "_faces"):
            self._load_data()

        size = sum(f.shape[0] for f in self._faces.values())
        ret = np.empty((size, 3), dtype=int)
        counter = 0
        verts_counter = 0

        for m, f in self._faces.items():
            size = f.shape[0]
            ret[counter:counter + size, :] = f + verts_counter
            counter += size
            verts_counter += self._vertices[m].shape[0]

        return ret

    @property
    def data(self) -> np.ndarray:
        """Returns a global array of the loaded data for the quantity with data from all meshes.
        """
        if not hasattr(self, "_data"):
            self._load_data()

        ret = np.empty((self.n_t,), dtype=object)
        for t in range(self.n_t):
            size = sum(d[t].shape[0] for d in self._data.values())
            counter = 0
            ret[t] = np.empty((size,), dtype=float)

            for d in self._data.values():
                size = d[t].shape[0]
                ret[t][counter:counter + size] = d[t]
                counter += size

        return ret

    @property
    def vmin(self) -> float:
        """Minimum value of all faces at any time.
        """
        curr_min = min(np.min(b) for b in self.lower_bounds.values())
        if curr_min == 0.0:
            return min(min(np.min(p.data) for p in ps) for ps in self._faces.values())
        return curr_min

    @property
    def vmax(self) -> float:
        """Maximum value of all faces at any time.
        """
        curr_max = max(np.max(b) for b in self.upper_bounds.values())
        if curr_max == np.float32(-1e33):
            return max(max(np.max(p.data) for p in ps) for ps in self._faces.values())
        return curr_max

    def __repr__(self, *args, **kwargs):
        return f"GeomBoundary(quantity={self.quantity})"


class Geometry:
    """Obstruction defined as a complex geometry.

    :ivar file_path: Path to the .ge file that defines the geom.
    :ivar texture_map: Path to the texture map of the geometry.
    :ivar texture_origin: Origin position of the texture provided by the surface.
    :ivar is_terrain: Indicates whether the geometry is a regular complex geom or terrain geometry.
    :ivar rgb: Color of the geometry in form of a 3-element tuple.
    :ivar surface: Surface object used for the geometry.
    """

    def __init__(self, file_path: str, texture_mapping: str,
                 texture_origin: Tuple[float, float, float],
                 is_terrain: bool, rgb: Tuple[int, int, int], surface: Surface = None):
        self.file_path = file_path
        self.texture_map = texture_mapping
        self.texture_origin = texture_origin
        self.is_terrain = is_terrain
        self.rgb = rgb
        self.surface = surface

    def __repr__(self, *args, **kwargs):
        return f"Geometry(file_path={self.file_path})"
