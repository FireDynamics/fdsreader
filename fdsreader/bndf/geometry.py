from typing import Tuple, Dict, Iterable

import numpy as np

from fdsreader.utils import Surface, Quantity


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

        self._vertices: Dict[int, np.ndarray] = dict()
        self._faces: Dict[int, np.ndarray] = dict()
        self._data: Dict[int, np.ndarray] = dict()

    def _add_data(self, mesh: int, vertices: np.ndarray, faces: np.ndarray, data: np.ndarray,
                  lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self._vertices[mesh] = vertices
        self._faces[mesh] = faces
        self._data[mesh] = data

        self.lower_bounds[mesh] = lower_bounds
        self.upper_bounds[mesh] = upper_bounds

    @property
    def vertices(self) -> Iterable:
        """Returns a global array of the vertices from all meshes.
        """
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
        size = sum(f.shape[0] for f in self._faces.values())
        ret = np.empty((size, 3), dtype=int)
        counter = 0

        for f in self._faces.values():
            size = f.shape[0]
            ret[counter:counter+size, :] = f + counter
            counter += size

        return ret

    @property
    def data(self) -> np.ndarray:
        """Returns a global array of the loaded data for the quantity with data from all meshes.
        """
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


class Geometry:
    """Obstruction defined as a complex geometry.
    """
    def __init__(self, file_path: str, texture_mapping: str,
                 texture_origin: Tuple[float, float, float],
                 is_terrain: bool, rgb: Tuple[int, int, int], surface: Surface = None):
        self.file_path = file_path
        self.texture_mapping = texture_mapping
        self.texture_origin = texture_origin
        self.is_terrain = is_terrain
        self.rgb = rgb
        self.surface = surface
