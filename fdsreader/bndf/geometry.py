from typing import Tuple, Dict

import numpy as np

from fdsreader.utils import Surface, Mesh, Quantity


class GeomBoundary:
    def __init__(self, quantity: Quantity, times: np.ndarray, n_t: int):
        self.quantity = quantity
        self.times = times
        self.n_t = n_t

        self.lower_bounds: Dict[Mesh, np.ndarray] = dict()
        self.upper_bounds: Dict[Mesh, np.ndarray] = dict()

    def _add_data(self, mesh: Mesh, vertices, faces, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        # self._vertices.extend(vertices)
        # self._faces.extend(faces)

        self.lower_bounds[mesh] = lower_bounds
        self.upper_bounds[mesh] = upper_bounds

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

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        pass


class Geometry:
    def __init__(self, file_path: str, texture_mapping: str, texture_origin: Tuple[float, float, float],
                 is_terrain: bool, rgb: Tuple[int, int, int], surface: Surface = None):
        self.file_path = file_path
        self.texture_mapping = texture_mapping
        self.texture_origin = texture_origin
        self.is_terrain = is_terrain
        self.rgb = rgb
        self.surface = surface

        # Maps boundary id to the GeomBoundary object
        self._boundary_data: Dict[int, GeomBoundary] = dict()

    def _add_data(self, bid: int, quantity: str, label: str, unit: str,
                     mesh: Mesh, times: np.ndarray, n_t: int,
                     lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        if bid not in self._boundary_data:
            self._boundary_data[bid] = GeomBoundary(Quantity(quantity, label, unit), times, n_t)
        self._boundary_data[bid]._add_data(mesh, vertices, faces, lower_bounds, upper_bounds)

        if not settings.LAZY_LOAD:
            self._boundary_data[bid].get_patches_in_mesh(mesh)

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        for bndf in self._boundary_data.values():
            bndf.clear_cache()
