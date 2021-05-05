from typing import List, Tuple, Dict, Sequence

import numpy as np

from fdsreader.utils import Quantity, Mesh


class Particle:
    """Container to store particle data from particle simulations with FDS.

    :ivar class_name: Name of the particle class defined in the FDS input-file.
    :ivar quantities: List of all quantities for which data has been written out.
    :ivar color: Color assigned to the particle.
    :ivar n_particles: Number of existing particles for each timestep per mesh.
    :ivar lower_bounds: Dictionary with lower bounds for each timestep with quantities as keys.
    :ivar upper_bounds: Dictionary with upper bounds for each timestep with quantities as keys.
    """

    def __init__(self, class_name: str, quantities: List[Quantity], color: Tuple[float, float, float]):
        self.class_name = class_name
        self.quantities = quantities
        self.color = color
        self.n_particles: Dict[Mesh, List[int]] = dict()

        self._positions: List[np.ndarray] = list()
        self._tags: List[np.ndarray] = list()
        self._data: Dict[str, List[np.ndarray]] = {q.quantity: [] for q in self.quantities}
        self.times: Sequence[float] = list()

        self.lower_bounds = {q.quantity: [] for q in self.quantities}
        self.upper_bounds = {q.quantity: [] for q in self.quantities}

        self._init_callback = None

    @property
    def id(self):
        return self.class_name

    def filter_by_tag(self, tag: int):
        """Filter all particles by a single one with the specified tag.
        """
        data = self.data
        tags = self.tags
        positions = self.positions
        part = Particle(self.class_name, self.quantities, self.color)
        part._tags = tag

        part._data = {quantity: list() for quantity in data.keys()}
        part._positions = list()
        part.times = list()

        for t, tags in enumerate(tags):
            if tag in tags:
                idx = np.where(tags == tag)[0]

                for quantity in data.keys():
                    part._data[quantity].append(data[quantity][t][idx][0])
                part._positions.append(positions[t][idx][0])
                part.times.append(self.times[t])

        part.lower_bounds = dict()
        part.upper_bounds = dict()
        for q in self.quantities:
            if len(part._positions) != 0:
                part.lower_bounds[q.quantity] = np.min(part._data[q.quantity])
                part.upper_bounds[q.quantity] = np.max(part._data[q.quantity])
            else:
                part.lower_bounds[q.quantity] = 0
                part.upper_bounds[q.quantity] = 0

        return part

    @property
    def data(self) -> Dict[str, List[np.ndarray]]:
        """Dictionary with quantities as keys and a list with a numpy array for each timestep which
            contains data for each particle in that timestep.
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._data

    @property
    def tags(self) -> List[np.ndarray]:
        """List with a numpy array for each timestep which contains a tag for each particle in that
            timestep.
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._tags

    @property
    def positions(self) -> List[np.ndarray]:
        """List with a numpy array for each timestep which contains the position of each particle in
            that timestep.
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._positions

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        del self._data

    def __repr__(self):
        return f"Particle(name={self.class_name}, quantities={self.quantities})"
