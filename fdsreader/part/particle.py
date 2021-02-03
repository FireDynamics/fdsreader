from typing import List, Tuple

from fdsreader.utils import Quantity


class Particle:
    """Container to store particle data from particle simulations with FDS.

    :ivar class_name: Name of the particle class defined in the FDS input-file.
    :ivar quantities: List of all quantities for which data has been written out.
    """

    def __init__(self, class_name: str, quantities: List[Quantity], color: Tuple[float, float, float]):
        self.class_name = class_name
        self.quantities = quantities
        self.color = color

        self._positions = list()
        self._tags = list()
        self._data = {q.quantity: [] for q in self.quantities}

        self.lower_bounds = {q.quantity: [] for q in self.quantities}
        self.upper_bounds = {q.quantity: [] for q in self.quantities}

        self._init_callback = None

    @property
    def data(self):
        """
        """
        if len(self._positions) == 0:
            self._init_callback()
        return self._data

    @property
    def tags(self):
        """
        """
        if len(self._positions) == 0:
            self._init_callback()
        return self._tags

    @property
    def positions(self):
        """
        """
        if len(self._positions) == 0:
            self._init_callback()
        return self._positions

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        del self._data

    def __repr__(self):
        return f"Particle(name={self.class_name}, quantities={self.quantities})"
