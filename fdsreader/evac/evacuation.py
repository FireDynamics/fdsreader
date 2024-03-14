from typing import List, Tuple, Dict, Sequence, Union

import numpy as np

from fdsreader.fds_classes import Mesh
from fdsreader.utils import Quantity


# class Entrance:
#     def __init__(self):
#         self.id = ""
#         self.extent = Extent(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#         self.orientation = 1
#         self.color = (0.0, 0.0, 0.0)


# class Exit:
#     def __init__(self):
#         self.id = ""
#         self.extent = Extent(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#         self.xyz = (1.0, 1.0, 1.0)
#         self.orientation = 1
#         self.color = (0.0, 0.0, 0.0)
#         self.count_only = True
#         self.known_door = True


# class Door:
#     def __init__(self):
#         self.id = ""
#         self.extent = Extent(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
#         self.xyz = (1.0, 1.0, 1.0)
#         self.orientation = 1
#         self.color = (0.0, 0.0, 0.0)
#         self.to_node = Entrance()
#         self.known_door = True
#         self.exit_sign = True


class Evacuation:
    """Container to store evac data from evac simulations with FDS.
        Note: Evac support was removed from FDS in all versions after 6.7.7!

    :ivar class_name: Name of the evac class defined in the FDS input-file.
    :ivar quantities: List of all quantities for which data has been written out.
    :ivar color: Color assigned to the evac.
    :ivar n_humans: Number of existing evacs for each timestep per mesh.
    :ivar lower_bounds: Dictionary with lower bounds for each timestep with quantities as keys.
    :ivar upper_bounds: Dictionary with upper bounds for each timestep with quantities as keys.
    """

    def __init__(self, class_name: str, quantities: List[Quantity], color: Tuple[float, float, float]):
        self.class_name = class_name
        self.quantities = quantities
        self.color = color
        self.n_humans: Dict[str, List[int]] = dict()

        self._positions: List[np.ndarray] = list()
        self._body_angles: List[np.ndarray] = list()
        self._semi_major_axis: List[np.ndarray] = list()
        self._semi_minor_axis: List[np.ndarray] = list()
        self._agent_heights: List[np.ndarray] = list()
        self._tags: List[np.ndarray] = list()
        self._data: Dict[str, List[np.ndarray]] = {q.name: [] for q in self.quantities}
        self.times: Sequence[float] = list()

        self.lower_bounds = {q.name: [] for q in self.quantities}
        self.upper_bounds = {q.name: [] for q in self.quantities}

        self._init_callback = lambda: None

    @property
    def id(self):
        return self.class_name

    def has_quantity(self, quantity: Union[Quantity, str]):
        if type(quantity) == Quantity:
            quantity = quantity.name
        return any(
            q.name.lower() == quantity.lower() or q.short_name.lower() == quantity.lower() for q in self.quantities)

    def filter_by_tag(self, tag: int):
        """Filter all evacs by a single one with the specified tag.
        """
        data = self.data
        tags = self.tags
        positions = self.positions
        evac = Evacuation(self.class_name, self.quantities, self.color)
        evac._tags = tag

        evac._data = {quantity: list() for quantity in data.keys()}
        evac._positions = list()
        evac.times = list()

        for t, tags in enumerate(tags):
            if tag in tags:
                idx = np.where(tags == tag)[0]

                for quantity in data.keys():
                    evac._data[quantity].append(data[quantity][t][idx][0])
                evac._positions.append(positions[t][idx][0])
                evac.times.append(self.times[t])

        evac.lower_bounds = dict()
        evac.upper_bounds = dict()
        for q in self.quantities:
            if len(evac._positions) != 0:
                evac.lower_bounds[q.name] = np.min(evac._data[q.name])
                evac.upper_bounds[q.name] = np.max(evac._data[q.name])
            else:
                evac.lower_bounds[q.name] = 0
                evac.upper_bounds[q.name] = 0

        return evac

    @property
    def data(self) -> Dict[str, List[np.ndarray]]:
        """Dictionary with quantities as keys and a list with a numpy array for each timestep which
            contains data for each person in that timestep.
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._data

    def get_data(self, quantity: Union[Quantity, str]) -> List[np.ndarray]:
        """Returns a list with a numpy array for each timestep which contains data about the specified quantity for
            each person in that timestep.
        """
        if self.has_quantity(quantity):
            if type(quantity) == Quantity:
                quantity = quantity.name
            return self.data[quantity]
        return []

    @property
    def tags(self) -> List[np.ndarray]:
        """List with a numpy array for each timestep which contains a tag for each evac in that
            timestep.
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._tags

    @property
    def positions(self) -> List[np.ndarray]:
        """List with a numpy array for each timestep which contains the position of each evac in
            that timestep.
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._positions

    @property
    def body_angles(self) -> List[np.ndarray]:
        """
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._body_angles

    @property
    def semi_major_axis(self) -> List[np.ndarray]:
        """
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._semi_major_axis

    @property
    def semi_minor_axis(self) -> List[np.ndarray]:
        """
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._semi_minor_axis

    @property
    def agent_heights(self) -> List[np.ndarray]:
        """
        """
        if len(self._positions) == 0 and len(self._tags) == 0:
            self._init_callback()
        return self._agent_heights

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if len(self._positions) != 0:
            del self._positions
            self._positions = list()
            del self._tags
            self._tags = list()
            del self._data
            self._data = {q.name: [] for q in self.quantities}

    def __repr__(self):
        return f"Evacuation(name={self.class_name}, quantities={self.quantities})"
