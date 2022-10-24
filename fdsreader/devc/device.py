from typing import Tuple, Union

from fdsreader.utils import Quantity


class Device:
    """Represents a single Device.

    :ivar id: The id the device was given.
    :ivar quantity: The :class:`Quantity` the device measured.
    :ivar position: Position of the device in the simulation space.
    :ivar orientation: The direction the device was facing.
    :ivar data: All data the device measured.
    """
    def __init__(self, device_id: str, quantity: Quantity, position: Tuple[float, float, float],
                 orientation: Tuple[float, float, float]):
        self.id = device_id
        self.quantity = quantity
        self.position = position
        self.orientation = orientation
        self._data_callback = lambda: None

    @property
    def data(self):
        if not hasattr(self, "_data"):
            # While this design is suboptimal, there is no other way of doing it at this point in time. When a device
            # is encountered that does not have any data loaded yet, the device-loading function of the Simulation
            # class is called, which is needed to get the path to the device file as well as fill in the data for all
            # other devices as well as we are reading all the data anyway
            self._data_callback()
        return self._data

    @property
    def quantity_name(self):
        """Alias for :class:`Device`.quantity.name.
        """
        return self.quantity.name

    @property
    def unit(self):
        """Alias for :class:`Device`.quantity.unit.
        """
        return self.quantity.unit

    @property
    def xyz(self):
        """Alias for :class:`Device`.position.
        """
        return self.position

    def clear_cache(self):
        """Remove all data from the internal cache that has been loaded so far to free memory.
        """
        if hasattr(self, "_data"):
            del self._data

    def __eq__(self, other):
        if type(other) == str:
            return self.id == other
        return self.id == other.id

    def __repr__(self):
        return f"Device(id='{self.id}', xyz={self.position}, quantity={self.quantity})"

