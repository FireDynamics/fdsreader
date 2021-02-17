from typing import Iterable, Union, List, Tuple
import numpy as np

from fdsreader.bndf import Boundary, Obstruction
from fdsreader.utils.data import FDSDataCollection, Quantity


class ObstructionCollection(FDSDataCollection):
    """Collection of :class:`Obstruction` objects. Offers extensive functionality for filtering and
        using obstructions as well as dependent such as :class:`Boundary`.
    """

    def __init__(self, *obstructions: Iterable[Obstruction]):
        super().__init__(*obstructions)

    def filter_by_quantity(self, quantity: Union[str, Quantity]) -> List[Boundary]:
        """Filters all obstructions for its boundary data by a specific quantity.
        """
        if type(quantity) != str:
            quantity = quantity.quantity
        return [x.get_boundary_data(quantity) for x in self]

    def get_nearest_obstruction(self, point: Tuple[float, float, float]) -> Obstruction:
        """Filters the obstruction with the shortest distance to the given point.
        """
        d_min = np.finfo(float).min
        obst_min = None

        for obst in self:
            dx = max(obst.extent.x_start - point[0], 0, point[0] - obst.extent.x_end)
            dy = max(obst.extent.y_start - point[1], 0, point[1] - obst.extent.y_end)
            dz = max(obst.extent.z_start - point[2], 0, point[2] - obst.extent.z_end)
            d = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d < d_min:
                d_min = d
                obst_min = obst

        return obst_min

    def get_border_obstructions(self) -> List[Obstruction]:
        """Filters all obstructions with at least one face on the border of a mesh.
        """
        raise NotImplementedError("If you need this feature, please open an issue on GitHub "
                                  "describing your exact needs and your specific use case.")

    def __repr__(self):
        return "ObstructionCollection(" + super(ObstructionCollection, self).__repr__() + ")"
