from typing import Iterable, Tuple, List
import numpy as np

from fdsreader.bndf import Obstruction
from fdsreader.utils.data import FDSDataCollection, Quantity


class ObstructionCollection(FDSDataCollection):
    """Collection of :class:`Obstruction` objects. Offers extensive functionality for filtering and
        using obstructions as well as dependent such as :class:`Boundary`.
    """

    def __init__(self, *obstructions: Iterable[Obstruction]):
        super().__init__(*obstructions)

    @property
    def quantities(self) -> List[Quantity]:
        qs = set()
        for obst in self._elements:
            for q in obst.quantities:
                qs.add(q)
        return list(qs)

    def filter_by_boundary_data(self):
        """Filters all obstructions for which output data exists.
        """
        return ObstructionCollection(x for x in self._elements if x.has_boundary_data)

    def get_by_id(self, obst_id: str):
        """Get the obstruction with corresponding id if it exists.
        """
        return next((obst for obst in self._elements if obst.id == obst_id), None)

    def get_nearest(self, x: float = None, y: float = None, z: float = None) -> Obstruction:
        """Filters the obstruction with the shortest distance to the given point.
        """
        d_min = np.finfo(float).max
        obst_min = None

        for obst in self:
            for subobst in obst:
                dx = max(subobst.extent.x_start - x, 0, x - subobst.extent.x_end) if x is not None else 0
                dy = max(subobst.extent.y_start - y, 0, y - subobst.extent.y_end) if y is not None else 0
                dz = max(subobst.extent.z_start - z, 0, z - subobst.extent.z_end) if z is not None else 0
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d < d_min:
                    d_min = d
                    obst_min = obst

        return obst_min

    def __repr__(self):
        return "ObstructionCollection(" + super(ObstructionCollection, self).__repr__() + ")"
