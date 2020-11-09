from typing import Tuple, Union
from fastcore.basics import store_attr

from utils import Extent, Surface


class Obstruction:
    """

    """
    def __init__(self, oid: int, extent: Extent,
                 side_surfaces: Tuple[Surface, Surface, Surface, Surface, Surface, Surface],
                 bound_indices: Tuple[int, int, int, int, int, int], color_index: int, block_type: int,
                 texture_origin: Tuple[float, float, float],
                 rgba: Union[Tuple[()], Tuple[float, float, float, float]] = ()):
        store_attr(but=('oid',))
        if len(rgba) == 0:
            self.rgba = rgba
        self.id = oid

    def __str__(self, *args, **kwargs):
        return f"{self.id}, " + str(self.extent)
