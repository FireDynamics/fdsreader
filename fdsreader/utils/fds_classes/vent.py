from typing import Union, Tuple

from fastcore.basics import store_attr

from . import Surface, Extent


class Ventilation:
    """

    """
    def __init__(self, extent: Extent, vid: int, surface: Surface, bound_indices: Tuple[int, int, int, int, int, int],
                 color_index: int, draw_type: int, rgba: Union[Tuple[()], Tuple[float, float, float, float]] = (),
                 texture_origin: Union[Tuple[()], Tuple[float, float, float]] = (),
                 circular_vent_origin: Union[Tuple[()], Tuple[float, float, float]] = (), radius: float = -1,
                 open_time: float = -1, close_time: float = -1):
        store_attr(but=('vid',))
        self.id = vid
        if len(rgba) != 0:
            self.rgba = rgba
        if len(texture_origin) != 0:
            self.texture_origin = texture_origin
        if len(circular_vent_origin) != 0:
            self.circular_vent_origin = circular_vent_origin
        if radius != -1:
            self.radius = radius

    def __str__(self, *args, **kwargs):
        return f"{self.id}, " + str(self.extent)
