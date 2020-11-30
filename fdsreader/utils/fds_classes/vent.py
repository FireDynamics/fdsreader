from typing import Union, Tuple

from fdsreader.utils import Surface, Extent


class Ventilation:
    """
    A ventilation can be used to model components of the ventilation system in a building, like a diffuser or a return.
    A ventilation can also be used as a means of applying a particular boundary condition to a rectangular patch on a
     solid surface.

    :ivar id: ID of the ventilation.
    :ivar extent: Tuple with three tuples containing minimum and maximum coordinate value on the
     corresponding dimension.
    :ivar surface: Surface object used for the ventilation.
    :ivar bound_indices: Indices used to define ventilation bounds in terms of mesh locations.
    :ivar color_index: Type of coloring used to color ventilation.
     -99 or +99 - use default color
     -n or +n - use n’th palette color
     < 0 - do not draw boundary file over vent
     > 0 - draw boundary file over vent
    :ivar draw_type: Defines how the ventilation is drawn.
     0 - solid surface
     2 - outline
     -2 - hidden
    :ivar open_time: Point in time the ventilation should open.
    :ivar close_time: Point in time the ventilation should close.
    :ivar rgba: Color of the ventilation in form of a 4-element tuple (ranging from 0.0 to 1.0).
    :ivar texture_origin: Origin position of the texture provided by the surface. When the texture does have a pattern,
     for example windows or bricks, the texture_origin specifies where the pattern should begin.
    :ivar circular_vent_origin: Origin of the ventilation relative to bounding box.
    :ivar radius: Radius of the ventilation circle.
    """
    def __init__(self, extent: Extent, vid: int, surface: Surface, bound_indices: Tuple[int, int, int, int, int, int],
                 color_index: int, draw_type: int, rgba: Union[Tuple[()], Tuple[float, float, float, float]] = (),
                 texture_origin: Union[Tuple[()], Tuple[float, float, float]] = (),
                 circular_vent_origin: Union[Tuple[()], Tuple[float, float, float]] = (), radius: float = -1,
                 open_time: float = -1, close_time: float = -1):
        self.id = vid
        self.extent = extent
        self.surface = surface
        self.bound_indices = bound_indices
        self.color_index = color_index
        self.draw_type = draw_type
        self.open_time = open_time
        self.close_time = close_time

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
