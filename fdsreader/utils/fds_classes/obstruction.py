from typing import Tuple, Union

from fdsreader.utils import Extent, Surface


class Obstruction:
    """A box-shaped obstruction with specific surfaces (materials) on each side.

    :ivar id: ID of the obstruction.
    :ivar extent: Tuple with three tuples containing minimum and maximum coordinate value on the
        corresponding dimension together forming a cuboid.
    :ivar side_surfaces: Tuple of six surfaces for each side of the cuboid.
    :ivar bound_indices: Indices used to define obstruction bounds in terms of mesh locations.
    :ivar color_index: Type of coloring used to color obstruction.
     \n-1 - default color
     \n-2 - invisible
     \n-3 - use red, green, blue and alpha (rgba attribute)
     \nn>0 - use n’th color table entry
    :ivar block_type: Defines how the obstruction is drawn.
     \n-1 - use surface to obtain blocktype
     \n0 - regular block
     \n2 - outline
    :ivar texture_origin: Origin position of the texture provided by the surface. When the texture does have a pattern,
        for example windows or bricks, the texture_origin specifies where the pattern should begin.
    :ivar rgba: Optional color of the obstruction in form of a 4-element tuple (ranging from 0.0 to 1.0).
    """
    def __init__(self, oid: int, extent: Extent,
                 side_surfaces: Tuple[Surface, Surface, Surface, Surface, Surface, Surface],
                 bound_indices: Tuple[int, int, int, int, int, int], color_index: int, block_type: int,
                 texture_origin: Tuple[float, float, float],
                 rgba: Union[Tuple[()], Tuple[float, float, float, float]] = ()):
        self.id = oid
        self.extent = extent
        self.side_surfaces = side_surfaces
        self.bound_indices = bound_indices
        self.color_index = color_index
        self.block_type = block_type
        self.texture_origin = texture_origin
        if len(rgba) != 0:
            self.rgba = rgba

    def __str__(self, *args, **kwargs):
        return f"{self.id}, " + str(self.extent)
