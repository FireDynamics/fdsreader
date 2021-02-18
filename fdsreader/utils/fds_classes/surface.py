from typing import Optional, Tuple


class Surface:
    """Surface objects describe what bounding surfaces consist of. Boundary conditions for
        obstructions and vents are prescribed by referencing the appropriate surface.

    :ivar name: Name of the surface.
    :ivar material_emissivity: Emissivity of the material.
    :ivar surface_type: Type of the surface.
    :ivar texture_width:Width of the texture of the surface.
    :ivar texture_height: Height of the texture of the surface.
    :ivar texture_map: Path to the texture map used for the surface.
    :ivar rgb: Color of the surface in form of a 3-element tuple.
    :ivar transparency: Transparency of the color (alpha channel).
    """

    def __init__(self, name: str, tmpm: float, material_emissivity: float, surface_type: int,
                 texture_width: float, texture_height: float, texture_map: Optional[str],
                 rgb: Tuple[float, float, float], transparency: float):
        self.name = name
        self.tmpm = tmpm
        self.material_emissivity = material_emissivity
        self.surface_type = surface_type
        self.texture_width = texture_width
        self.texture_height = texture_height
        self.texture_map = texture_map
        self.rgb = rgb
        self.transparency = transparency

    def id(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self, *args, **kwargs):
        return f'Surface(name="{self.name}")'
