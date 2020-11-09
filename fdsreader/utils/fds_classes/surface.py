from typing import Optional, Tuple
from fastcore.basics import store_attr


class Surface:
    """

    """
    def __init__(self, name: str, tmpm: float, material_emissivity: float, surface_type: int, texture_width: float,
                 texture_height: float, texture_map: Optional[str], rgb: Tuple[float, float, float],
                 transparency: float):
        store_attr()

    def __str__(self, *args, **kwargs):
        return f'Surface "{self.name}"'
