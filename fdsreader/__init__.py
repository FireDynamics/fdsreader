from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fdsreader")
except PackageNotFoundError:
    __version__ = "unknown"

from . import settings as settings
from .simulation import Simulation as Simulation
