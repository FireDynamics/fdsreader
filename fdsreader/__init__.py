from . import _version
__version__ = str(_version.Version)

import logging

from .simulation import Simulation

from . import settings

logging.getLogger('name.of.library').addHandler(logging.NullHandler())
