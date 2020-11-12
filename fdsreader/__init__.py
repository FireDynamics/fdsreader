import logging

logging.getLogger('name.of.library').addHandler(logging.NullHandler())

from .simulation import Simulation
