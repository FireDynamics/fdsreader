import time

import pyvista as pv
from fdsreader import Simulation
import numpy as np


def main():
    sim = Simulation("fds_geomslices")

    for slc in sim.geomslices:
        print(slc.vertices, slc.faces, slc.data)


if __name__ == "__main__":
    main()
