import time

import pyvista as pv
from fdsreader import Simulation
import numpy as np


def main():
    sim = Simulation("./fds_data")

    geom = sim.geom_data.filter_by_quantity("Radiative Heat Flux")[0]
    print(len(geom.data[-1]))
    exit()

    faces = np.hstack(np.append(np.full((geom.faces.shape[0], 1), 3), geom.faces, axis=1))
    pv.PolyData(geom.vertices, faces).plot(scalars=geom.data[0])


if __name__ == "__main__":
    main()
