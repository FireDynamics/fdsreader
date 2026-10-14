import numpy as np
import pyvista as pv

from fdsreader import Simulation


def main():
    sim = Simulation("./fds_data")

    geom = sim.geom_data.filter_by_quantity("Radiative Heat Flux")[0]
    faces = np.hstack(np.append(np.full((geom.faces.shape[0], 1), 3), geom.faces, axis=1))
    # Plot the values of the last time step on the geometry
    pv.PolyData(geom.vertices, faces).plot(scalars=geom.data[-1], scalar_bar_args={"title": "Radiative Heat Flux"})


if __name__ == "__main__":
    main()
