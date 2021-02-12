import pyvista as pv
import numpy as np
from fdsreader import Simulation


def main():
    sim = Simulation("../../examples/isof/fds_steckler")

    isosurface = sim.isosurfaces.get_by_quantity("TEMP")

    vertices, triangles = isosurface.to_global(len(isosurface.times) - 1)

    plotter = pv.Plotter()

    tris1 = np.hstack(np.append(np.full((triangles[1].shape[0], 1), 3), triangles[1], axis=1))
    plotter.add_mesh(pv.PolyData(vertices, tris1), color=[1, 0, 0, 1], opacity=0.95)

    tris2 = np.hstack(np.append(np.full((triangles[2].shape[0], 1), 3), triangles[2], axis=1))
    plotter.add_mesh(pv.PolyData(vertices, tris2), color=[0, 1, 0, 1], opacity=0.95)
    plotter.show()


if __name__ == "__main__":
    main()
