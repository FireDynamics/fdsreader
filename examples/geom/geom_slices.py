import numpy as np
import pyvista as pv

from fdsreader import Simulation


def main():
    sim = Simulation("./fds_geomslices")

    # SLICETYPE='CUT_CELLS' on &SLCF writes the cells that are cut by the complex geometry
    slc = sim.geomslices[0]
    print(slc)

    # Build a triangle mesh from the cut cells and show it, semi-transparent, around the cone
    faces = np.hstack(np.append(np.full((slc.faces.shape[0], 1), 3), slc.faces, axis=1))
    plotter = pv.Plotter()
    plotter.add_mesh(pv.PolyData(slc.vertices, faces), show_edges=True, color="lightgrey", opacity=0.3)
    geom = sim.geom_data[0]
    geom_faces = np.hstack(np.append(np.full((geom.faces.shape[0], 1), 3), geom.faces, axis=1))
    plotter.add_mesh(pv.PolyData(geom.vertices, geom_faces), color="green")
    plotter.show()


if __name__ == "__main__":
    main()
