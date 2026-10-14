import pyvista as pv

from fdsreader import Simulation


def main():
    sim = Simulation("./fds_geomslices")

    # SLICETYPE='CUT_CELLS' on &SLCF writes the cells that are cut by the complex geometry
    slc = sim.geomslices[0]
    print(slc)

    # Build a triangle mesh from the cut cells and show it, semi-transparent, around the cone
    plotter = pv.Plotter()
    plotter.add_mesh(
        pv.PolyData.from_regular_faces(slc.vertices, slc.faces), show_edges=True, color="lightgrey", opacity=0.3
    )
    geom = sim.geom_data[0]
    plotter.add_mesh(pv.PolyData.from_regular_faces(geom.vertices, geom.faces), color="green")
    plotter.show()


if __name__ == "__main__":
    main()
