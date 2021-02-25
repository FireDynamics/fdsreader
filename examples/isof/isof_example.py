import pyvista as pv
from fdsreader import Simulation


def main():
    sim = Simulation("./fds_steckler")

    isosurface = sim.isosurfaces.filter_by_quantity("TEMP")[0]

    vertices, triangles, _ = isosurface.to_global(len(isosurface.times) - 1)

    level1 = isosurface.get_pyvista_mesh(vertices, triangles[2])
    level2 = isosurface.get_pyvista_mesh(vertices, triangles[1])
    # We ignore level 1 as it does not contain any vertices
    # level3 = isosurface.get_pyvista_mesh(vertices, triangles[0])

    # Either plot both meshes directly...
    # isosurface.join_pyvista_meshes([level1, level2]).plot()

    # ...or plot them seperately to adjust properties such as the color
    plotter = pv.Plotter()
    plotter.add_mesh(level1, color=[1, 0, 0, 1], opacity=0.95)
    plotter.add_mesh(level2, color=[0, 1, 0, 1], opacity=0.95)
    plotter.show()


if __name__ == "__main__":
    main()
