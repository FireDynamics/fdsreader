import pyvista as pv
from fdsreader import Simulation


def main():
    sim = Simulation("./fds_steckler")

    isosurface = sim.isosurfaces.filter_by_quantity("TEMP")[0]

    vertices, triangles, _ = isosurface.to_global(len(isosurface.times) - 1)

    # We ignore level 1 as it does not contain any vertices
    # level1 = isosurface.get_pyvista_mesh(vertices, triangles[0])
    level2 = isosurface.get_pyvista_mesh(vertices, triangles[1])
    level3 = isosurface.get_pyvista_mesh(vertices, triangles[2])

    # Either plot both meshes directly...
    # isosurface.join_pyvista_meshes([level2, level3]).plot()

    # ...or plot them separately to adjust properties such as the color
    plotter = pv.Plotter()
    plotter.add_mesh(level2, color=[0, 0, 255, 255], opacity=0.95)
    plotter.add_mesh(level3, color=[255, 0, 0, 255], opacity=0.95)
    plotter.show()


if __name__ == "__main__":
    main()
