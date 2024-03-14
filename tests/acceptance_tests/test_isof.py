from fdsreader import Simulation


def test_isof():
    sim = Simulation("./steckler_data")
    isosurface = sim.isosurfaces.filter_by_quantity("TEMP")[0]
    vertices, triangles, _ = isosurface.to_global(len(isosurface.times) - 1)

    assert abs(vertices[-1][0] - 2.80595016) < 1e-6 and abs(vertices[-1][1] - 0.1) < 1e-6 and abs(vertices[-1][2] - 1.83954549) < 1e-6
    assert abs(triangles[-1][-1][0] - 4625) < 1e-6 and abs(triangles[-1][-1][0] - 4627) < 1e-6 and abs(triangles[-1][-1][0] - 4708) < 1e-6
