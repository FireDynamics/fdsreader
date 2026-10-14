import numpy as np
import pyvista as pv

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_data")

    # Load temperature 3D data
    quantity = "Temperature"
    pl_t1 = sim.data_3d.get_by_quantity(quantity)
    data, coordinates = pl_t1.to_global(masked=True, return_coordinates=True, fill=np.nan)

    # Select the last available timestep
    t = -1

    # Create 3D grid
    x_ = np.linspace(coordinates["x"][0], coordinates["x"][-1], len(coordinates["x"]))
    # y_ = np.array([29])  # when using a slice
    y_ = np.linspace(coordinates["y"][0], coordinates["y"][-1], len(coordinates["y"]))
    z_ = np.linspace(coordinates["z"][0], coordinates["z"][-1], len(coordinates["z"]))
    x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    color_data = data[t, :, :, :]
    # It is also possible to just plot a slice
    # color_data = pl_t1[mesh].data[t, :, 29:30, :]

    # Plot 3D data
    plotter = pv.Plotter()
    plotter.add_mesh(
        pv.PolyData(points),
        scalars=color_data.flatten(),
        opacity=0.3,
        render_points_as_spheres=False,
        point_size=25,
        scalar_bar_args={"title": quantity},
    )
    plotter.show()


if __name__ == "__main__":
    main()
