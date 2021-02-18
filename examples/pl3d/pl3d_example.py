import pyvista as pv
import numpy as np
import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_data")

    # Select the seconds mesh (index counting starts at 0)
    mesh = sim.meshes[1]
    extent = mesh.extent

    t = -1
    pl_t1 = sim.data_3d[t]

    # Create 3D grid
    x_ = np.linspace(extent.x_start, extent.x_end, mesh.dimension['x'])
    y_ = np.linspace(extent.y_start, extent.y_end, mesh.dimension['y'])  # y_ = np.array([29])
    z_ = np.linspace(extent.z_start, extent.z_end, mesh.dimension['z'])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    # Select a quantity
    quantity_idx = 0
    quantity = pl_t1.quantities[quantity_idx]

    # Get 3D data for a specific quantity
    color_data = pl_t1[mesh].data[:, :, :, quantity_idx]
    # It is also possible to just plot a slice
    # color_data = pl_t1[mesh].data[:, 29:30, :, quantity_idx]

    # Plot 3D data
    plotter = pv.Plotter()
    plotter.add_mesh(pv.PolyData(points), scalars=color_data.flatten(),
                     opacity=0.3, render_points_as_spheres=False, point_size=25)
    plotter.add_scalar_bar(title=quantity.quantity)
    plotter.show()


if __name__ == "__main__":
    main()
