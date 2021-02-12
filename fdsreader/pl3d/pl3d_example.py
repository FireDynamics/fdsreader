import pyvista as pv
import numpy as np
import fdsreader as fds


def main():
    sim = fds.Simulation("../../examples/pl3d/fds_data")

    mesh = sim.meshes[1]
    extent = mesh.extent

    pl_t1 = sim.data_3d[-1]

    x_ = np.linspace(extent.x_start, extent.x_end, mesh.dimension['x'])
    y_ = np.linspace(extent.y_start, extent.y_end, mesh.dimension['y'])  # y_ = np.array([29])
    z_ = np.linspace(extent.z_start, extent.z_end, mesh.dimension['z'])

    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    quantity_idx = 0

    plotter = pv.Plotter()

    color_data = pl_t1[mesh].data[:, :, :, quantity_idx]  # pl_t1[mesh].data[:, 29:30, :, quantity_idx]

    plotter.add_mesh(pv.PolyData(points), scalars=color_data.flatten(),
                     opacity=0.3, render_points_as_spheres=False, point_size=25)
    plotter.add_scalar_bar(title=pl_t1.quantities[quantity_idx].quantity)

    plotter.show()


if __name__ == "__main__":
    main()
