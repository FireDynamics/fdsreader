import time
from threading import Thread
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_steckler/")

    # Select the seconds mesh (index counting starts at 0)
    mesh = sim.meshes[0]
    extent = mesh.extent

    # Get 3D smoke data for a specific quantity in one of the meshes
    quantity = "Temperature"  # "SOOT MASS FRACTION"
    smoke = sim.smoke_3d.get_by_quantity(quantity)
    data = smoke[mesh].data

    # Create 3D grid
    x_ = np.linspace(extent.x_start, extent.x_end, mesh.dimension['x'])
    y_ = np.linspace(extent.y_start, extent.y_end, mesh.dimension['y'])
    z_ = np.linspace(extent.z_start, extent.z_end, mesh.dimension['z'])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    # Initialize data grid
    grid = pv.PolyData(points)
    grid.point_arrays['scalars'] = data[0].flatten()
    grid.set_active_scalars('scalars')

    # Show 3D data
    plotter = pvqt.BackgroundPlotter()
    plotter.add_mesh(grid, scalars='scalars', opacity=0.3, render_points_as_spheres=False, point_size=25)
    plotter.add_scalar_bar(title=smoke.quantity.name)
    plotter.view_isometric()

    # Animate data
    def animate():
        for d in data:
            grid.point_arrays['scalars'] = d.flatten()
            time.sleep(0.1)

    thread = Thread(target=animate)
    thread.start()


if __name__ == "__main__":
    main()
