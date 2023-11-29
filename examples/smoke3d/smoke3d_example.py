import pyvista as pv
import numpy as np

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_steckler/")

    # Get 3D smoke data for a specific quantity in one of the meshes
    quantity = "Temperature"  # "SOOT MASS FRACTION"
    smoke = sim.smoke_3d.get_by_quantity(quantity)

    data, coordinates = smoke.to_global(masked=True, return_coordinates=True, fill=np.nan)

    # Create 3D grid
    x_ = np.linspace(coordinates['x'][0], coordinates['x'][-1], len(coordinates['x']))
    y_ = np.linspace(coordinates['y'][0], coordinates['y'][-1], len(coordinates['y']))
    z_ = np.linspace(coordinates['z'][0], coordinates['z'][-1], len(coordinates['z']))
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    # Initialize data grid
    grid = pv.PolyData(points)

    # Plot 3D data
    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.add_mesh(grid, scalars=data[0].flatten(), opacity=0.3, render_points_as_spheres=False, point_size=25)
    plotter.add_scalar_bar(title=quantity)

    # Open a gif
    plotter.open_gif("smoke.gif")

    for d in data:
        plotter.update_scalars(d.flatten(), render=False)
        plotter.write_frame()

    # Closes and finalizes movie
    plotter.close()


if __name__ == "__main__":
    main()
