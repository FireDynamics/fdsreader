import numpy as np
import pyvista as pv

import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_steckler/")

    # Get 3D smoke data for a specific quantity. Smoke3D values are Smokeview display
    # codes (0-254), not physical values: fine for rendering, not for analysis.
    quantity = "HRRPUV"  # or "SOOT DENSITY" (zero here, the case sets no SOOT_YIELD)
    smoke = sim.smoke_3d.get_by_quantity(quantity)

    data, coordinates = smoke.to_global(masked=True, return_coordinates=True, fill=np.nan)

    # Create 3D grid
    x_ = np.linspace(coordinates["x"][0], coordinates["x"][-1], len(coordinates["x"]))
    y_ = np.linspace(coordinates["y"][0], coordinates["y"][-1], len(coordinates["y"]))
    z_ = np.linspace(coordinates["z"][0], coordinates["z"][-1], len(coordinates["z"]))
    x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

    # Initialize data grid
    grid = pv.PolyData(points)
    grid["values"] = data[0].flatten()

    # Plot 3D data
    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.add_mesh(
        grid,
        scalars="values",
        clim=(np.nanmin(data), np.nanmax(data)),
        opacity="linear",
        render_points_as_spheres=False,
        point_size=25,
        scalar_bar_args={"title": quantity},
    )

    # Open a gif
    plotter.open_gif("smoke.gif")

    for d in data:
        grid["values"] = d.flatten()
        plotter.write_frame()

    # Closes and finalizes movie
    plotter.close()


if __name__ == "__main__":
    main()
