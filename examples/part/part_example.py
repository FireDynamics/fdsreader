import pyvista as pv
import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_data")

    # Get all particles with specified id
    particles = sim.particles["WATER PARTICLES"]

    # Get all data for the specified quantity
    # quantity = "PARTICLE VELOCITY"
    quantity = "PARTICLE DIAMETER"
    color_data = particles.data[quantity]
    t_0 = next(i for i, pos in enumerate(particles.positions) if pos.shape[0] != 0)

    # Create PyVista animation
    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.open_movie("anim.mp4")

    actor = plotter.add_mesh(pv.PolyData(particles.positions[t_0]), scalars=color_data[t_0],
                             render_points_as_spheres=True, point_size=15)
    plotter.add_scalar_bar(title=quantity)

    # Open a gif
    plotter.open_gif("particles.gif")

    for t in range(t_0 + 1, len(color_data)):
        plotter.remove_actor(actor)
        actor = plotter.add_mesh(pv.PolyData(particles.positions[t]), scalars=color_data[t],
                                 render_points_as_spheres=True, point_size=15)
        plotter.write_frame()

    # Closes and finalizes movie
    plotter.close()


if __name__ == "__main__":
    main()
