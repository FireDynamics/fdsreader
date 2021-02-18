import pyvista as pv
import fdsreader as fds


def main():
    sim = fds.Simulation("./fds_data")

    # Get all particles with specified id
    particles = sim.particles["water droplets"]

    # Get all data for the specified quantity
    # quantity = "PARTICLE TEMPERATURE"
    quantity = "PARTICLE DIAMETER"
    color_data = particles.data[quantity]

    t_0 = next(i for i, pos in enumerate(particles.positions) if pos.shape[0] != 0)

    # Create PyVista animation
    plotter = pv.Plotter()
    plotter.open_movie("anim.mp4")

    actor = plotter.add_mesh(pv.PolyData(particles.positions[t_0]), scalars=color_data[t_0],
                             render_points_as_spheres=True, point_size=15)
    plotter.add_scalar_bar(title=quantity)
    plotter.show(auto_close=False)
    plotter.write_frame()

    for t in range(t_0 + 1, len(color_data)):
        if particles.positions[t].shape[0] == 0:
            continue
        plotter.remove_actor(actor)
        actor = plotter.add_mesh(pv.PolyData(particles.positions[t]), scalars=color_data[t],
                                 render_points_as_spheres=True, point_size=15)
        plotter.remove_scalar_bar()
        plotter.add_scalar_bar(title=quantity)
        plotter.write_frame()

    plotter.close()


if __name__ == "__main__":
    main()
