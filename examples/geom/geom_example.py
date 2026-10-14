import pyvista as pv

from fdsreader import Simulation


def main():
    sim = Simulation("./fds_data")

    geom = sim.geom_data.filter_by_quantity("Radiative Heat Flux")[0]
    # Plot the values of the last time step on the geometry
    pv.PolyData.from_regular_faces(geom.vertices, geom.faces).plot(
        scalars=geom.data[-1], scalar_bar_args={"title": "Radiative Heat Flux"}
    )


if __name__ == "__main__":
    main()
