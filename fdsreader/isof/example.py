from fdsreader import Simulation


def main():
    sim = Simulation("C:\\Users\\janv1\\Desktop\\fds\\sample_data\\fds_demo")

    mesh = sim.meshes[1]

    isosurfaces = sim.isosurfaces

    print(isosurfaces[0].get_subsurface(mesh).vertices[20].shape)


if __name__ == "__main__":
    main()
