from fdsreader.utils import scan_directory_smv
from fdsreader import Simulation


def main():
    # smv_file_paths = scan_directory_smv("../../examples/isof/fds_data")
    smv_file_paths = scan_directory_smv("C:\\Users\\janv1\\Desktop\\fds\\sample_data\\fds_demo")

    sim = Simulation(smv_file_paths[0])

    mesh = sim.meshes[0]

    isosurfaces = sim.isosurfaces

    print(isosurfaces[0].get_subsurface(mesh).vertices)


if __name__ == "__main__":
    main()
