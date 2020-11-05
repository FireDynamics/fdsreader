from utils import Simulation, scan_directory_smv


def main():

    smv_file_paths = scan_directory_smv("../../examples/isof/fds_data")

    sim = Simulation(smv_file_paths[0])

    isosurfaces = sim.isosurfaces

    print(isosurfaces)


if __name__ == "__main__":
    main()
