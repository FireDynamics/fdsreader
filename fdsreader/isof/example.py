from fdsreader.utils import scan_directory_smv
from fdsreader import Simulation


def main():
    smv_file_paths = scan_directory_smv("C:\\Users\\janv1\\Desktop\\fds\\sample_data\\flash1e_fds6")

    sim = Simulation(smv_file_paths[0])

    isosurfaces = sim.isosurfaces

    isosurfaces[0].vertices


if __name__ == "__main__":
    main()
