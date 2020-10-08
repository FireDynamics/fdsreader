from utils import Simulation, scan_directory_smv


smv_file_paths = scan_directory_smv("../../examples/plot_slice/fds_data")

sim = Simulation(smv_file_paths[0])

slice = sim.slices[0]
print(slice._subslices[0].get_data(slice.quantities[0].quantity, slice.root_path, slice.cell_centered))