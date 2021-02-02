import os
from typing import List, TextIO, Dict, AnyStr, Sequence, Tuple, Union

import numpy as np
import logging
import pickle

from fdsreader.bndf import Obstruction, Patch
from fdsreader.isof import Isosurface
from fdsreader.isof.IsosurfaceCollection import IsosurfaceCollection
from fdsreader.part import Particle
from fdsreader.part.ParticleCollection import ParticleCollection
from fdsreader.plot3d import Plot3D
from fdsreader.plot3d.Plot3dCollection import Plot3DCollection
from fdsreader.slcf import Slice
from fdsreader.slcf.SliceCollection import SliceCollection
from fdsreader.utils import Mesh, Dimension, Surface, Quantity, Ventilation, Extent
from fdsreader.utils.data import create_hash, get_smv_file, Device
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings


class Simulation:
    """Master class managing all data for a given simulation.

    :ivar smv_file_path: Path to the .smv file of the simulation.
    :ivar root_path: Path to the root directory of the simulation.
    :ivar fds_version: Version of FDS the simulation was performed with.
    :ivar chid: Name (ID) of the simulation.
    :ivar hrrpuv_cutoff: The hrrpuv_cutoff value.
    :ivar default_texture_origin: The default origin used for textures with no explicit origin.
    :ivar out_file_path: Path to the .out file of the simulation.
    :ivar surfaces: List containing all surfaces defined in this simulation.
    :ivar meshes: List containing all meshes (grids) defined in this simulation.
    """

    def __new__(cls, path: str):
        if settings.ENABLE_CACHING:
            smv_file_path = get_smv_file(path)
            root_path = os.path.dirname(path)

            with open(smv_file_path, 'r') as infile:
                for line in infile:
                    if line.strip() == "CHID":
                        chid = infile.readline().strip()
                        break

            pickle_file_path = Simulation._get_pickle_filename(root_path, chid)

            if path and os.path.isfile(path):
                with open(pickle_file_path) as f:
                    sim = pickle.load(f)
                if not isinstance(sim, cls):
                    os.remove(pickle_file_path)
                else:
                    if sim._hash == create_hash(smv_file_path):
                        return sim

        return super(Simulation, cls).__new__(cls)

    def __init__(self, path: str):
        """
        :param path: Either the path to the directory containing the simulation data or direct path
            to the .smv file for the simulation in case that multiple simulation output was written to
            the same directory.
        """
        # Check if the file has already been instantiated via a cached pickle file
        if not hasattr(self, "_hash"):
            self.smv_file_path = get_smv_file(path)

            self.root_path = os.path.dirname(self.smv_file_path)

            self.meshes: List[Mesh] = list()
            self.surfaces: List[Surface] = list()
            self.obstructions: List[Obstruction] = list()
            self.ventilations: Dict[Ventilation] = list()

            # First collect all meta-information for any FDS data to later combine the gathered
            # information into data collections
            self.slices = dict()
            self.data_3d = dict()
            self.isosurfaces = dict()
            self.particles = None
            self.devices: Dict[str, Union[Device, List[Device]]] = dict()
            device_tmp = str()

            with open(self.smv_file_path, 'r') as smv_file:
                for line in smv_file:
                    keyword = line.strip()
                    if keyword == "VERSION":
                        self.fds_version = smv_file.readline().strip()
                    elif keyword == "CHID":
                        self.chid = smv_file.readline().strip()
                    elif keyword == "CSVF":
                        csv_type = smv_file.readline().strip()
                        filename = smv_file.readline().strip()
                        file_path = os.path.join(self.root_path, filename)
                        if csv_type == "hrr":
                            self.hrr = self._load_HRR_data(file_path)
                        elif csv_type == "steps":
                            self.steps = self._load_step_data(file_path)
                        elif csv_type == "devc":
                            device_tmp = file_path
                            self.devices["Time"] = Device(Quantity("Time", "Time", ""),
                                                          (.0, .0, .0), (.0, .0, .0))
                    elif keyword == "HRRPUVCUT":
                        self.hrrpuv_cutoff = float(smv_file.readline().strip())
                    elif keyword == "TOFFSET":
                        offsets = smv_file.readline().strip().split()
                        self.default_texture_origin = tuple(float(offsets[i]) for i in range(3))
                    elif keyword == "CLASS_OF_PARTICLES":
                        times, particles = self._load_particle_meta(smv_file)
                        self.particles = ParticleCollection(times, particles)
                    elif "GRID" in keyword:
                        self.meshes.append(self._load_mesh(smv_file, keyword))
                    elif keyword == "SURFACE":
                        self.surfaces.append(self._load_surface(smv_file))
                    elif keyword == "DEVICE":
                        name, device = self._register_device(smv_file)
                        if name in self.devices:
                            if type(self.devices) == list:
                                self.devices[name].append(device)
                            else:
                                self.devices[name] = [self.devices[name], device]
                        else:
                            self.devices[name] = device
                    elif "SLC" in keyword:
                        self._load_slice(smv_file, keyword)
                    elif "ISOG" in keyword:
                        self._load_isosurface(smv_file, keyword)
                    elif keyword == "PL3D":
                        self._load_data_3d(smv_file, keyword)
                    elif "BND" in keyword:
                        self._load_boundary_data(smv_file, keyword)
                    elif "PRT5" in keyword:
                        self._load_particles(smv_file, keyword)

                self.cpu = self._load_CPU_data()
                if device_tmp != "":
                    self._load_DEVC_data(device_tmp)

            # POST INIT (post read)
            self.out_file_path = os.path.join(self.root_path, self.chid + ".out")
            for obst in self.obstructions:
                obst._post_init()
            # Combine the gathered temporary information into data collections
            self.slices = SliceCollection(
                Slice(self.root_path, slice_data[0]["cell_centered"], slice_data[0]["times"],
                      slice_data[1:]) for slice_data
                in self.slices.values())
            self.data_3d = Plot3DCollection(self.data_3d.keys(), self.data_3d.values())
            self.isosurfaces = IsosurfaceCollection(self.isosurfaces.values())
            if self.particles is None:
                self.particles = ParticleCollection(())

            if settings.ENABLE_CACHING:
                # Hash will be saved to simulation pickle file and compared to new hash when loading
                # the pickled simulation again in the next run of the prpogram.
                self._hash = create_hash(self.smv_file_path)
                pickle.dump(self,
                            open(Simulation._get_pickle_filename(self.root_path, self.chid), 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def _get_pickle_filename(cls, root_path: str, chid: str) -> AnyStr:
        """Get the filename used to save the pickled simulation.
        """
        return os.path.join(root_path, chid + ".pickle")

    def _load_mesh(self, smv_file: TextIO, line: str) -> Mesh:
        """Load information for a single mesh from the smv file at current pointer position.
        """
        mesh_id = line.split()[1]
        logging.debug("Found MESH with id: %s", mesh_id)

        grid_numbers = smv_file.readline().strip().split()
        grid_dimensions = {'x': int(grid_numbers[0]) + 1, 'y': int(grid_numbers[1]) + 1,
                           'z': int(grid_numbers[2]) + 1}
        logging.debug("Number of cells: %i x %i x %i", *grid_dimensions.values())

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "PDIM"
        coordinates = dict()
        extents = smv_file.readline().split()
        extents = {'x': (float(extents[0]), float(extents[1])),
                   'y': (float(extents[2]), float(extents[3])),
                   'z': (float(extents[4]), float(extents[5]))}

        for dim in ('x', 'y', 'z'):
            smv_file.readline()  # Blank line
            assert smv_file.readline().strip()[:3] == "TRN"
            noc = int(smv_file.readline().strip())
            for _ in range(noc):
                smv_file.readline()
            coordinates[dim] = np.empty(grid_dimensions[dim], dtype=np.float32)
            for i in range(grid_dimensions[dim]):
                coordinates[dim][i] = float(smv_file.readline().split()[1])

        mesh = Mesh(coordinates, extents, mesh_id)

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "OBST"
        self._load_obstructions(smv_file, mesh)

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "VENT"
        self._load_vents(smv_file, mesh)

        # TODO: Offset und closed/open vents

        return mesh

    def _load_obstructions(self, smv_file: TextIO, mesh: Mesh):
        temp_data = list()
        n = int(smv_file.readline().strip())
        for _ in range(n):
            line = smv_file.readline().strip().split()
            ext = [float(line[i]) for i in range(6)]
            obst_index = int(line[6]) - 1

            side_surfaces = tuple(self.surfaces[int(line[i]) - 1] for i in range(7, 13))
            if len(line) > 13:
                texture_origin = (float(line[13]), float(line[14]), float(line[15]))
                temp_data.append((obst_index, Extent(*ext), side_surfaces, texture_origin))
            else:
                temp_data.append((obst_index, Extent(*ext), side_surfaces))

        for tmp in temp_data:
            line = smv_file.readline().strip().split()
            bound_indices = (int(float(line[0])), int(float(line[1])), int(float(line[2])),
                             int(float(line[3])), int(float(line[4])), int(float(line[5])))
            color_index = int(line[6])
            block_type = int(line[7])
            if color_index == -3:
                rgba = tuple(float(line[i]) for i in range(8, 12))
            else:
                rgba = ()

            if len(tmp) == 4:
                obst_index, extent, side_surfaces, texture_origin = tmp
            else:
                obst_index, extent, side_surfaces = tmp
                texture_origin = self.default_texture_origin

            if obst_index not in self.obstructions:
                self.obstructions[obst_index] = Obstruction(obst_index, side_surfaces,
                                                            bound_indices, color_index, block_type,
                                                            texture_origin, rgba=rgba)
            self.obstructions[obst_index]._extents[mesh] = extent
            mesh.obstructions.append(self.obstructions[obst_index])

    def _load_vents(self, smv_file: TextIO, mesh: Mesh):
        line = smv_file.readline().split()
        n, n_dummies = int(line[0]), int(line[1])

        temp_data = list()

        def read_common_info():
            line = smv_file.readline().strip().split()
            return line, [float(line[i]) for i in range(6)], int(line[6]), self.surfaces[
                int(line[7])]

        def read_common_info2():
            line = smv_file.readline().strip().split()
            bound_indices = tuple(int(line[i]) for i in range(6))
            color_index = int(line[6])
            draw_type = int(line[7])
            if len(line) > 8:
                rgba = tuple(float(line[i]) for i in range(8, 12))
            else:
                rgba = ()
            return bound_indices, color_index, draw_type, rgba

        texture_origin = ()
        for _ in range(n - n_dummies):
            line, ext, vent_id, surface = read_common_info()
            texture_origin = (float(line[8]), float(line[9]), float(line[10]))
            temp_data.append((Extent(*ext), vent_id, surface, texture_origin))

        for _ in range(n_dummies):
            _, extents, vent_id, surface = read_common_info()
            temp_data.append((Extent(*ext), vent_id, surface))

        for v in range(n):
            if v < n - n_dummies:
                extent, vent_id, surface, texture_origin = temp_data[v]
            else:
                extent, vent_id, surface = temp_data[v]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            if vent_id not in self.ventilations:
                self.ventilations[vent_id] = Ventilation(vent_id, surface, bound_indices,
                                                         color_index, draw_type, rgba=rgba,
                                                         texture_origin=texture_origin)
            self.ventilations[vent_id]._add_subventilation(mesh, extent)

        smv_file.readline()
        assert "CVENT" in smv_file.readline()

        n = int(smv_file.readline().strip())
        temp_data.clear()
        for _ in range(n):
            line, extent, vid, surface = read_common_info()
            circular_vent_origin = (float(line[12]), float(line[13]), float(line[14]))
            radius = float(line[15])
            temp_data.append((extent, vid, surface, texture_origin, circular_vent_origin, radius))

        for v in range(n):
            extent, vent_id, surface, texture_origin, circular_vent_origin, radius = temp_data[v]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            if vent_id not in self.obstructions:
                self.ventilations[vent_id] = Ventilation(vent_id, surface, bound_indices,
                                                         color_index, draw_type, rgba=rgba,
                                                         texture_origin=texture_origin,
                                                         circular_vent_origin=circular_vent_origin,
                                                         radius=radius)
            self.ventilations[vent_id]._add_subventilation(mesh, extent)

    def _load_surface(self, smv_file: TextIO) -> Surface:
        """Load the information for a single surface from the smv file at current pointer position.
        """

        surface_id = smv_file.readline().strip()

        line = smv_file.readline().strip().split()
        tmpm, material_emissivity = float(line[0]), float(line[1])

        line = smv_file.readline().strip().split()

        surface_type = int(line[0])
        texture_width, texture_height = float(line[1]), float(line[2])
        rgb = (float(line[3]), float(line[4]), float(line[5]))
        transparency = float(line[6])

        texture_map = smv_file.readline().strip()
        texture_map = None if texture_map == "null" else os.path.join(self.root_path, texture_map)

        return Surface(surface_id, tmpm, material_emissivity, surface_type, texture_width,
                       texture_height, texture_map, rgb, transparency)

    def _load_slice(self, smv_file: TextIO, line: str):
        """Loads the slice at current pointer position.
        """
        if "SLCC" in line:
            cell_centered = True
        else:
            cell_centered = False

        slice_id = int(line.split('!')[1].strip().split()[0])

        mesh_id = int(line.split('&')[0].strip().split()[1]) - 1
        mesh = self.meshes[mesh_id]

        # Read in index ranges for x, y and z
        bound_indices = [int(i.strip()) for i in line.split('&')[1].split('!')[0].strip().split()]
        extent, dimension = self._indices_to_extent(bound_indices, mesh)

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        file_path = os.path.join(self.root_path, filename)

        times = list()
        with open(file_path + ".bnd", 'r') as bnd_file:
            for line in bnd_file:
                times.append(float(line.split()[0]))
        times = np.array(times)

        if slice_id not in self.slices:
            self.slices[slice_id] = [{"cell_centered": cell_centered, "times": times}]
        self.slices[slice_id].append(
            {"dimension": dimension, "extent": extent, "mesh": mesh, "filename": filename,
             "quantity": quantity, "label": label, "unit": unit})

        logging.debug("Found SLICE with id: :i", slice_id)

    def _load_boundary_data(self, smv_file: TextIO, line: str):
        """Loads the boundary data at current pointer position.
        """
        line = line.split()
        if line[0] == 'BNDC':
            cell_centered = True
        else:
            cell_centered = False
        mesh_id = int(line[1]) - 1
        mesh = self.meshes[mesh_id]

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        bid = int(filename.split('_')[-1][:-3])

        file_path = os.path.join(self.root_path, filename)

        patches = dict()

        times = list()
        lower_bounds = list()
        upper_bounds = list()
        with open(file_path + ".bnd", 'r') as bnd_file:
            for line in bnd_file:
                splits = line.split()
                times.append(float(splits[0]))
                lower_bounds.append(float(splits[1]))
                upper_bounds.append(float(splits[2]))
        times = np.array(times)
        lower_bounds = np.array(times, dtype=np.float32)
        upper_bounds = np.array(times, dtype=np.float32)
        n_t = times.shape[0]

        with open(file_path, 'rb') as infile:
            # Offset of the binary file to the end of the file header.
            offset = 3 * fdtype.new((('c', 30),)).itemsize
            infile.seek(offset)

            n_patches = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            dtype_patches = fdtype.new((('i', 9),))
            patch_infos = fdtype.read(infile, dtype_patches, n_patches)
            offset += fdtype.INT.itemsize + dtype_patches.itemsize * n_patches
            patch_offset = fdtype.FLOAT.itemsize
            cell_centered = False
            for patch_info in patch_infos:
                patch_info = patch_info[0]
                extent, dimension = self._indices_to_extent(patch_info[:6], mesh)
                orientation = patch_info[6]
                obst_index = patch_info[7]
                p = Patch(file_path, dimension, extent, orientation, cell_centered,
                          patch_offset + offset, n_t)

                # Skip obstacles with ID 0, which just gives the extent of the (whole) mesh faces
                # These might be needed in case of "closed" mesh faces
                if obst_index != 0:
                    obst_id = mesh.obstructions[obst_index - 1].id
                    if obst_id not in patches:
                        patches[obst_id] = list()
                    patches[obst_id].append(p)
                patch_offset += fdtype.new((('f', str(p.shape)),)).itemsize

        for obst_id, p in patches.items():
            for patch in p:
                patch._post_init(patch_offset)
            self.obstructions[obst_id]._add_patches(bid, cell_centered, quantity, label, unit, mesh,
                                                    p, times, n_t, lower_bounds, upper_bounds)

    def _load_data_3d(self, smv_file: TextIO, line: str):
        """Loads the plot3d at current pointer position.
        """
        # Todo: Also read SMOKG3D data?
        line = line.strip().split()

        time = float(line[1])

        mesh_index = int(line[2]) - 1

        filename = smv_file.readline().strip()
        quantities = list()
        for _ in range(5):
            quantity = smv_file.readline().strip()
            label = smv_file.readline().strip()
            unit = smv_file.readline().strip()
            quantities.append(Quantity(quantity, label, unit))

        if time not in self.data_3d:
            self.data_3d[time] = Plot3D(self.root_path, time, quantities)
        self.data_3d[time]._add_subplot(filename, self.meshes[mesh_index])

    def _load_isosurface(self, smv_file: TextIO, line: str):
        """Loads the isosurface at current pointer position.
        """
        double_quantity = line[0] == 'T'
        mesh_index = int(line.strip().split()[1]) - 1

        iso_filename = smv_file.readline().strip()
        iso_id = int(iso_filename.split('_')[-1][:-4])

        if double_quantity:
            viso_filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        label = smv_file.readline().strip()
        unit = smv_file.readline().strip()
        if double_quantity:
            v_quantity = smv_file.readline().strip()
            v_label = smv_file.readline().strip()
            v_unit = smv_file.readline().strip()

        if double_quantity:
            if iso_id not in self.isosurfaces:
                self.isosurfaces[iso_id] = Isosurface(iso_id,
                                                      os.path.dirname(self.smv_file_path),
                                                      double_quantity, quantity, label, unit,
                                                      v_quantity=v_quantity, v_label=v_label,
                                                      v_unit=v_unit)
            self.isosurfaces[iso_id]._add_subsurface(self.meshes[mesh_index], iso_filename,
                                                     viso_filename=viso_filename)
        else:
            if iso_id not in self.isosurfaces:
                self.isosurfaces[iso_id] = Isosurface(iso_id,
                                                      os.path.dirname(self.smv_file_path),
                                                      double_quantity, quantity, label, unit)
            self.isosurfaces[iso_id]._add_subsurface(self.meshes[mesh_index], iso_filename)

    def _load_particle_meta(self, smv_file: TextIO) -> List[Particle]:
        particle_class = smv_file.readline().strip()
        particles = list()
        while particle_class != "":
            # Todo: What do these mean?
            some_vals = smv_file.readline().strip().split()

            n_q = int(smv_file.readline().strip())
            quantities = list()
            for _ in range(n_q):
                quantity = smv_file.readline().strip()
                label = smv_file.readline().strip()
                unit = smv_file.readline().strip()
                quantities.append(Quantity(quantity, label, unit))
            particles.append(Particle(particle_class, quantities))
            particle_class = smv_file.readline().strip()
            print(particle_class)

        # Read times of an arbitrary .prt5.bnd file
        filename = next(file for file in os.listdir(self.root_path) if file.endswith(".prt5.bnd"))
        file_path = os.path.join(self.root_path, filename)

        with open(file_path, 'r') as bnd_file:
            line = bnd_file.readline().strip().split()
            n_classes = int(line[1])
            times = list()
            n_q = list()
            for _ in range(n_classes):
                line = bnd_file.readline().strip().split()
                n_q.append(int(line[0]))
                for _ in range(n_q[-1]):
                    bnd_file.readline()
            bnd_file.seek(0)

            for line in bnd_file:
                times.append(float(line.strip().split()[0]))
                for i in range(n_classes):
                    bnd_file.readline()
                    for q in range(n_q[i]):
                        line = bnd_file.readline().strip().split()
                        particle = particles[i]
                        quantity = particle.quantities[q].quantity
                        particle.lower_bounds[quantity].append(float(line[0]))
                        particle.upper_bounds[quantity].append(float(line[1]))

        return particles

    def _load_particles(self, smv_file: TextIO, line: str):
        some_value = int(line.split()[1].strip())

        self.n_t, self.times, self.n_particles, self.positions, self.tags, self.quantities = _read_multiple_prt5_files(
            self.classes)

    def _register_device(self, smv_file: TextIO) -> Tuple[str, Device]:
        line = smv_file.readline().strip().split('%')
        name = line[0].strip()
        quantity_label = line[1].strip()
        line = smv_file.readline().strip().split('#')[0].split()
        position = (float(line[0]), float(line[1]), float(line[2]))
        orientation = (float(line[3]), float(line[4]), float(line[5]))
        return name, Device(Quantity(name, quantity_label, ""), position, orientation)

    def _load_DEVC_data(self, file_path: str):
        with open(file_path, 'r') as infile:
            units = infile.readline().split(',')
            names = [name.replace('"', '').replace('\n', '').strip() for name in
                     infile.readline().split(',')]
            values = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
            for k in range(len(names)):
                devc = self.devices[names[k]]
                devc.quantity.unit = units[k]
                size = values.shape[0]
                devc.data = np.empty((size,), dtype=np.float32)
                for i in range(size):
                    devc.data[i] = values[i][k]

        line_path = file_path.replace("devc", "steps")
        if os.path.exists(line_path):
            with open(line_path, 'r') as infile:
                infile.readline()
                names = infile.readline().split(',')
                data = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
                for k, key in enumerate(names):
                    if key in self.devices:
                        devc = self.devices[key]
                        for i in range(len(devc)):
                            devc[i] = data[k, i]

    def _load_HRR_data(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path, 'r') as infile:
            infile.readline()
            keys = infile.readline().split(',')
            values = np.genfromtxt(infile, delimiter=',', dtype=np.float32, autostrip=True)
            return self._transform_csv_data(keys, values, [np.float32] * len(keys))

    def _load_step_data(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path, 'r') as infile:
            infile.readline()
            keys = infile.readline().split(',')
            dtypes = [np.float32] * len(keys)
            dtypes[0] = int
            dtypes[1] = np.dtype("datetime64[ms]")
            values = np.genfromtxt(infile, delimiter=',', dtype=dtypes, autostrip=True)
            return self._transform_csv_data(keys, values, dtypes)

    def _load_CPU_data(self) -> Dict[str, np.ndarray]:
        file_path = os.path.join(self.root_path, self.chid + "_cpu.csv")
        if os.path.exists(file_path):
            with open(file_path, 'r') as infile:
                keys = infile.readline().split(",")
                dtypes = [np.float32] * len(keys)
                dtypes[0] = int
                values = np.genfromtxt(infile, delimiter=',', dtype=dtypes, autostrip=True)
                if len(values.shape) != 0:
                    return self._transform_csv_data(keys, values, dtypes)
                else:
                    return self._transform_csv_data(keys, values.reshape((1,)), dtypes)

    def _transform_csv_data(self, keys, values, dtypes):
        size = values.shape[0]
        data = {keys[i]: np.empty((size,), dtype=dtypes[i]) for i in range(len(keys))}
        for k, arr in enumerate(data.values()):
            for i in range(size):
                arr[i] = values[i][k]
        return data

    def _indices_to_extent(self, indices: Sequence[Union[int, str]], mesh: Mesh) -> Tuple[
        Extent, Dimension]:
        co = mesh.coordinates
        x_min, x_max, y_min, y_max, z_min, z_max = (
            int(indices[0]), int(indices[1]), int(indices[2]), int(indices[3]), int(indices[4]),
            int(indices[5]))
        co_x_min, co_x_max, co_y_min, co_y_max, co_z_min, co_z_max = (
            co['x'][x_min], co['x'][x_max], co['y'][y_min],
            co['y'][y_max], co['z'][z_min], co['z'][z_max])
        dimension = Dimension(x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)

        extent = Extent(co_x_min, co_x_max, co_y_min, co_y_max, co_z_min, co_z_max)
        return extent, dimension

    def clear_cache(self, clear_persistent_cache=False):
        """Remove all data from the internal cache that has been loaded so far to free memory.

        :param clear_persistent_cache: Whether to clear the persistent simulation cache as well.
        """
        self.slices.clear_cache()
        self.data_3d.clear_cache()
        self.isosurfaces.clear_cache()
        self.particles.clear_cache()
        for obst in self.obstructions:
            obst.clear_cache()

        if clear_persistent_cache:
            os.remove(Simulation._get_pickle_filename(self.root_path, self.chid))
