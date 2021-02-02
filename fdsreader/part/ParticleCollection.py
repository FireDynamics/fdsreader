from typing import Iterable, List

import numpy as np

from fdsreader.part import Particle
from fdsreader.utils.data import FDSDataCollection
import fdsreader.utils.fortran_data as fdtype
from fdsreader import settings


class ParticleCollection(FDSDataCollection):
    """Collection of :class:`Plot3D` objects. Offers extensive functionality for filtering and
        using plot3Ds as well as its subclasses such as :class:`SubPlot3D`.
    """

    def __init__(self, times: Iterable[float], *particles: Iterable[Particle]):
        super().__init__(*particles)
        self.times = list(times)
        self._filenames = list()

        if not settings.LAZY_LOAD:
            for particle in self:
                particle._init_callback = self._load_data

    def _post_init(self):
        if not settings.LAZY_LOAD:
            self._load_data()

    def _load_data(self):
        pass

    def _extract_prt5_file(self, file_path: str):
        """
        Function to read in all particle data written by FDS from one single
        mesh into a single file.
                - n_q : list[i1] -> int
                        Number of quantities for particle class with index i1
                - quantity_labels : list[i1][i2] -> str
                        Label for quantity with index i2 of particle class with index i1
                - quantity_units : list[i1][i2] -> str
                        Units for quantity with index i2 of particle class with index i1
        n_outputsteps : int
            Number of timesteps that have been read with fdspsp, i.e.,
            n_outputsteps = n_timesteps // output_each + 1
        times : list[i1] -> float
            Time at output step i1
        n_particles : dict[s1] -> list[i2] -> int
            Number of particles in class with label s1 at output step i2
        positions : dict[s1] -> list[i2][i3] -> numpy.array(float)
            positions holds coordinate lists (as numpy-arrays) for particle
            class with label s1 at output step i2 for component (x,y,z) i3,
            where the length of the array is given by the number of particles of
            the selected class and step, i.e., n_particles[s1][i2]
        tags : dict[s1] -> list[i2] -> numpy.array(int)
            numpy-array of particle tags for class with label s1 at
            output step i2
        quantities : dict[s1][s2] -> list[i3] -> numpy.array(float)
            numpy-array of particle quantity with label s2 for class with
            label s1 at output step i3
        """
        classes = [p.class_name for p in particles]

        with open(file_path, 'rb') as infile:
            infile.seek(3 * fdtype.INT.itemsize + fdtype.CHAR*60*len(classes))

            # Size of zero particles for each particle class in a single timestep
            zero_particles_size = (6 if fdtype.HAS_POST_BORDER else 3) * fdtype.PRE_BORDER.itemsize

            # Create empty lists for each particle class
            data = {class_name: {q.label: [] for q in self[i].quantities} for i, class_name in enumerate(classes)}

            for _ in self.times:
                # Skip time value
                infile.seek(fdtype.FLOAT.itemsize, whence=1)

                # Read data for each particle class
                for particle in self:
                    # Read number of particles in each class
                    n_particles = fdtype.read(infile, fdtype.INT, 1)[0]

                    # If no particles were found, skip this class
                    # if n_particles == 0:
                    #     infile.seek(zero_particles_size, whence=1)
                    #
                    #     # Store empty data for completeness
                    #     positions[class_name].append([np.array([]), np.array([]), np.array([])])
                    #     tags[class_name].append(np.array([]))
                    #     for arr in data[class_name].values():
                    #         arr.append(np.array([]))
                    #     continue

                    # Read positions
                    dtype_positions = fdtype.new((('f', 3 * n_particles),))
                    particle._positions = fdtype.read(infile, dtype_positions, 1)[0][0].reshape((n_particles, 3), order='F')
                    print("Positions:", particle._positions)

                    # Read tags
                    dtype_tags = fdtype.new((('i', n_particles),))
                    particle._tags = fdtype.read(infile, dtype_tags, 1)[0]
                    print("Tags:", particle._tags)

                    # Read actual quantity values
                    dtype_data = fdtype.new((('f', n_particles * len(particle.quantities)),))
                    particle._data = fdtype.read(infile, dtype_data, 1)[0][0].reshape((n_particles, len(particle.quantities)), order='F')
                    print("Data:", particle._data)

    def _read_multiple_prt5_files(self):
        """
        Function to parse and read in all particle data written by FDS from
        multiple meshes into multiple files.
        Parameters
                - n_q : list[i1] -> int
                        Number of quantities for particle class with index i1
                - quantity_labels : list[i1][i2] -> str
                        Label for quantity with index i2 of particle class with
                        index i1
                - quantity_units : list[i1][i2] -> str
                        Units for quantity with index i2 of particle class with
                        index i1
        n_outputsteps : int
            Number of timesteps that have been read with fdspsp, i.e.,
            n_outputsteps = n_timesteps // output_each + 1
        times : list[i1] -> float
            Time at output step i1
        n_particles : dict[s1] -> list[i2] -> int
            Number of particles in class with label s1 at output step i2
        positions : dict[s1] -> list[i2][i3] -> numpy.array(float)
            positions holds coordinate lists (as numpy-arrays) for particle
            class with label s1 at output step i2 for component (x,y,z) i3,
            where the length of the array is given by the number of particles of
            the selected class and step, i.e., n_particles[s1][i2]
        tags : dict[s1] -> list[i2] -> numpy.array(int)
            numpy-array of particle tags for class with label s1 at
            output step i2
        quantities : dict[s1][s2] -> list[i3] -> numpy.array(float)
            numpy-array of particle quantity with label s2 for class with
            label s1 at output step i3
        """
        # Read remaining input files
        if parallel:
            pool = Pool(None)
            worker = partial(_read_prt5_file, classes_dict=classes_dict,
                             output_each=output_each,
                             n_timesteps=n_timesteps, logs=logs)
            results[1:] = pool.map(worker, filelist)
            pool.close()
            pool.join()
        else:
            for filename in filelist:
                results.append(
                    _read_prt5_file(filename, classes_dict, output_each, n_timesteps, logs))

        #
        # CONCATENATE: results of all files
        #

        # Calculate global number of particles for every particle class in
        # each timestep
        n_particles = {c_label: [0] *
                                n_outputsteps for c_label in classes_dict.values()}
        for res in results:
            local_n_particles = res[3]
            for c_label in classes_dict.values():
                for (o_step, local_n_particle) in enumerate(local_n_particles[c_label]):
                    n_particles[c_label][o_step] += local_n_particle

        # Prepare empty data containers one after another to avoid memory
        # fragmentation
        positions = {c_label: [[np.empty(n_particles[c_label][o_step]),
                                np.empty(n_particles[c_label][o_step]),
                                np.empty(n_particles[c_label][o_step])]
                               for o_step in range(n_outputsteps)]
                     for c_label in classes_dict.values()}
        tags = {c_label: [np.empty(n_particles[c_label][o_step])
                          for o_step in range(n_outputsteps)]
                for c_label in classes_dict.values()}
        quantities = {c_label: {q_label: [np.empty(n_particles[c_label][o_step])
                                          for o_step in range(n_outputsteps)]
                                for q_label in quantity_labels[c_index]}
                      for (c_index, c_label) in classes_dict.items()}

        # Continuous index offsets to build consecutive buffer
        offsets = {c_label: np.zeros(n_outputsteps, dtype="int")
                   for c_label in classes_dict.values()}

        # Attach data
        for res in results:
            (_, _, _, local_n_particles, local_positions, local_tags, local_quantities) = res

            for c_label in classes_dict.values():
                for (o_step, n) in enumerate(local_n_particles[c_label]):
                    o = offsets[c_label][o_step]
                    positions[c_label][o_step][0][o:o + n] = \
                        np.copy(local_positions[c_label][o_step][0])
                    positions[c_label][o_step][1][o:o + n] = \
                        np.copy(local_positions[c_label][o_step][1])
                    positions[c_label][o_step][2][o:o + n] = \
                        np.copy(local_positions[c_label][o_step][2])
                    tags[c_label][o_step][o:o + n] = \
                        np.copy(local_tags[c_label][o_step])
                    for q_label in local_quantities[c_label].keys():
                        quantities[c_label][q_label][o_step][o:o + n] = \
                            np.copy(local_quantities[c_label][q_label][o_step])

                    offsets[c_label][o_step] += n

        for c_label in classes_dict.values():
            for (o_step, n_particle) in enumerate(n_particles[c_label]):
                assert offsets[c_label][o_step] == n_particle

        return n_outputsteps, times, n_particles, positions, tags, quantities
