from typing import List


class Particle:
    """Container to store particle data from particle simulations with FDS.

    :ivar t_n: Number of timesteps that were written out
    :ivar classes: Labels of particle classes that have been read with fdspsp
    :ivar times: Time at output step i1
    :ivar n_particles: Number of particles in class with label s1 at output step i2
    :ivar positions: Holds coordinate lists (as numpy-arrays) for particle
        class with label s1 at output step i2 for component (x,y,z) i3,
        where the length of the array is given by the number of particles of
        the selected class and step, i.e., n_particles[s1][i2]
    :ivar tags: Numpy-array of particle tags for class with label s1 at
        output step i2
    :ivar quantities: Numpy-array of particle quantity with label s2 for class with
        label s1 at output step i3
    """

    def __init__(self, classes: List[str], root_path: str):
        self.classes = classes

        # Read particle data from prt5 files
        self.info, self.n_outputsteps, self.times, self.n_particles, self.positions, self.tags, self.quantities = _read_multiple_prt5_files(self.classes)

    def _read_prt5_file(filename, classes_dict,
                        output_each=1, n_timesteps=None, logs=True):
      """
      Function to read in all particle data written by FDS from one single
      mesh into a single file.
      Parameters
      ----------
      filename : str
        Name of the prt5 file.
      classes_dict : dict -> (i1:s2)
        Specify all particle classes to be read. Translates particle class
        indices i1 occurring in the smv and prt5 file to the corresponding
        class label s2 from the smv file.
      output_each : int
        Each n-th timestep will actually be read, those inbetween will be
        skipped. By default, every timestep will be considered.
      n_timesteps : int
        Integer to indicate the number of timesteps of the specified FDS
        simulation. It will be used to identify whether there happen to be
        no more particles in subsequent timesteps during the reading
        process. By default, this feature is disabled.
      logs : bool
        Choose if logs will be printed out. This feature is enabled by
        default.
      Returns
      -------
      (info, n_outputsteps, times, n_particles, positions, tags, quantities)
      with:
      info : dict
        Contains meta information about the original FDS dataset, as:
          - filename : str
              The name of the read datafile
          - fds_version : str
              Particle data has been generated with this FDS version
          - n_classes : int
              Number of particle classes in the simulation
          - n_timesteps : int
              Number of total timesteps in the FDS simulation
          - n_quantities : list[i1] -> int
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

      # Verify user input
      assert output_each > 0
      assert all(c_index >= 0 for c_index in classes_dict.keys())
      assert n_timesteps is None or n_timesteps > 0

      #
      # INITIALIZATION: open file
      #

      if logs:
        time_start = time()
        print("Reading particle file '{}'.".format(filename))

      file_object = open(filename, 'rb')
      assert file_object, \
          "Could not open file '{}'! Aborting.".format(filename)

      # Map file content to memory
      file_mm = mmap(file_object.fileno(), 0, access=ACCESS_READ)
      file_mm.flush()

      file_mm_pos = 0

      def _read_from_buffer(dtype, skip=False):
        """
        Reads exactly one instance of dtype from the current memory-mapped
        file object and progresses the position in memory accordingly.
        Results will be returned as numpy.ndarray.
        """

        nonlocal file_mm, file_mm_pos

        # Proceed the input filestream by the corresponding amount of bytes
        if skip:
          file_mm_pos += dtype.itemsize
          return

        data_raw = np.frombuffer(file_mm, dtype, count=1, offset=file_mm_pos)
        file_mm_pos += dtype.itemsize

        # We are only interested in the actual data, ignore the remains of
        # the binary file
        return data_raw[0][1]

      #
      # READ: miscellaneous data
      #

      info = {"filename": filename}

      # Read endianess flag written by FDS which will be ignored
      _read_from_buffer(_get_fds_dtype("int"))

      # Read FDS version identifier
      info["fds_version"] = _read_from_buffer(_get_fds_dtype("int"))[0]

      # Read number of classes
      info["n_classes"] = _read_from_buffer(_get_fds_dtype("int"))[0]
      assert all(c_index < info["n_classes"] for c_index in classes_dict.keys())

      # Read quantitiy properties
      # Create empty lists for each particle class
      info["n_quantities"] = [None for _ in range(info["n_classes"])]
      info["quantity_labels"] = [[] for _ in range(info["n_classes"])]
      info["quantity_units"] = [[] for _ in range(info["n_classes"])]

      # Loop over all classes and parse their individual quantity labels
      # and units
      for c_index in range(info["n_classes"]):
        # Read number of quantities for current  class and skip a placeholder
        info["n_quantities"][c_index] = _read_from_buffer(
            _get_fds_dtype("int", count=2))[0]

        # Read particle quantities as character lists, add as strings to the
        # info lists
        for _ in range(info["n_quantities"][c_index]):
          q_label = _read_from_buffer(_get_fds_dtype("char", count=30))
          info["quantity_labels"][c_index].append(q_label.decode().strip())

          q_unit = _read_from_buffer(_get_fds_dtype("char", count=30))
          info["quantity_units"][c_index].append(q_unit.decode().strip())

      #
      # READ: particle data
      #

      # Particles in simulations may be removed, thus it quite often occurs
      # that there are no more particles in subsequent timesteps after some
      # point during the simulation. Meshes may not even contain any
      # particle at all. We are safe to stop reading the remaining file in
      # this case.
      #
      # To do so, we will determine how much space an empty timestep
      # occupies, calculate how many timesteps there would be if each of the
      # subsequent timesteps contains zero particles, and compare it to the
      # actual number of timesteps.

      # Size of zero particles for each particle class in a single timestep
      zero_particles_size = [
          _get_fds_dtype("positions", count=0).itemsize +
          _get_fds_dtype("tags", count=0).itemsize +
          (_get_fds_dtype("quantities", count=0, n_quantities=info["n_quantities"][c_index]).itemsize
              if info["n_quantities"][c_index] > 0 else 0)
          for c_index in range(info["n_classes"])]

      # Size of a timestep without any particles in all classes
      # Current timestep
      empty_timestep_size = _get_fds_dtype("float").itemsize
      for c_index in range(info["n_classes"]):
        empty_timestep_size += (
            # Number of particles is zero
            _get_fds_dtype("int").itemsize +
            # Size that zero particles occupy
            zero_particles_size[c_index]
        )

      # Create empty lists for each particle class
      times = []
      n_particles = {c_label: [] for c_label in classes_dict.values()}
      positions = {c_label: [] for c_label in classes_dict.values()}
      tags = {c_label: [] for c_label in classes_dict.values()}
      quantities = {c_label: {q_label: []
                              for q_label in info["quantity_labels"][c_index]}
                    for (c_index, c_label) in classes_dict.items()}

      t_step = 0
      while file_mm_pos < file_mm.size():
        # If all remaining timesteps have no particles, we will skip this file
        if n_timesteps:
          n_timesteps_estimate_if_remaining_steps_empty = \
              t_step + (file_mm.size() - file_mm_pos) // empty_timestep_size
          if n_timesteps_estimate_if_remaining_steps_empty == n_timesteps:
            if logs:
              print("No more particles. Skip remaining file.")
            file_mm_pos = file_mm.size()
            break

        # Decide whether we want to process this timestep
        skip_timestep = (t_step % output_each) > 0

        # Read time of current output step
        time_at_timestep = _read_from_buffer(_get_fds_dtype("float"))[0]
        if not skip_timestep:
          times.append(time_at_timestep)

        # Read data for each particle class
        for c_index in range(info["n_classes"]):
          # Decide whether we want to process this particle class
          skip_class = c_index not in classes_dict.keys()
          skip_current = skip_timestep or skip_class

          # Read number of particles
          n_particle = _read_from_buffer(_get_fds_dtype("int"))[0]

          # If no particles were found, skip this timestep
          if n_particle == 0:
            file_mm_pos += zero_particles_size[c_index]

            # Store empty data if required
            if not skip_current:
              c_label = classes_dict[c_index]
              n_particles[c_label].append(0)
              positions[c_label].append([np.array([]),
                                         np.array([]),
                                         np.array([])])
              tags[c_label].append(np.array([]))
              for q_label in info["quantity_labels"][c_index]:
                quantities[c_label][q_label].append(np.array([]))

            continue

          # Read position lists
          raw_position = _read_from_buffer(_get_fds_dtype("positions",
                                                          count=n_particle),
                                           skip=skip_current)
          # Read tags
          raw_tag = _read_from_buffer(_get_fds_dtype("tags",
                                                     count=n_particle),
                                      skip=skip_current)
          # Read each quantity data, if there is any
          raw_quantity = None
          if info["n_quantities"][c_index] > 0:
            raw_quantity = _read_from_buffer(_get_fds_dtype("quantities",
                                                            count=n_particle,
                                                            n_quantities=info["n_quantities"][c_index]),
                                             skip=skip_current)

          # Store data if required
          if not skip_current:
            c_label = classes_dict[c_index]
            n_particles[c_label].append(n_particle)
            positions[c_label].append([np.copy(raw_position[0]),
                                       np.copy(raw_position[1]),
                                       np.copy(raw_position[2])])
            tags[c_label].append(np.copy(raw_tag))
            for (q_index, q_label) in enumerate(info["quantity_labels"][c_index]):
              quantities[c_label][q_label].append(np.copy(raw_quantity[q_index]))

        # Continue with next timestep
        t_step += 1

      assert file_mm_pos == file_mm.size()

      info["n_timesteps"] = t_step
      n_outputsteps = len(times)

      if logs:
        data_size = file_mm.size()
        data_size_in_mb = data_size / (1024 ** 2)
        time_end = time()
        print("file size: {:.3f} MB, speed: {:.3f} MB/s".format(
              data_size_in_mb, data_size_in_mb / (time_end - time_start)))

      file_object.close()

      return (info, n_outputsteps, times, n_particles, positions, tags, quantities)


    def _read_multiple_prt5_files(filestem, classes_dict,
                                  output_each=1, logs=True, parallel=True):
      """
      Function to parse and read in all particle data written by FDS from
      multiple meshes into multiple files.
      Parameters
      ----------
      filestem : str
        All files beginning with filestem and ending with prt5 will be read.
      classes_dict : dict -> (i1:s2)
        Specify all particle classes to be read. Translates particle class
        indices i1 occurring in the smv and prt5 file to the corresponding
        class label s2 from the smv file.
      output_each : int
        Each n-th timestep will actually be read, those inbetween will be
        skipped. By default, every timestep will be considered.
      logs : bool
        Choose if logs will be printed out. This feature is enabled by
        default.
      parallel : bool
        Decide if files from filesystem will be read in parallel. This
        feature is enabled by default.
      Returns
      -------
      (info, n_outputsteps, times, n_particles, positions, tags, quantities)
      with:
      info : dict
        Contains meta information about the original FDS dataset, as:
          - filename : str
              The name of all read datafiles in wildcard notation
          - fds_version : str
              Particle data has been generated with this FDS version
          - n_classes : int
              Number of particle classes in the simulation
          - n_timesteps : int
              Number of total timesteps in the FDS simulation
          - n_quantities : list[i1] -> int
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

      # Verify user input
      filename_wildcard = filestem + "*.prt5"
      filelist = sorted(glob(filename_wildcard))
      assert len(filelist) > 0, \
          "No files were found with the specified credentials."

      #
      # READ: all files
      #

      # Read very first file to know the total number of timesteps
      result_first = _read_prt5_file(filelist.pop(0),
                                     classes_dict, output_each,
                                     n_timesteps=None, logs=logs)
      # If there are no more files, we are done
      if not filelist:
        return result_first

      # Extract global information
      info = result_first[0]
      info["filename"] = filename_wildcard
      n_outputsteps = result_first[1]
      times = result_first[2]

      # Read remaining input files
      results = [result_first]
      if parallel:
        pool = Pool(None)
        worker = partial(_read_prt5_file,
                         classes_dict=classes_dict, output_each=output_each,
                         n_timesteps=info["n_timesteps"], logs=logs)
        results[1:] = pool.map(worker, filelist)
        pool.close()
        pool.join()
      else:
        for filename in filelist:
          results.append(
              _read_prt5_file(filename,
                              classes_dict, output_each,
                              info["n_timesteps"], logs))

      if logs:
        print("Finished reading.")

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
                              for q_label in info["quantity_labels"][c_index]}
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
            positions[c_label][o_step][0][o:o+n] = \
                np.copy(local_positions[c_label][o_step][0])
            positions[c_label][o_step][1][o:o+n] = \
                np.copy(local_positions[c_label][o_step][1])
            positions[c_label][o_step][2][o:o+n] = \
                np.copy(local_positions[c_label][o_step][2])
            tags[c_label][o_step][o:o+n] = \
                np.copy(local_tags[c_label][o_step])
            for q_label in local_quantities[c_label].keys():
              quantities[c_label][q_label][o_step][o:o+n] = \
                  np.copy(local_quantities[c_label][q_label][o_step])

            offsets[c_label][o_step] += n

      for c_label in classes_dict.values():
        for (o_step, n_particle) in enumerate(n_particles[c_label]):
          assert offsets[c_label][o_step] == n_particle

      return info, n_outputsteps, times, n_particles, positions, tags, quantities
