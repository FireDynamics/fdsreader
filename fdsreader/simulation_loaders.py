"""Per-datatype loading methods for :class:`fdsreader.simulation.Simulation`, extracted into a
mixin purely for file-size/readability reasons (fdsreader/simulation.py used to hold every
loader plus the SMV parser, caching machinery, and public collections in one file). This is a
pure structural move: every method here is unchanged from its previous home on `Simulation`
itself, and since a mixin shares `self` with the class that uses it, all existing state access
(`self._meshes`, `self.load_errors` via the `@log_error` decorator, etc.) keeps working exactly
as before.
"""

import csv
import glob
import os
from typing import Dict, List, Sequence, TextIO, Tuple, Union

import numpy as np

import fdsreader.utils.fortran_data as fdtype
from fdsreader.bndf import Obstruction, Patch, SubObstruction
from fdsreader.devc import Device
from fdsreader.evac import EvacCollection, Evacuation
from fdsreader.fds_classes import Mesh, Surface, Ventilation
from fdsreader.geom import GeomBoundary, Geometry
from fdsreader.isof import Isosurface
from fdsreader.part import Particle, ParticleCollection
from fdsreader.smoke3d import Smoke3D
from fdsreader.utils import Dimension, Extent, Quantity, log_error
from fdsreader.utils.data import Profile


class SimulationLoaderMixin:
    """Mixin providing all per-datatype SMV/CSV/binary loading methods used by `Simulation`."""

    def _load_mesh(self, smv_file: TextIO, line: str) -> Mesh:
        """Load information for a single mesh from the smv file at current pointer position."""
        mesh_id = "".join(line.split()[1:])

        grid_numbers = smv_file.readline().strip().split()
        grid_dimensions = {"x": int(grid_numbers[0]) + 1, "y": int(grid_numbers[1]) + 1, "z": int(grid_numbers[2]) + 1}

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "PDIM"
        coordinates = dict()
        extents = smv_file.readline().split()
        extents = {
            "x": (float(extents[0]), float(extents[1])),
            "y": (float(extents[2]), float(extents[3])),
            "z": (float(extents[4]), float(extents[5])),
        }

        for dim in ("x", "y", "z"):
            smv_file.readline()  # Blank line
            assert smv_file.readline().strip()[:3] == "TRN"
            noc = int(smv_file.readline().strip())
            for _ in range(noc):
                smv_file.readline()
            coordinates[dim] = np.empty(grid_dimensions[dim], dtype=np.float32)
            for i in range(grid_dimensions[dim]):
                coordinates[dim][i] = float(smv_file.readline().split()[1])

        # Mesh ids might not be unique if using MULT in combination with a MESH with its ID set
        for other_mesh in reversed(self._meshes):
            if other_mesh.id.startswith(mesh_id):
                if mesh_id == other_mesh.id:  # first mesh with same id
                    mesh_id = mesh_id + "_1"
                else:  # n-th mesh with same id
                    # apply increment, only if the naming of the mesh id allows for it
                    if len(other_mesh.id.split("_")) == 2:
                        other_mesh_id, last_num = other_mesh.id.split("_")
                        if other_mesh_id == mesh_id and last_num.isdigit():
                            mesh_id = mesh_id + "_" + str(int(last_num) + 1)

        mesh = Mesh(coordinates, extents, mesh_id)

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "OBST"
        self._load_obstructions(smv_file, mesh)

        smv_file.readline()  # Blank line
        assert smv_file.readline().strip() == "VENT"
        self._load_vents(smv_file, mesh)

        return mesh

    @log_error("obst")
    def _load_obstructions(self, smv_file: TextIO, mesh: Mesh):
        temp_data = list()
        n = int(smv_file.readline().strip())

        if n > 0:
            self._subobstructions[mesh.id] = list()

        for _ in range(n):
            line = smv_file.readline().strip().split("!")
            line_floats = line[0].strip().split()
            ext = [float(line_floats[i]) for i in range(6)]
            # The ordinal is negative if the obstruction was created due to a hole, the negative
            # sign is ignored here and obstructions created by FDS due to holes are handled later
            # when there are multiple obstructions with the same ID
            obst_id = line[1].strip() if len(line) > 1 else str(abs(int(line_floats[6])))

            side_surfaces = tuple(self.surfaces[int(line_floats[i])] for i in range(7, 13))
            if len(line_floats) > 13:
                texture_origin = (float(line_floats[13]), float(line_floats[14]), float(line_floats[15]))
            else:
                texture_origin = self.default_texture_origin

            # Check if there is already an obst in this mesh with the same ID (due to holes).
            # This will only be the case the first time we notice that there are multiple
            # obstructions with the same name, as their IDs will now change...
            for obst_data in temp_data:
                if obst_data[0] == obst_id:
                    # The obstruction that was already added to the temp_data has to receive an
                    # updated ID as well, so the user can recognize it as special obstruction
                    obst_data[0] = obst_data[0] + "_from-hole-1"

                    # Now set the next obst_id to '_from-hole-2'
                    obst_id += "_from-hole-2"

            # ...For subsequent cases (i.e., when the third or fourth obstruction with the same id
            # is found), we need to catch the case differently
            for obst_data in reversed(temp_data):
                if obst_data[0][:-1] == obst_id + "_from-hole-":
                    # Find the last obstruction with the same id and set the number of the current
                    # one accordingly
                    obst_id += "_from-hole-" + str(int(obst_data[0][-1]) + 1)

            temp_data.append([obst_id, Extent(*ext), side_surfaces, texture_origin])

        for obst_data in temp_data:
            obst_id, extent, side_surfaces, texture_origin = obst_data

            line = smv_file.readline().strip().split()
            bound_indices = (
                int(float(line[0])),
                int(float(line[1])),
                int(float(line[2])),
                int(float(line[3])),
                int(float(line[4])),
                int(float(line[5])),
            )
            color_index = int(line[6])
            block_type = int(line[7])
            rgba = tuple(float(line[i]) for i in range(8, 12)) if color_index == -3 else ()

            obst = next((o for o in self._obstructions if obst_id == o.id), None)
            if obst is None:
                obst = Obstruction(obst_id, color_index, block_type, texture_origin, rgba)
                self._obstructions.append(obst)

            if not any(obst_id == o.id for o in mesh.obstructions):
                mesh.obstructions.append(obst)

            subobst = SubObstruction(side_surfaces, bound_indices, extent, mesh)

            self._subobstructions[mesh.id].append(subobst)
            obst._subobstructions.setdefault(mesh.id, []).append(subobst)

    def _toggle_obst(self, smv_file: TextIO, line: str):
        line = line.split()
        mesh = self._meshes[int(line[-1]) - 1]

        obst_index, time = smv_file.readline().split()
        time = float(time)
        subobst = self._subobstructions[mesh.id][int(obst_index) - 1]

        if "HIDE_OBST" in line[0]:
            subobst._hide(time)
        else:
            subobst._show(time)

    def _load_geoms(self, smv_file: TextIO, line: str):
        ngeoms = int(line.split()[1])

        filename = smv_file.readline().strip()
        file_path = os.path.join(self.root_path, filename)

        for g in range(ngeoms):
            line = smv_file.readline().split("!")
            texture_line = line[0].split()
            rgb_line = line[1].split()

            texture_mapping = texture_line[0]
            texture_origin = (float(texture_line[1]), float(texture_line[2]), float(texture_line[3]))
            is_terrain = bool(int(texture_line[4]))
            rgb = (int(rgb_line[0]), int(rgb_line[1]), int(rgb_line[2]))

            if "%" in line[0]:
                surface_id = line[0].split("%")[-1].strip()
                surface = next((s for s in self.surfaces if s.id == surface_id), None)
                geom = Geometry(file_path, texture_mapping, texture_origin, is_terrain, rgb, surface=surface)
            else:
                geom = Geometry(file_path, texture_mapping, texture_origin, is_terrain, rgb)
            self.geoms.append(geom)

    @log_error("vents")
    def _load_vents(self, smv_file: TextIO, mesh: Mesh):
        line = smv_file.readline().split()
        n, n_dummies = int(line[0]), int(line[1])

        temp_data = list()

        def read_common_info():
            # Strip inline FDS comments (everything from '!' onward) before parsing
            line = smv_file.readline().split("!")[0].strip().split()
            return line, [float(line[i]) for i in range(6)], int(line[6]) - 1, self.surfaces[int(line[7])]

        def read_common_info2():
            # Strip inline FDS comments (everything from '!' onward) before parsing
            line = smv_file.readline().split("!")[0].strip().split()
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
            line, ext, vent_index, surface = read_common_info()
            texture_origin = (float(line[8]), float(line[9]), float(line[10]))
            temp_data.append((Extent(*ext), vent_index, surface, texture_origin))

        for _ in range(n_dummies):
            _, ext, vent_index, surface = read_common_info()
            temp_data.append((Extent(*ext), vent_index, surface))

        for v in range(n):
            if v < n - n_dummies:
                extent, vent_index, surface, texture_origin = temp_data[v]
            else:
                extent, vent_index, surface = temp_data[v]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            if vent_index not in self.ventilations:
                self.ventilations[vent_index] = Ventilation(
                    surface, bound_indices, color_index, draw_type, rgba=rgba, texture_origin=texture_origin
                )
            self.ventilations[vent_index]._add_subventilation(mesh, extent)

        smv_file.readline()
        assert "CVENT" in smv_file.readline()

        n = int(smv_file.readline().strip())
        temp_data.clear()
        for _ in range(n):
            line, extent, vent_index, surface = read_common_info()
            circular_vent_origin = (float(line[12]), float(line[13]), float(line[14]))
            radius = float(line[15])
            temp_data.append((extent, vent_index, surface, texture_origin, circular_vent_origin, radius))

        for v in range(n):
            extent, vent_index, surface, texture_origin, circular_vent_origin, radius = temp_data[v]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            if vent_index not in self.ventilations:
                self.ventilations[vent_index] = Ventilation(
                    surface,
                    bound_indices,
                    color_index,
                    draw_type,
                    rgba=rgba,
                    texture_origin=texture_origin,
                    circular_vent_origin=circular_vent_origin,
                    radius=radius,
                )
            self.ventilations[vent_index]._add_subventilation(mesh, extent)

    @log_error("surface")
    def _load_surface(self, smv_file: TextIO) -> Surface:
        """Load the information for a single surface from the smv file at current pointer position."""

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

        return Surface(
            surface_id,
            tmpm,
            material_emissivity,
            surface_type,
            texture_width,
            texture_height,
            texture_map,
            rgb,
            transparency,
        )

    @log_error("slcf")
    def _load_slice(self, smv_file: TextIO, line: str):
        """Loads the slice at current pointer position."""
        if "SLCC" in line:
            cell_centered = True
        else:
            cell_centered = False

        slice_index = int(line.split("!")[1].strip().split()[0])

        slice_id = line.split("%")[1].split("&")[0].strip() if "%" in line else ""

        mesh_index = int(line.split("&")[0].strip().split()[1]) - 1
        mesh = self._meshes[mesh_index]

        # Read in index ranges for x, y and z
        bound_indices = [int(i.strip()) for i in line.split("&")[1].split("!")[0].strip().split()]
        extent, dimension = self._indices_to_extent(bound_indices, mesh)

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        if slice_index not in self._slices:
            self._slices[slice_index] = [{"cell_centered": cell_centered, "id": slice_id}]
        self._slices[slice_index].append(
            {
                "dimension": dimension,
                "extent": extent,
                "mesh": mesh,
                "filename": filename,
                "quantity": quantity,
                "short_name": short_name,
                "unit": unit,
            }
        )

    @log_error("slcf")
    def _load_geomslice(self, smv_file: TextIO, line: str):
        """Loads the geomslice at current pointer position."""
        slice_index = int(line.split("!")[1].strip().split()[0])

        slice_id = "".join(line.split("%")[1].split("&")).strip() if "%" in line else ""

        mesh_index = int(line.split("&")[0].strip().split()[1]) - 1
        mesh = self._meshes[mesh_index]

        # Read in index ranges for x, y and z
        bound_indices = [int(i.strip()) for i in line.split("&")[1].split("!")[0].strip().split()]
        extent, _ = self._indices_to_extent(bound_indices, mesh)

        filename = smv_file.readline().strip()
        geom_filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        file_path = os.path.join(self.root_path, filename)

        if os.path.exists(file_path + ".bnd"):
            times = list()
            with open(file_path + ".bnd") as bnd_file:
                for line in bnd_file:
                    times.append(float(line.split()[0]))
            times = np.array(times)
        else:
            times = None

        if slice_index not in self._geomslices:
            self._geomslices[slice_index] = [{"times": times, "id": slice_id}]
        self._geomslices[slice_index].append(
            {
                "extent": extent,
                "mesh": mesh,
                "filename": filename,
                "geomfilename": geom_filename,
                "quantity": quantity,
                "short_name": short_name,
                "unit": unit,
            }
        )

    @log_error("bndf")
    def _load_boundary_data(self, smv_file: TextIO, line: str, cell_centered: bool):
        """Loads the boundary data at current pointer position."""
        line = line.split()
        mesh_index = int(line[1]) - 1
        mesh = self._meshes[mesh_index]

        filename = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        bid = int(filename.split("_")[-1][:-3]) - 1

        file_path = os.path.join(self.root_path, filename)

        patches = dict()
        mesh_patches = dict()

        lower_bounds = np.array([np.float32(+1e33)], dtype=np.float32)
        upper_bounds = np.array([np.float32(-1e33)], dtype=np.float32)

        with open(file_path, "rb") as infile:
            # Offset of the binary file to the end of the file header.
            initial_offset = 3 * fdtype.new((("c", 30),)).itemsize
            infile.seek(initial_offset)

            n_patches = fdtype.read(infile, fdtype.INT, 1)[0][0][0]

            dtype_patches = fdtype.new((("i", 9),))
            patch_infos = fdtype.read(infile, dtype_patches, n_patches)
            initial_offset += fdtype.INT.itemsize + dtype_patches.itemsize * n_patches
            patch_offset = fdtype.FLOAT.itemsize

            # Determine the size of the data block for all patches
            patches_data_bytes = 0
            for patch_info in patch_infos:
                patch_info = patch_info[0]
                extent, dimension = self._indices_to_extent(patch_info[:6], mesh)
                patches_data_bytes += fdtype.new((("f", str(dimension.shape(cell_centered=False))),)).itemsize

            # Time info
            time_bytes = fdtype.FLOAT.itemsize
            n_t = (os.stat(file_path).st_size - initial_offset) // (time_bytes + patches_data_bytes)

            # Read all time values in a single pass instead of one record (plus a seek) at a
            # time, which is far slower for files with many timesteps: skip over each patches
            # data block via a raw padding field instead of parsing and discarding it.
            #
            # Trade-off: unlike the seek-based approach below, np.fromfile physically transfers
            # the skipped patches-data bytes from disk (numpy has no way to make a void/padding
            # field skip bytes at the OS level) -- it just discards them after reading. This is
            # still faster in every real case we've benchmarked (up to 148MB simulations) because
            # eliminating the per-record Python loop/syscall overhead outweighs the extra I/O
            # volume, but for a single, very large boundary file (many patches x large spatial
            # extent) with comparatively few timesteps, the old seek-based loop could in theory
            # transfer fewer bytes overall. If that ever becomes a real bottleneck, the old
            # approach was:
            #
            # times = list()
            # offset = initial_offset
            # for _ in range(n_t):
            #     time = fdtype.read(infile, fdtype.FLOAT, 1)[0][0][0]
            #     times.append(time)
            #     offset += time_bytes + patches_data_bytes
            #     infile.seek(offset)
            if n_t > 0:
                if patches_data_bytes > 0:
                    row_dtype = np.dtype([("time", fdtype.FLOAT), ("_skip", "V" + str(patches_data_bytes))])
                else:
                    row_dtype = np.dtype([("time", fdtype.FLOAT)])
                rows = np.fromfile(infile, dtype=row_dtype, count=n_t)
                times = rows["time"]["f1"].reshape(-1).tolist()
            else:
                times = list()

            for patch_info in patch_infos:
                patch_info = patch_info[0]

                extent, dimension = self._indices_to_extent(patch_info[:6], mesh)
                orientation = patch_info[6]
                obst_index = patch_info[7] - 1

                p = Patch(
                    file_path, dimension, extent, orientation, cell_centered, patch_offset, initial_offset, n_t, mesh
                )

                # "Obstacles" with index -1 give the extent of the (whole) mesh faces and refer to
                # "closed" mesh faces, therefore that data will be added to the corresponding mesh instead
                if obst_index != -1:
                    if obst_index not in patches:
                        patches[obst_index] = list()
                    patches[obst_index].append(p)
                else:
                    if mesh.id not in mesh_patches:
                        mesh_patches[mesh.id] = list()
                    mesh_patches[mesh.id].append(p)

                patch_offset += fdtype.new((("f", str(p.dimension.shape(cell_centered=False))),)).itemsize

        for obst_index, p in patches.items():
            for patch in p:
                patch._post_init(patch_offset)

            self._subobstructions[mesh.id][obst_index]._add_patches(
                bid, cell_centered, quantity, short_name, unit, p, times, lower_bounds, upper_bounds
            )

        for p in mesh_patches.values():
            for patch in p:
                patch._post_init(patch_offset)
            patch.mesh._add_patches(
                bid, cell_centered, quantity, short_name, unit, p, times, lower_bounds, upper_bounds
            )

    @log_error("geom")
    def _load_boundary_data_geom(self, smv_file: TextIO, line: str):
        line = line.split()
        mesh_index = int(line[1]) - 1
        # Meshes are not loaded yet
        # mesh = self.meshes[mesh_index]

        filename_be = smv_file.readline().strip()
        filename_gbf = smv_file.readline().strip()
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        bid = int(filename_be.split("_")[-1][:-3]) - 1

        file_path_be = os.path.join(self.root_path, filename_be)
        if filename_gbf == "-":
            # FDS 6.10+ no longer writes a per-quantity .gbf file. The boundary geometry
            # (vertices/faces) instead lives in a per-mesh .gcf file with a deterministic
            # name (see FN_CFACE_GEOM in FDS' Source/dump.f90).
            is_gcf = True
            file_path_gbf = os.path.join(self.root_path, f"{self.chid}_{mesh_index + 1}.gcf")
        else:
            is_gcf = False
            file_path_gbf = os.path.join(self.root_path, filename_gbf)

        times = list()
        lower_bounds = list()
        upper_bounds = list()
        with open(file_path_be + ".bnd") as bnd_file:
            for line in bnd_file:
                splits = line.split()
                times.append(float(splits[0]))
                lower_bounds.append(float(splits[1]))
                upper_bounds.append(float(splits[2]))
        times = np.array(times)
        lower_bounds = np.array(lower_bounds, dtype=np.float32)
        upper_bounds = np.array(upper_bounds, dtype=np.float32)
        n_t = times.shape[0]

        if bid >= len(self._geom_data):
            self._geom_data.append(GeomBoundary(Quantity(quantity, short_name, unit), times, n_t))
        self._geom_data[bid]._add_data(mesh_index, file_path_be, file_path_gbf, is_gcf, lower_bounds, upper_bounds)

    @log_error("pl3d")
    def _load_plot_3d(self, smv_file: TextIO, line: str):
        """Loads the pl3d at current pointer position."""
        line = line.strip().split()

        time = float(line[1])

        mesh_index = int(line[2]) - 1

        filename = smv_file.readline().strip()
        for i in range(5):
            quantity = smv_file.readline().strip()
            short_name = smv_file.readline().strip()
            unit = smv_file.readline().strip()

            self.data_3d[i]._add_subplot(
                filename, time, Quantity(quantity, short_name, unit), i, self._meshes[mesh_index]
            )

    @log_error("smoke3d")
    def _load_smoke_3d(self, smv_file: TextIO, line: str):
        """Loads the smoke3d at current pointer position."""
        line = line.strip().split()

        mesh_index = int(line[1]) - 1

        filename = smv_file.readline().strip()

        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()

        sz_file_path = os.path.join(self.root_path, filename + ".sz")
        with open(sz_file_path) as sizefile:
            data_lines = sizefile.readlines()[1:]  # Skip the version line.
        if data_lines:
            try:
                # Columns are: time, some int, some int, upper_bound (only the first and last
                # are used). This assumes every row has the same number of columns, which is
                # true in practice almost always, but not guaranteed by the file format.
                data = np.loadtxt(data_lines, ndmin=2)
                times = data[:, 0]
                upper_bounds = data[:, -1]
            except ValueError:
                # Fall back to parsing just the first/last whitespace-separated token per line,
                # which tolerates rows with a differing number of columns.
                times = np.array([float(line.split()[0]) for line in data_lines])
                upper_bounds = np.array([float(line.split()[-1]) for line in data_lines])
        else:
            times = np.array([])
            upper_bounds = np.array([])

        quantity = Quantity(quantity, short_name, unit)

        if quantity not in self._smoke_3d:
            self._smoke_3d[quantity] = Smoke3D(self.root_path, times, quantity)
        self._smoke_3d[quantity]._add_subsmoke(filename, self._meshes[mesh_index], upper_bounds)

    @log_error("isof")
    def _load_isosurface(self, smv_file: TextIO, line: str):
        """Loads the isosurface at current pointer position."""
        double_quantity = line[0] == "T"
        mesh_index = int(line.strip().split()[1]) - 1

        iso_filename = smv_file.readline().strip()
        iso_id = int(iso_filename.split("_")[-1][:-4])
        iso_file_path = os.path.join(self.root_path, iso_filename)

        if double_quantity:
            viso_file_path = os.path.join(self.root_path, smv_file.readline().strip())
        quantity = smv_file.readline().strip()
        short_name = smv_file.readline().strip()
        unit = smv_file.readline().strip()
        if double_quantity:
            v_quantity = smv_file.readline().strip()
            v_short_name = smv_file.readline().strip()
            v_unit = smv_file.readline().strip()

        if iso_id not in self._isosurfaces:
            with open(iso_file_path, "rb") as infile:
                nlevels = fdtype.read(infile, fdtype.INT, 3)[2][0][0]

                dtype_header_levels = fdtype.new((("f", nlevels),))
                levels = fdtype.read(infile, dtype_header_levels, 1)[0]
        if double_quantity:
            if iso_id not in self._isosurfaces:
                self._isosurfaces[iso_id] = Isosurface(
                    iso_id,
                    double_quantity,
                    quantity,
                    short_name,
                    unit,
                    levels,
                    v_quantity=v_quantity,
                    v_short_name=v_short_name,
                    v_unit=v_unit,
                )
            self._isosurfaces[iso_id]._add_subsurface(
                self._meshes[mesh_index], iso_file_path, viso_file_path=viso_file_path
            )
        else:
            if iso_id not in self._isosurfaces:
                self._isosurfaces[iso_id] = Isosurface(iso_id, double_quantity, quantity, short_name, unit, levels)
            self._isosurfaces[iso_id]._add_subsurface(self._meshes[mesh_index], iso_file_path)

    @log_error("part")
    def _register_particle(self, smv_file: TextIO) -> Particle:
        particle_class = smv_file.readline().strip()
        color = tuple(float(c) for c in smv_file.readline().strip().split())

        n_quantities = int(smv_file.readline().strip())
        quantities = list()
        for _ in range(n_quantities):
            quantity = smv_file.readline().strip()
            short_name = smv_file.readline().strip()
            unit = smv_file.readline().strip()
            quantities.append(Quantity(quantity, short_name, unit))
        return Particle(particle_class, quantities, color)

    def _load_prt5_meta(
        self,
        prts: Union[List[Particle], ParticleCollection, List[Evacuation], EvacCollection],
        file_path: str,
        mesh: Mesh,
    ) -> List[float]:
        is_evac = isinstance(prts[0], Evacuation)
        with open(file_path) as bnd_file:
            line = bnd_file.readline().strip().split()
            n_classes = int(line[1])
            times = list()
            n_quantities = list()
            for i in range(n_classes):
                line = bnd_file.readline().strip().split()
                n_quantities.append(int(line[0]))
                for _ in range(n_quantities[-1]):
                    bnd_file.readline()
                if is_evac:
                    prts[i].n_humans[mesh.id] = list()
                else:
                    prts[i].n_particles[mesh.id] = list()
            bnd_file.seek(0)

            for line in bnd_file:
                times.append(float(line.strip().split()[0]))
                for i in range(n_classes):
                    prt = prts[i]
                    n = int(bnd_file.readline().strip().split()[1].strip())
                    if is_evac:
                        prt.n_humans[mesh.id].append(n)
                    else:
                        prt.n_particles[mesh.id].append(n)
                    for q in range(n_quantities[i]):
                        line = bnd_file.readline().strip().split()
                        quantity = prt.quantities[q].name
                        prt.lower_bounds[quantity].append(float(line[0]))
                        prt.upper_bounds[quantity].append(float(line[1]))
        return times

    @log_error("part")
    def _load_particle_data(self, smv_file: TextIO, line: str):
        file_path = os.path.join(self.root_path, smv_file.readline().strip())

        mesh_index = int(line.split()[1].strip()) - 1
        mesh = self._meshes[mesh_index]

        times = self._load_prt5_meta(self._particles, file_path + ".bnd", mesh)
        if isinstance(self._particles, list):
            self._particles = ParticleCollection(times, self._particles)

        self._particles._file_paths[mesh.id] = file_path

        n_classes = int(smv_file.readline().strip())
        for i in range(n_classes):
            smv_file.readline()  # Skip "N" values

    @log_error("evac")
    def _register_evac(self, smv_file: TextIO) -> Evacuation:
        class_name = smv_file.readline().split(" % % ")[0].strip()
        color = tuple(float(c) for c in smv_file.readline().strip().split())

        n_quantities = int(smv_file.readline().strip())
        quantities = list()
        for _ in range(n_quantities):
            quantity = smv_file.readline().strip()
            short_name = smv_file.readline().strip()
            unit = smv_file.readline().strip()
            quantities.append(Quantity(quantity, short_name, unit))
        return Evacuation(class_name, quantities, color)

    @log_error("evac")
    def _load_evac_data(self, smv_file: TextIO, line: str):
        file_path = os.path.join(self.root_path, smv_file.readline().strip())

        mesh_index, z_offset = line.split()[1:]
        mesh = self._meshes[int(mesh_index) - 1]

        times = self._load_prt5_meta(self._evacs, file_path + ".bnd", mesh)[1:]  # First timestep is weird somehow
        if isinstance(self._evacs, list):
            self.evacs = EvacCollection(self._evacs, os.path.join(self.root_path, self.chid + "_evac"), times)
            # Mirror the assignment onto _evacs too, so a later mesh's EVA5 block sees the guard
            # above as already satisfied instead of silently rebuilding (and discarding
            # z_offsets/_file_paths recorded so far) on every subsequent mesh.
            self._evacs = self.evacs

        self.evacs.z_offsets[mesh.id] = float(z_offset)
        self.evacs._file_paths[mesh.id] = file_path

        n_evacs = int(smv_file.readline().strip())
        for i in range(n_evacs):
            smv_file.readline()  # Skip "N" values

    @log_error("prof")
    def _load_profiles(self):
        for f in glob.glob(str(os.path.join(self.root_path, self.chid)) + "_prof*"):
            with open(f) as infile:
                infile.readline()  # Skip "ID, IOR, face center x(m), ..." header
                profile_id = infile.readline().split(",")[0].strip()
                infile.readline()  # Skip "Time(s), Npoints, ..." header
                raw_rows = [row for row in csv.reader(infile) if row]

            # Npoints can change between timesteps (FDS shrinks/grows the profile), which
            # produces rows with different lengths - np.genfromtxt can't handle that, so each row
            # is parsed on its own instead. Only trailing padding is stripped (not fields in the
            # middle of a row), and a row that's shorter than its own declared Npoints - e.g. the
            # last line of a profile still being written by a running simulation - is skipped
            # rather than raising, so one incomplete row doesn't take out every other profile file.
            times_list, npoints_list, depths_list, values_list = [], [], [], []
            for row in raw_rows:
                while row and row[-1].strip() == "":
                    row.pop()
                if len(row) < 2:
                    continue
                try:
                    row_values = np.array(row, dtype=np.float32)
                except ValueError:
                    continue
                n = int(row_values[1])
                if len(row_values) < 2 + 2 * n:
                    continue
                times_list.append(row_values[0])
                npoints_list.append(n)
                depths_list.append(row_values[2 : 2 + n])
                values_list.append(row_values[2 + n : 2 + 2 * n])

            times = np.array(times_list, dtype=np.float32)
            npoints = np.array(npoints_list, dtype=int)
            depths = np.empty(len(depths_list), dtype=object)
            depths[:] = depths_list
            values = np.empty(len(values_list), dtype=object)
            values[:] = values_list

            self.profiles[profile_id] = Profile(profile_id, times, npoints, depths, values)

    @log_error("devc")
    def _register_device(self, smv_file: TextIO) -> Tuple[str, Device]:
        line = smv_file.readline().strip().split("%")
        device_id = line[0].strip()
        quantity = None
        if len(line) > 1:
            quantity_name = line[1].strip()
            quantity = Quantity(quantity_name, quantity_name, "")
        line = smv_file.readline().strip().split("#")[0].split()
        position = (float(line[0]), float(line[1]), float(line[2]))
        orientation = (float(line[3]), float(line[4]), float(line[5]))
        return device_id, Device(device_id, quantity, position, orientation)

    def _load_DEVC_data(self):
        with open(self.devc_path) as infile:
            units = [unit.strip() for unit in infile.readline().split(",")]
            # Device names are only quoted by FDS when they contain a comma or space, so a plain
            # split on "," or ',"' would misparse a header where none (or all) of the names need
            # quoting. Use a real CSV parser instead so it doesn't matter which fields are quoted.
            names = [name.strip() for name in next(csv.reader([infile.readline()]))]
            values = np.genfromtxt(infile, delimiter=",", dtype=np.float32, autostrip=True)
            for k in range(len(names)):
                if isinstance(self.devices[names[k]], list):
                    for devc in self.devices[names[k]]:
                        if not hasattr(devc, "_data"):
                            # Find the first device in the list that does not yet have any data associated with it
                            break
                else:
                    devc = self.devices[names[k]]

                devc.quantity.unit = units[k]
                devc._data = values[:, k].copy()

        line_path = os.path.join(self.root_path, self.chid + "_line.csv")
        if os.path.exists(line_path):
            with open(line_path) as infile:
                units = [unit.replace('"', "").strip() for unit in infile.readline().split(",")]
                names = [name.strip() for name in next(csv.reader([infile.readline()]))]
                data = np.genfromtxt(infile, delimiter=",", dtype=np.float32, autostrip=True)
                for k, key in enumerate(names):
                    if key in self.devices:
                        devc = self.devices[key]
                        devices = devc if isinstance(devc, list) else [devc]
                        for i, d in enumerate(devices):
                            d.quantity.unit = units[k]
                            d._data = data[i, k]
                    else:
                        pass  # Probably only x,y,z coordinates

    @log_error("csv")
    def _load_HRR_data(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path) as infile:
            infile.readline()
            keys = [name.replace('"', "").replace("\n", "").strip() for name in infile.readline().split(",")]

        values = np.loadtxt(file_path, delimiter=",", ndmin=2, skiprows=2)
        return self._transform_csv_data(keys, values)

    @log_error("csv")
    def _load_step_data(self, file_path: str) -> Dict[str, np.ndarray]:
        with open(file_path) as infile:
            infile.readline()
            keys = [name.replace('"', "").replace("\n", "").strip() for name in infile.readline().split(",")][2:]
        timesteps = np.loadtxt(
            file_path, dtype=np.dtype("datetime64[ms]"), delimiter=",", ndmin=1, usecols=1, skiprows=2
        )
        float_values = np.loadtxt(file_path, delimiter=",", usecols=range(2, len(keys) + 2), ndmin=2, skiprows=2)
        data = self._transform_csv_data(keys, float_values)
        data["Time Step"] = timesteps
        return data

    @log_error("csv")
    def _load_named_csv_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """Loads a generic two-header-row FDS csv export (e.g. CTRL or mass/species history)."""
        with open(file_path) as infile:
            infile.readline()  # Skip units header
            keys = [name.strip() for name in next(csv.reader([infile.readline()]))]
        values = np.loadtxt(file_path, delimiter=",", ndmin=2, skiprows=2)
        if values.size == 0:
            # np.loadtxt can't infer the column count from zero data rows (e.g. a run that was
            # killed before its first sample was written), so it comes back as shape (0, 1).
            values = np.empty((0, len(keys)))
        return self._transform_csv_data(keys, values)

    @log_error("csv")
    def _load_CPU_data(self) -> Dict[str, np.ndarray]:
        file_path = os.path.join(self.root_path, self.chid + "_cpu.csv")
        if os.path.exists(file_path):
            with open(file_path) as infile:
                keys = [name.replace('"', "").replace("\n", "").strip() for name in infile.readline().split(",")]
            values = np.loadtxt(file_path, delimiter=",", ndmin=2, skiprows=1)
        else:
            return dict()
        data = self._transform_csv_data(keys, values)
        data["Rank"] = data["Rank"].astype(int)
        return data

    def _transform_csv_data(self, keys, values):
        return {key: values[:, i] for i, key in enumerate(keys)}

    def _indices_to_extent(self, indices: Sequence[Union[int, str]], mesh: Mesh) -> Tuple[Extent, Dimension]:
        co = mesh.coordinates

        indices = tuple(int(index) for index in indices)

        x_min, x_max, y_min, y_max, z_min, z_max = indices
        co_x_min, co_x_max, co_y_min, co_y_max, co_z_min, co_z_max = (
            co["x"][x_min],
            co["x"][x_max],
            co["y"][y_min],
            co["y"][y_max],
            co["z"][z_min],
            co["z"][z_max],
        )
        dimension = Dimension(indices[1] - indices[0] + 1, indices[3] - indices[2] + 1, indices[5] - indices[4] + 1)

        extent = Extent(co_x_min, co_x_max, co_y_min, co_y_max, co_z_min, co_z_max)
        return extent, dimension
