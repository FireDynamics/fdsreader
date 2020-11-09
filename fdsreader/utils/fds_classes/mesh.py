import mmap
from typing import Dict, List, Union, Tuple
import numpy as np

from utils import Extent, Obstruction, Ventilation, Surface


class Mesh:
    """
    3-dimensional Mesh of fixed, defined size.
    :ivar coordinates: Coordinate values for each of the 3 dimension.
    :ivar extent: Tuple with three tuples containing minimum and maximum coordinate value on the
      corresponding dimension.
    :ivar mesh:
    :ivar n: Number of elements for each of the 3 dimensions.
    :ivar n_size: Total number of blocks in this mesh.
    :ivar label: Label associated with this mesh.
    :ivar obstructions: All obstructions inside of the mesh.
    :ivar vents: All vents inside of the mesh.
    """

    def __init__(self, x_coordinates: np.ndarray, y_coordinates: np.ndarray, z_coordinates: np.ndarray, mid: str,
                 smv_file: Union[str, mmap.mmap], pos: int, surfaces: List[Surface],
                 default_texture_origin: Tuple[float, float, float]):
        """
        :param x_coordinates: Coordinate values of x-axis.
        :param y_coordinates: Coordinate values of y-axis.
        :param z_coordinates: Coordinate values of z-axis.
        :param mid: ID of this mesh.
        :param smv_file: Either provide an initialized mmap object to read in additional data without having to open
         the smv_file again or the name of the smv-file containing the mesh information.
        :param pos: The current file pointer position so the correct obstacles and ventilations for this mesh are read.
        :param surfaces: List of surfaces available for obstacles and ventilations.
        """
        self.coordinates = [x_coordinates, y_coordinates, z_coordinates]
        self.extent = Extent(x_coordinates[0], x_coordinates[-1], y_coordinates[0],
                             y_coordinates[-1], z_coordinates[0], z_coordinates[-1])
        # Todo: Does this really do what it is supposed to do? What is it even supposed to do?
        # Todo: Numpy: Deprecated
        self.mesh = np.meshgrid(self.coordinates)

        self.n = [x_coordinates.size, y_coordinates.size, z_coordinates.size]
        self.n_size = self.n[0] * self.n[1] * self.n[2]

        self.id = mid

        if type(smv_file) == str:
            infile = open(smv_file, 'r')
            smv_file = mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ)

        self.obstructions = self._load_obstructions(smv_file, pos, surfaces, default_texture_origin)
        self.vents = self._load_vents(smv_file, pos)

        if type(smv_file) == str:
            smv_file.close()
            infile.close()

    def _load_obstructions(self, smv_file: mmap.mmap, pos: int, surfaces: List[Surface],
                           default_texture_origin: Tuple[float, float, float]) -> Dict[int, Obstruction]:
        obstructions = dict()
        pos = smv_file.find(b'OBST', pos)
        smv_file.seek(pos)
        smv_file.readline()
        n = int(smv_file.readline().decode().strip())

        temp_data = list()

        for _ in range(n):
            line = smv_file.readline().decode().strip().split()
            obst_id = int(line[6])
            extent = Extent(*[float(line[i]) for i in range(6)])
            side_surfaces = tuple(surfaces[int(line[i])] for i in range(7, 13))
            if len(line) > 13:
                texture_origin = (float(line[13]), float(line[14]), float(line[15]))
                temp_data.append((obst_id, extent, side_surfaces, texture_origin))
            else:
                temp_data.append((obst_id, extent, side_surfaces))

        for i in range(n):
            line = smv_file.readline().decode().strip().split()
            bound_indices = tuple(int(line[i]) for i in range(6))
            color_index = int(line[6])
            block_type = int(line[7])
            if color_index == -3:
                rgba = tuple(float(line[i]) for i in range(8, 12))
            else:
                rgba = ()

            if len(temp_data[i]) == 4:
                obst_id, extent, side_surfaces, texture_origin = temp_data[i]
            else:
                obst_id, extent, side_surfaces = temp_data[i]
                texture_origin = default_texture_origin

            obstructions[obst_id] = Obstruction(obst_id, extent, side_surfaces, bound_indices, color_index, block_type,
                                                texture_origin, rgba=rgba)

        return obstructions

    def _load_vents(self, smv_file: mmap.mmap, startpos: int, surfaces: List[Surface], mesh_id: int) -> List[
        Ventilation]:
        # Read information about opening and closing of vents
        def get_vents(oc):
            vents = dict()
            pos = smv_file.find(oc + b'_VENT', 0)
            while pos > 0:
                smv_file.seek(pos)
                pos = smv_file.find(b'OPEN_VENT', pos)

                target_mesh_id = int(smv_file.readline().decode().strip().split()[1])
                if target_mesh_id != mesh_id:
                    continue

                line = smv_file.readline().decode().strip()
                vents[int(line[0])] = float(line[1])
            return vents

        open_vents = get_vents(b'OPEN')
        close_vents = get_vents(b'CLOSE')

        ventilations = list()
        pos = smv_file.find(b'VENT', startpos)
        smv_file.seek(pos)
        smv_file.readline()
        line = smv_file.readline().decode().strip().split()
        n, n_dummies = int(line[0]), int(line[1])

        temp_data = list()

        def read_common_info():
            line = smv_file.readline().decode().strip().split()
            return Extent(*[float(line[i]) for i in range(6)]), int(line[6]), surfaces[int(line[7])]

        def read_common_info2():
            line = smv_file.readline().decode().strip().split()
            bound_indices = tuple(int(line[i]) for i in range(6))
            color_index = int(line[6])
            draw_type = int(line[7])
            if len(line) > 8:
                rgba = tuple(float(line[i]) for i in range(8, 12))
            else:
                rgba = ()
            return bound_indices, color_index, draw_type, rgba

        for _ in range(n - n_dummies):
            extent, vid, surface = read_common_info()
            texture_origin = (float(line[8]), float(line[9]), float(line[10]))
            temp_data.append((extent, vid, surface, texture_origin))

        for _ in range(n_dummies):
            extent, vid, surface = read_common_info()
            temp_data.append((extent, vid, surface))

        for i in range(n - n_dummies):
            extent, vid, surface, texture_origin = temp_data[i]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            ventilations.append(Ventilation(extent, vid, surface, bound_indices, color_index, draw_type, rgba=rgba,
                                            texture_origin=texture_origin, open_time=open_vents.get(vid, -1),
                                            close_time=close_vents.get(vid, -1)))

        for i in range(n_dummies):
            extent, vid, surface = temp_data[i]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            ventilations.append(Ventilation(extent, vid, surface, bound_indices, color_index, draw_type, rgba=rgba,
                                            open_time=open_vents.get(vid, -1), close_time=close_vents.get(vid, -1)))

        pos = smv_file.find(b'CVENT', pos)
        smv_file.seek(pos)
        smv_file.readline()

        n = int(smv_file.readline().decode().strip())

        temp_data.clear()

        for _ in range(n):
            extent, vid, surface = read_common_info()
            circular_vent_origin = (float(line[12]), float(line[13]), float(line[14]))
            radius = float(line[15])
            temp_data.append((extent, vid, surface, circular_vent_origin, radius))

        for i in range(n):
            extent, vid, surface, circular_vent_origin, radius = temp_data[i]
            bound_indices, color_index, draw_type, rgba = read_common_info2()
            ventilations.append(
                Ventilation(extent, vid, surface, bound_indices, color_index, draw_type, rgba=rgba,
                            circular_vent_origin=circular_vent_origin, radius=radius, open_time=open_vents.get(vid, -1),
                            close_time=close_vents.get(vid, -1)))

        return ventilations

    def __str__(self, *args, **kwargs):
        return f"{self.id}, {self.n[0]} x {self.n[1]} x {self.n[2]}, " + str(self.extent)
