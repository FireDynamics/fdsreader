import os
from typing import List

from utils import Extent
import utils.fortran_data as fdtype

class Plot3D:
    def __init__(self, root_path: str, cell_centered: bool, filename: str, quantities: List[str], label: List[str],
                 unit: List[str], extent: Extent, mesh_id: int):

        self.file_path = os.path.join(root_path, filename)
        with open(self.file_path, 'rb') as infile:
            self.offset = fdtype.new((('i', 3),)).itemsize + fdtype.new((('i', 4),)).itemsize


class _SubPlot3D:
    def __init__(self):
        pass