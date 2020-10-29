"""
.sio (isog + tisog)
Header: 3 int + (iso levels)*real + 3 int
.viso (tisog)
Header: isog + 2 int


.iso (isog + tisog)
Time: real + int + int(nverts) + int(ntris)
Data: (3*nverts) real + (3*ntris) int + (1*ntris) int

.viso (tisog)
Time: real + 2 int + int (nverts) + int
Data: (1*nverts) real
"""


class Isosurface:
    def __init__(self, root_path: str, double_quantity: bool, iso_filename: str, quantity: str,
                 label: str, unit: str, viso_filename: str = "", v_quantity: str = "",
                 v_label: str = "", v_unit: str = ""):
        double_quantity = double_quantity
        n_vertices = 0
        n_triangles = 0
