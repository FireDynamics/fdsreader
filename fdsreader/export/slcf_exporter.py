import os
from pathlib import Path
import numpy as np
from typing_extensions import Literal
from ..slcf import Slice


def export_slcf_raw(slc: Slice, output_dir: str, ordering: Literal['C', 'F'] = 'C'):
    """Exports the 3d arrays to raw binary files with corresponding .yaml meta files.

    :param slc: The :class:`Slice` object to export.
    :param output_dir: The directory in which to save all files.
    :param ordering: Whether to write the data in C or Fortran ordering.
    """
    from pathos.pools import ProcessPool as Pool
    from multiprocess import Lock, Manager
    slc2d = slc.type == '2D'
    meta = {"DataValMax": float(slc.vmax), "DataValMin": float(slc.vmin), "ScaleFactor": 255. / (float(slc.vmax) - float(slc.vmin)),
            "MeshNum": len(slc.subslices), "Quantity": slc.quantity.name}

    filename_base = ("slice" + ("2D-" if slc2d else "3D-") + slc.id.lower()).replace(" ", "_").replace(".", "-")
    # Create all requested directories if they don't exist yet
    Path(os.path.join(output_dir, filename_base + "-data")).mkdir(parents=True, exist_ok=True)

    m = Manager()
    meta["Meshes"] = m.list()
    lock = m.Lock()

    def worker(mesh, subslice):
        mesh_id = mesh.id.replace(" ", "_").replace(".", "-")
        filename = filename_base + "_mesh-" + mesh_id + ".dat"

        data = ((subslice.data - meta["DataValMin"]) * meta["ScaleFactor"]).astype(np.uint8)
        shape = data.shape
        if slc2d:
            shape = shape[:subslice.orientation] + (1,) + shape[subslice.orientation:]

        with open(os.path.join(output_dir, filename_base + "-data", filename), 'wb') as rawfile:
            for d in data:
                if ordering == 'F':
                    d = d.T
                d.tofile(rawfile)

        spacing = [slc.times[1] - slc.times[0],
                   mesh.coordinates['x'][1] - mesh.coordinates['x'][0],
                   mesh.coordinates['y'][1] - mesh.coordinates['y'][0],
                   mesh.coordinates['z'][1] - mesh.coordinates['z'][0]]
        with lock:
            meta["Meshes"].append({
                "Mesh": mesh_id,
                "DataFile": os.path.join(filename_base + "-data", filename),
                "MeshPos": f"{subslice.extent['x'][0]:.6} {subslice.extent['y'][0]:.6} {subslice.extent['z'][0]:.6}",
                "Spacing": f"{spacing[0]:.6} {spacing[1]:.6} {spacing[2]:.6} {spacing[3]:.6}",
                "DimSize": f"{shape[0]} {shape[1]} {shape[2]} {shape[3]}"
            })

    with Pool() as pool:
        pool.map(lambda args: worker(*args), list(slc._subslices.items()))

    meta["Meshes"] = list(meta["Meshes"])

    meta_file_path = os.path.join(output_dir, filename_base + ".yaml")
    with open(meta_file_path, 'w') as meta_file:
        import yaml
        yaml.dump(meta, meta_file)

    return meta_file_path
