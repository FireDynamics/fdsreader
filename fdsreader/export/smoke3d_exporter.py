import os
from pathlib import Path
import numpy as np
from typing_extensions import Literal
from ..smoke3d import Smoke3D


def export_smoke_raw(smoke3d: Smoke3D, output_dir: str, ordering: Literal['C', 'F'] = 'C'):
    """Exports the 3d arrays to raw binary files with corresponding .yaml meta files.

    :param smoke3d: The :class:`Smoke3D` object to export.
    :param output_dir: The directory in which to save all files.
    :param ordering: Whether to write the data in C or Fortran ordering.
    """
    from pathos.pools import ProcessPool as Pool
    from multiprocess import Lock, Manager
    filename_base = ("smoke-" + smoke3d.quantity.name.lower()).replace(" ", "_").replace(".", "-")
    # Create all requested directories if they don't exist yet
    Path(os.path.join(output_dir, filename_base + "-data")).mkdir(parents=True, exist_ok=True)

    meta = {"DataValMax": -100000., "DataValMin": 100000., "ScaleFactor": 1, "MeshNum": len(smoke3d.subsmokes),
            "Quantity": smoke3d.quantity.name}

    for subsmoke in smoke3d._subsmokes.values():
        meta["DataValMax"] = max(meta["DataValMax"], np.max(subsmoke.data))
        meta["DataValMin"] = min(meta["DataValMin"], np.min(subsmoke.data))
    meta["DataValMax"] = float(meta["DataValMax"])
    meta["DataValMin"] = float(meta["DataValMin"])

    # Abort if no useful data is available
    if meta["DataValMax"] <= 0:
        return ""

    meta["ScaleFactor"] = 255.0 / meta["DataValMax"]

    m = Manager()
    meta["Meshes"] = m.list()
    lock = m.Lock()

    def worker(mesh, subsmoke):
        mesh_id = mesh.id.replace(" ", "_").replace(".", "-")
        filename = filename_base + "_mesh-" + mesh_id + ".dat"

        data = (subsmoke.data * meta["ScaleFactor"]).astype(np.uint8)

        with open(os.path.join(output_dir, filename_base + "-data", filename), 'wb') as rawfile:
            for d in data:
                if ordering == 'F':
                    d = d.T
                d.tofile(rawfile)

        spacing = [smoke3d.times[1] - smoke3d.times[0],
                   mesh.coordinates['x'][1] - mesh.coordinates['x'][0],
                   mesh.coordinates['y'][1] - mesh.coordinates['y'][0],
                   mesh.coordinates['z'][1] - mesh.coordinates['z'][0]]
        with lock:
            meta["Meshes"].append({
                "Mesh": mesh_id,
                "DataFile": os.path.join(filename_base + "-data", filename),
                "MeshPos": f"{mesh.coordinates['x'][0]:.6} {mesh.coordinates['y'][0]:.6} {mesh.coordinates['z'][0]:.6}",
                "Spacing": f"{spacing[0]:.6} {spacing[1]:.6} {spacing[2]:.6} {spacing[3]:.6}",
                "DimSize": f"{data.shape[0]} {data.shape[1]} {data.shape[2]} {data.shape[3]}"
            })

    with Pool() as pool:
        pool.map(lambda args: worker(*args), list(smoke3d._subsmokes.items()))

    meta["Meshes"] = list(meta["Meshes"])

    meta_file_path = os.path.join(output_dir, filename_base + ".yaml")
    with open(meta_file_path, 'w') as meta_file:
        import yaml
        yaml.dump(meta, meta_file)

    return meta_file_path
