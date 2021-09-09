import yaml
import os
from pathlib import Path
import numpy as np
from typing_extensions import Literal
from ..slcf import Slice


def export_raw(slc: Slice, output_dir: str, ordering: Literal['C', 'F'] = 'C'):
    """Exports the 3d arrays to raw binary files with corresponding .yaml meta files.

    :param slc: The :class:`Slice` object to export.
    :param output_dir: The directory in which to save all files.
    :param ordering: Whether to write the data in C or Fortran ordering.
    """
    # Create all requested directories if they don't exist yet
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    meta = {"DataValMax": -100000.0, "DataValMin": 100000.0, "MeshNum": len(slc.subslices), "Meshes": list()}

    filename_base = ("slice-" + slc.quantity.name.lower()).replace(" ", "_").replace(".", "-")
    for mesh, subslice in slc._subslices.items():
        mesh_id = mesh.id.replace(" ", "_").replace(".", "-")
        filename = filename_base + "_mesh-" + mesh_id + ".dat"

        data_max = np.max(subslice.data)
        meta["DataValMax"] = 1000  # max(meta["DataValMax"], data_max)
        meta["DataValMin"] = 0  # min(meta["DataValMin"], np.min(subslice.data))
        data = (subslice.data * (255.0 / data_max)).astype(np.uint8)[:, :-1, :-1, :-1]

        with open(os.path.join(output_dir, filename), 'wb') as rawfile:
            for d in data:
                if ordering == 'F':
                    d = d.T
                d.tofile(rawfile)

        spacing = [slc.times[1] - slc.times[0],
                   mesh.coordinates['x'][1] - mesh.coordinates['x'][0],
                   mesh.coordinates['y'][1] - mesh.coordinates['y'][0],
                   mesh.coordinates['z'][1] - mesh.coordinates['z'][0]]
        meta["Meshes"].append({
            "Mesh": mesh_id,
            "DataFile": filename,
            "MeshPos": f"{mesh.coordinates['x'][0]:.6} {mesh.coordinates['y'][0]:.6} {mesh.coordinates['z'][0]:.6}",
            "Spacing": f"{spacing[0]:.6} {spacing[1]:.6} {spacing[2]:.6} {spacing[3]:.6}",
            "DimSize": f"{data.shape[0]} {data.shape[1]} {data.shape[2]} {data.shape[3]}"
        })

    meta["DataValMax"] = float(meta["DataValMax"])
    meta["DataValMin"] = float(meta["DataValMin"])

    with open(os.path.join(output_dir, filename_base + ".yaml"), 'w') as metafile:
        yaml.dump(meta, metafile)
