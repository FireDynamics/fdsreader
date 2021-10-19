import os
from pathlib import Path
import numpy as np
from typing_extensions import Literal
from ..bndf import Obstruction
from ..bndf.utils import sort_patches_cartesian


def export_obst_raw(obst: Obstruction, output_dir: str, ordering: Literal['C', 'F'] = 'C'):
    """Exports the 3d arrays to raw binary files with corresponding .yaml meta files.

    :param smoke3d: The :class:`Smoke3D` object to export.
    :param output_dir: The directory in which to save all files.
    :param ordering: Whether to write the data in C or Fortran ordering.
    """
    from pathos.pools import ProcessPool as Pool
    from multiprocess import Lock, Manager
    filename_base = "obst-" + str(obst.id)
    # Create all requested directories if they don't exist yet
    Path(os.path.join(output_dir, filename_base + "-data")).mkdir(parents=True, exist_ok=True)

    meta = {"BoundingBox": obst.bounding_box.as_list(), "QuantityNum": len(obst.quantities)}
    m = Manager()
    lock = m.Lock()
    meta["Quantities"] = m.list()

    def worker(quantity, bndf_data):
        quantity_name = quantity.name.replace(" ", "_").replace(".", "-")
        filename = filename_base + "_quantity-" + quantity_name + ".dat"

        out = {"Quantity": quantity.name.replace(" ", "_").replace(".", "-"), "DataValMax": -100000.,
               "DataValMin": 100000., "ScaleFactor": 1, "DataFile": os.path.join(filename_base + "-data", filename),
               "Orientations": list()}
        for bndf in bndf_data:
            out["DataValMax"] = max(out["DataValMax"], np.max(bndf.upper_bounds))
            out["DataValMin"] = min(out["DataValMin"], np.min(bndf.lower_bounds))
        out["DataValMax"] = float(out["DataValMax"])
        out["DataValMin"] = float(out["DataValMin"])
        out["ScaleFactor"] = 255.0 / out["DataValMax"]

        # Abort if no useful data is available
        if meta[quantity]["DataValMax"] <= 0:
            return

        with open(os.path.join(output_dir, quantity_name, filename_base + "-data", filename), 'wb') as rawfile:
            orientations = set()
            for orientation in (-3, -2, -1, 1, 2, 3):
                patches = list()
                for bndf in bndf_data:
                    if orientation in bndf.data:
                        orientations.add(orientation)
                        patches.append(bndf.data[orientation])

                if len(patches) == 0:
                    continue

                # Combine patches to a single face for plotting
                patches = sort_patches_cartesian(patches)

                shape_dim1 = sum([patch_row[0].shape[0] for patch_row in patches])
                shape_dim2 = sum([patch.shape[1] for patch in patches[0]])
                n_t = patches[0][0].n_t  # Number of timesteps

                face = np.empty(shape=(n_t, shape_dim1, shape_dim2))
                dim1_pos = 0
                dim2_pos = 0
                for patch_row in patches:
                    d1 = patch_row[0].shape[0]
                    for patch in patch_row:
                        d2 = patch.shape[1]
                        face[:, dim1_pos:dim1_pos + d1, dim2_pos:dim2_pos + d2] = patch.data
                        dim2_pos += d2
                    dim1_pos += d1
                    dim2_pos = 0

                face = (face * meta["ScaleFactor"]).astype(np.uint8)

                if abs(orientation) == 1:
                    spacing1 = (meta["BoundingBox"][3] - meta["BoundingBox"][2]) / face.shape[1]
                    spacing2 = (meta["BoundingBox"][5] - meta["BoundingBox"][4]) / face.shape[2]
                elif abs(orientation) == 2:
                    spacing1 = (meta["BoundingBox"][1] - meta["BoundingBox"][0]) / face.shape[0]
                    spacing2 = (meta["BoundingBox"][5] - meta["BoundingBox"][4]) / face.shape[2]
                else:
                    spacing1 = (meta["BoundingBox"][1] - meta["BoundingBox"][0]) / face.shape[0]
                    spacing2 = (meta["BoundingBox"][3] - meta["BoundingBox"][2]) / face.shape[1]


                out["Orientations"].append({
                    "Orientation": orientation,
                    "MeshPos": f"{meta['BoundingBox'][0]:.6} {meta['BoundingBox'][2]:.6} {meta['BoundingBox'][4]:.6}",
                    "Spacing": f"{bndf_data[0].times[1] - bndf_data[0].times[0]:.6} {spacing1:.6} {spacing2:.6}",
                    "DimSize": f"{face.shape[2]} {face.shape[1]} {face.shape[2]}"
                })

                for d in face[:, [2, 0, 1]]:  # Make time the first dimension so we can easily iterate over it
                    if ordering == 'F':
                        d = d.T
                    d.tofile(rawfile)

            out["NumOrientations"] = len(orientations)

        with lock:
            meta["Quantities"].append(out)

    worker_args = list()
    for i, quantity in enumerate(obst.quantities):
        bndf_data = obst.get_boundary_data(quantity)
        worker_args.append((quantity, bndf_data))
    Pool(8).map(lambda args: worker(*args), worker_args)

    meta["Quantities"] = list(meta["Quantities"])

    with open(os.path.join(output_dir, filename_base + ".yaml"), 'w') as metafile:
        import yaml
    yaml.dump(meta, metafile)
