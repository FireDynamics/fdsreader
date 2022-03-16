import os
from pathlib import Path
from typing import Sequence

import numpy as np
from typing_extensions import Literal
from ..bndf import Obstruction, Boundary
from ..bndf.utils import sort_patches_cartesian


def export_obst_raw(obst: Obstruction, output_dir: str, ordering: Literal['C', 'F'] = 'C'):
    """Exports the 3d arrays to raw binary files with corresponding .yaml meta files.

    :param obst: The :class:`Obstruction` object to export including its :class:`Boundary` data.
    :param output_dir: The directory in which to save all files.
    :param ordering: Whether to write the data in C or Fortran ordering.
    """
    if len(obst.quantities) == 0:
        return ""

    from pathos.pools import ProcessPool as Pool
    from multiprocess import Lock, Manager
    obst_filename_base = "obst-" + str(obst.id)

    bounding_box = obst.bounding_box.as_list()
    meta = {"BoundingBox": " ".join(f"{b:.6f}" for b in bounding_box), "NumQuantities": len(obst.quantities),
            "Orientations": list()}
    m = Manager()
    lock = m.Lock()
    meta["Quantities"] = m.list()

    random_bndf = obst.get_boundary_data(obst.quantities[0])[0]
    meta["TimeSteps"] = len(random_bndf.times)
    meta["NumOrientations"] = len(random_bndf.data)
    for orientation, face in random_bndf.data.items():
        if abs(orientation) == 1:
            spacing1 = (bounding_box[3] - bounding_box[2]) / face.shape[0]
            spacing2 = (bounding_box[5] - bounding_box[4]) / face.shape[1]
        elif abs(orientation) == 2:
            spacing1 = (bounding_box[1] - bounding_box[0]) / face.shape[0]
            spacing2 = (bounding_box[5] - bounding_box[4]) / face.shape[1]
        else:
            spacing1 = (bounding_box[1] - bounding_box[0]) / face.shape[0]
            spacing2 = (bounding_box[3] - bounding_box[2]) / face.shape[1]

        meta["Orientations"].append({
            "BoundaryOrientation": orientation,
            # "MeshPos": f"{meta['BoundingBox'][0]:.6f} {meta['BoundingBox'][2]:.6f} {meta['BoundingBox'][4]:.6f}",
            "Spacing": f"{random_bndf.times[1] - random_bndf.times[0]:.6f} {spacing1:.6f} {spacing2:.6f}",
            "DimSize": f"{face.shape[0]} {face.shape[1]}"
        })

    def worker(quantity: str, bndf_data: Sequence[Boundary]):
        quantity_name = quantity.replace(" ", "_").replace(".", "-")
        filename = obst_filename_base + "_quantity-" + quantity_name + ".dat"

        out = {"BoundaryQuantity": quantity.replace(" ", "_").replace(".", "-"), "DataValMax": -100000.,
               "DataValMin": 100000., "ScaleFactor": 1, "DataFile": os.path.join(quantity_name, filename)}
        for bndf in bndf_data:
            out["DataValMax"] = max(out.get("DataValMax"), np.max(bndf.upper_bounds))
            out["DataValMin"] = min(out.get("DataValMin"), np.min(bndf.lower_bounds))
        out["DataValMax"] = float(out.get("DataValMax"))
        out["DataValMin"] = float(out.get("DataValMin"))
        out["ScaleFactor"] = 255.0 / out.get("DataValMax")

        # Abort if no useful data is available
        if out["DataValMax"] <= 0:
            return

        with open(os.path.join(output_dir, quantity_name, filename), 'wb') as rawfile:
            for orientation in (-3, -2, -1, 1, 2, 3):
                patches = list()
                for bndf in bndf_data:
                    if orientation in bndf.data:
                        patches.append(bndf.data[orientation])

                if len(patches) == 0:
                    continue

                # Combine patches to a single face for plotting
                patches = sort_patches_cartesian(patches)

                shape_dim1 = sum([patch_row[0].shape[0] for patch_row in patches])
                shape_dim2 = sum([patch.shape[1] for patch in patches[0]])
                n_t = patches[0][0].n_t  # Number of time steps

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

                face = (face * out["ScaleFactor"]).astype(np.uint8)

                for d in face:
                    if ordering == 'F':
                        d = d.T
                    d.tofile(rawfile)

        with lock:
            meta["Quantities"].append(out)

    worker_args = list()
    for i, bndf_quantity in enumerate(obst.quantities):
        worker_args.append((bndf_quantity.name, obst.get_boundary_data(bndf_quantity)))
        # Create all requested directories if they don't exist yet
        Path(os.path.join(output_dir, bndf_quantity.name.replace(" ", "_").replace(".", "-"))).mkdir(parents=True,
                                                                                                     exist_ok=True)
    with Pool(len(obst.quantities)) as pool:
        pool.map(lambda args: worker(*args), worker_args)

    meta["Quantities"] = list(meta["Quantities"])

    meta_file_path = os.path.join(output_dir, obst_filename_base + ".yaml")
    with open(meta_file_path, 'w') as meta_file:
        import yaml
        yaml.dump(meta, meta_file)

    return meta_file_path
