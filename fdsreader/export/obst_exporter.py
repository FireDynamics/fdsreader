import os
from pathlib import Path
from typing import Dict, Union, Tuple

import numpy as np
from typing_extensions import Literal
from ..bndf import Obstruction


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

    random_bndf = next(iter(obst.get_boundary_data(obst.quantities[0]).values()))
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

    def worker(quantity: str, faces: Dict[int, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]], vmin: float,
               vmax: float):
        quantity_name = quantity.replace(" ", "_").replace(".", "-")
        filename = obst_filename_base + "_quantity-" + quantity_name + ".dat"

        out = {
            "BoundaryQuantity": quantity.replace(" ", "_").replace(".", "-"),
            "DataFile": os.path.join(quantity_name, filename),
            "DataValMax": vmax,
            "DataValMin": vmin,
            "ScaleFactor": 255.0 / vmax
        }

        # Abort if no useful data is available
        if out["DataValMax"] <= 0:
            return

        with open(os.path.join(output_dir, quantity_name, filename), 'wb') as rawfile:
            for face in faces.values():  # face for each orientation
                for d in (face * out["ScaleFactor"]).astype(np.uint8):
                    if ordering == 'F':
                        d = d.T
                    d.tofile(rawfile)

        with lock:
            meta["Quantities"].append(out)

    worker_args = list()
    for i, bndf_quantity in enumerate(obst.quantities):
        worker_args.append((bndf_quantity.name, obst.get_global_boundary_data_arrays(bndf_quantity),
                            obst.vmin(bndf_quantity, 0), obst.vmax(bndf_quantity, 0)))
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
