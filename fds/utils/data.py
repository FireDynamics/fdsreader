import os
import glob
import logging

def scan_directory_smv(directory: str):
    list_fn_smv_abs = glob.glob(directory + "/*.smv")

    if len(list_fn_smv_abs) == 0:
        return None

    if len(list_fn_smv_abs) > 1:
        logging.warning("multiple smv files found, choosing an arbitrary file")

    return os.path.basename(list_fn_smv_abs[0])
