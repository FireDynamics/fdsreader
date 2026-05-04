import os
import tempfile
import numpy as np
import time
from typing import Callable

from fdsreader.utils import fortran_data as fdtype

def benchmark(name: str, func: Callable, args: tuple, iters: int = 5):
    print(f"Benchmarking {name}...", end=" ", flush=True)
    
    # Warmup
    func(*args)
    
    start = time.perf_counter()
    for _ in range(iters):
        func(*args)
    end = time.perf_counter()
    
    avg_time = (end - start) / iters
    print(f"Avg time: {avg_time:.6f}s")
    return avg_time

def verify_correctness(name: str, orig_func: Callable, opt_func: Callable, args: tuple):
    print(f"Verifying {name} correctness...", end=" ", flush=True)
    res_orig = orig_func(*args)
    res_opt = opt_func(*args)
    
    if isinstance(res_orig, dict) and isinstance(res_opt, dict):
        if set(res_orig.keys()) != set(res_opt.keys()):
            print("❌ Keys mismatch!")
            return False
        for k in res_orig:
            if not np.array_equal(res_orig[k], res_opt[k]):
                print(f"❌ Value mismatch for {k}!")
                return False
    elif not np.array_equal(res_orig, res_opt):
        print("❌ Value mismatch!")
        return False
    
    print("✅ OK")
    return True

# --- Optimization 1: Simulation._transform_csv_data ---
def sim_orig(keys, values):
    size = values.shape[0]
    data = {keys[i]: np.empty((size,), dtype=float) for i in range(len(keys))}
    for k, arr in enumerate(data.values()):
        for i in range(size):
            arr[i] = values[i][k]
    return data

def sim_opt(keys, values):
    return {key: values[:, i] for i, key in enumerate(keys)}

# --- Optimization 2: EvacCollection._load_csv_data ---
def evac_orig(names, values):
    dtypes = [int] * len(names)
    dtypes[0] = float
    dtypes[-2:] = (float, float)
    size = values.shape[0]
    data = {names[i]: np.empty((size,), dtype=dtypes[i]) for i in range(len(names))}
    for k, arr in enumerate(data.values()):
        for i in range(size):
            arr[i] = values[i][k]
    return data

def evac_opt(names, values):
    dtypes = [int] * len(names)
    dtypes[0] = float
    dtypes[-2:] = (float, float)
    return {names[i]: values[:, i].astype(dtypes[i]) for i in range(len(names))}

def xyz_orig(file_path, dtype_grid_data, n_i, n_j):
    with open(file_path, "rb") as infile:
        xyz = np.empty((n_i, n_j, 3))
        for i in range(n_i):
            for j in range(n_j):
                xyz[i, j] = fdtype.read(infile, dtype_grid_data, 1)[0][0]
    return xyz

def xyz_opt(file_path, dtype_grid_data, n_i, n_j):
    with open(file_path, "rb") as infile:
        grid_data = fdtype.read(infile, dtype_grid_data, n_i * n_j)
    return np.stack([row[0] for row in grid_data]).reshape((n_i, n_j, 3))

# --- Optimization 4: Simulation._load_DEVC_data (per-column copy) ---
def devc_orig(values, n_cols):
    out = {}
    for k in range(n_cols):
        size = values.shape[0]
        col = np.empty((size,), dtype=np.float32)
        for i in range(size):
            col[i] = values[i][k]
        out[k] = col
    return out

def devc_opt(values, n_cols):
    return {k: values[:, k].copy() for k in range(n_cols)}

def main():
    # CSV transforms: match production shape — genfromtxt yields float32,
    # EVAC CSVs have ~15 columns, not 50.
    n_rows, n_cols = 100000, 15
    keys = [f"k_{i}" for i in range(n_cols)]
    values = np.random.rand(n_rows, n_cols).astype(np.float32)

    results = []

    # Test Simulation transform
    if verify_correctness("Sim Transform", sim_orig, sim_opt, (keys, values)):
        t_orig = benchmark("Sim Transform (Orig)", sim_orig, (keys, values))
        t_opt = benchmark("Sim Transform (Opt)", sim_opt, (keys, values))
        results.append(("Sim Transform", t_orig, t_opt))

    # Test Evac transform
    if verify_correctness("Evac Transform", evac_orig, evac_opt, (keys, values)):
        t_orig = benchmark("Evac Transform (Orig)", evac_orig, (keys, values))
        t_opt = benchmark("Evac Transform (Opt)", evac_opt, (keys, values))
        results.append(("Evac Transform", t_orig, t_opt))

    # Evac XYZ grid load: write a real Fortran-record binary file so both
    # variants pay the full fdtype.read cost (np.fromfile requires a real fd).
    # The optimization eliminates per-cell reads, so the cost model must
    # include them.
    n_i, n_j = 100, 100
    dtype_grid_data = fdtype.new((("f", 3),))
    raw = np.empty(n_i * n_j, dtype=dtype_grid_data)
    for idx in range(n_i * n_j):
        raw[idx][1][:] = (idx * 0.1, idx * 0.2, idx * 0.3)
    fd, xyz_path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)
    try:
        raw.tofile(xyz_path)
        args = (xyz_path, dtype_grid_data, n_i, n_j)
        if verify_correctness("Evac XYZ", xyz_orig, xyz_opt, args):
            t_orig = benchmark("Evac XYZ (Orig)", xyz_orig, args)
            t_opt = benchmark("Evac XYZ (Opt)", xyz_opt, args)
            results.append(("Evac XYZ", t_orig, t_opt))
    finally:
        os.unlink(xyz_path)

    # DEVC load: per-device column copy from a genfromtxt float32 matrix.
    # Realistic shape: 100k timesteps, 20 devices.
    n_devc_rows, n_devc_cols = 100000, 20
    devc_values = np.random.rand(n_devc_rows, n_devc_cols).astype(np.float32)
    if verify_correctness("DEVC Load", devc_orig, devc_opt, (devc_values, n_devc_cols)):
        t_orig = benchmark("DEVC Load (Orig)", devc_orig, (devc_values, n_devc_cols))
        t_opt = benchmark("DEVC Load (Opt)", devc_opt, (devc_values, n_devc_cols))
        results.append(("DEVC Load", t_orig, t_opt))

    print("\nSummary:")
    print(f"{'Feature':<20} | {'Original':<12} | {'Optimized':<12} | {'Speedup':<10}")
    print("-" * 60)
    for name, t_orig, t_opt in results:
        speedup = t_orig / t_opt if t_opt > 0 else float('inf')
        print(f"{name:<20} | {t_orig:>10.6f}s | {t_opt:>10.6f}s | {speedup:>8.1f}x")

if __name__ == "__main__":
    main()
