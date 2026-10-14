"""Run the FDS inputs of the example notebooks and execute the notebooks again.

The notebooks are committed with their output, which the documentation shows as is. Run this
script before a release, with a current FDS, to check that the examples still work and to
refresh the stored output:

    python examples/run_examples.py              # all example folders with a notebook
    python examples/run_examples.py slcf         # only examples/slcf
    python examples/run_examples.py --skip-fds   # reuse existing FDS output

``fds`` (and ``mpiexec`` or ``mpirun`` for inputs with several meshes) must be on the PATH,
e.g. after sourcing FDS6VARS.sh. Executing the notebooks needs nbclient, ipykernel and the
plotting packages the notebooks use (matplotlib, pyvista).
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent


def _value(params: str, name: str, default: int) -> int:
    match = re.search(rf"\b{name}\s*=\s*(-?\d+)", params, flags=re.IGNORECASE)
    return int(match.group(1)) if match else default


def count_meshes(fds_file: Path) -> int:
    """Number of meshes, counting meshes multiplied with &MULT; FDS needs one MPI process each."""
    namelists = re.findall(r"^\s*&(\w+)(.*?)/", fds_file.read_text(errors="replace"), flags=re.MULTILINE | re.DOTALL)
    copies = {}
    for name, params in namelists:
        if name.upper() == "MULT":
            mult_id = re.search(r"\bID\s*=\s*'([^']*)'", params, flags=re.IGNORECASE).group(1)
            if re.search(r"\bN_UPPER\b", params, flags=re.IGNORECASE):
                copies[mult_id] = _value(params, "N_UPPER", 0) - _value(params, "N_LOWER", 0) + 1
            else:
                copies[mult_id] = 1
                for axis in "IJK":
                    copies[mult_id] *= _value(params, f"{axis}_UPPER", 0) - _value(params, f"{axis}_LOWER", 0) + 1
    n_meshes = 0
    for name, params in namelists:
        if name.upper() == "MESH":
            mult_id = re.search(r"\bMULT_ID\s*=\s*'([^']*)'", params, flags=re.IGNORECASE)
            n_meshes += copies.get(mult_id.group(1), 1) if mult_id else 1
    return n_meshes


def find_mpi_launcher() -> str | None:
    """The MPI launcher bundled with the fds on the PATH, else any on the PATH.

    With several FDS versions installed, the PATH can hold the fds of one and the mpiexec of
    another, and an fds only runs with the MPI it was built against.
    """
    fds_bin = Path(shutil.which("fds")).resolve().parent
    for launcher in ("mpiexec", "mpirun"):
        for bundled in sorted(fds_bin.glob(f"*mpi*/bin/{launcher}")):
            return str(bundled)
    return shutil.which("mpiexec") or shutil.which("mpirun")


def run_fds(fds_file: Path):
    n_meshes = count_meshes(fds_file)
    command = ["fds", fds_file.name]
    if n_meshes > 1:
        mpi = find_mpi_launcher()
        if mpi is None:
            sys.exit(f"{fds_file} has {n_meshes} meshes, but neither mpiexec nor mpirun is on the PATH")
        # Open MPI refuses more processes than cores unless told otherwise. Its version output
        # says "Open MPI" (5.x) or "OpenRTE ... open-mpi.org" (4.x).
        version = subprocess.run([mpi, "--version"], capture_output=True, text=True)
        is_open_mpi = re.search(r"open[ -]?mpi|openrte", version.stdout + version.stderr, flags=re.IGNORECASE)
        oversubscribe = ["--oversubscribe"] if is_open_mpi else []
        command = [mpi, *oversubscribe, "-n", str(n_meshes), *command]
    print(f"FDS  {fds_file.relative_to(EXAMPLES)} ({n_meshes} mesh{'es' if n_meshes > 1 else ''})", flush=True)
    with open(fds_file.with_suffix(".log"), "w") as log:
        subprocess.run(command, cwd=fds_file.parent, stdout=log, stderr=subprocess.STDOUT)
    # The exit code is not reliable: on macOS the bundled MPI can crash while shutting down
    # after FDS finished, so check FDS's own completion message instead.
    if "STOP: FDS completed successfully" not in fds_file.with_suffix(".log").read_text(errors="replace"):
        sys.exit(f"FDS failed for {fds_file}, see {fds_file.with_suffix('.log')}")


def execute_notebook(notebook: Path):
    """Execute the notebook with this Python's kernel and save it with the new output."""
    import nbformat
    from nbclient import NotebookClient

    print(f"run  {notebook.relative_to(EXAMPLES)}", flush=True)
    nb = nbformat.read(notebook, as_version=4)
    NotebookClient(nb, kernel_name="python3", resources={"metadata": {"path": str(notebook.parent)}}).execute()
    nbformat.write(nb, notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="*", help="example folders, e.g. slcf (default: all with a notebook)")
    parser.add_argument("--skip-fds", action="store_true", help="do not run FDS, reuse the existing output")
    args = parser.parse_args()

    folders = [EXAMPLES / f for f in args.folders] or sorted({nb.parent for nb in EXAMPLES.glob("*/*.ipynb")})
    if not args.skip_fds and shutil.which("fds") is None:
        sys.exit("fds is not on the PATH (source FDS6VARS.sh first, or use --skip-fds)")

    for folder in folders:
        if not args.skip_fds:
            for fds_file in sorted(folder.glob("*/*.fds")):
                run_fds(fds_file)
        for notebook in sorted(folder.glob("*.ipynb")):
            execute_notebook(notebook)


if __name__ == "__main__":
    main()
