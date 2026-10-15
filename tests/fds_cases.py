"""Shared helpers for tests on stored FDS cases that also run against fresh FDS output.

The FDS compatibility and nightly workflows run tests/cases/fds_inputs/<name>.fds through a new FDS build and point
FDS_FRESH_OUTPUT_DIR at the output, so tests using these helpers check that build too.
"""

import os
from pathlib import Path

from fdsreader import Simulation, settings

CASES_DIR = Path(__file__).parent / "cases"


def case_dirs(stored: str, fresh: str) -> list[Path]:
    """The stored case, and the fresh output of tests/cases/fds_inputs/<fresh>.fds if the workflow provides it."""
    dirs = [CASES_DIR / stored]
    fresh_dir = Path(os.environ.get("FDS_FRESH_OUTPUT_DIR") or "/nonexistent") / fresh
    if fresh_dir.is_dir():
        dirs.append(fresh_dir)
    return dirs


def case_id(case_dir: Path) -> str:
    return "fresh" if not case_dir.is_relative_to(CASES_DIR) else case_dir.name.rsplit("_", 1)[-1]


def load(case_dir: Path) -> Simulation:
    """The simulation, parsed without the pickle cache."""
    original = settings.ENABLE_CACHING
    settings.ENABLE_CACHING = False
    try:
        return Simulation(str(case_dir))
    finally:
        settings.ENABLE_CACHING = original
