import os
from pathlib import Path

import pytest

from fdsreader import settings


def pytest_configure(config):
    """Change CWD to tests/cases/ so acceptance tests can use relative paths like './steckler_data'.

    Also clears any *.pickle simulation caches left over in tests/cases/ from a previous local run.
    These are .gitignore'd (never present on a fresh CI checkout), but Simulation.__new__'s cache
    validity check only compares the fdsreader version and the smv-file hash - not e.g.
    settings.LAZY_LOAD - so a stale local cache from an earlier run (or an earlier test in the same
    run using different settings) can make a later test silently load pre-baked data instead of
    exercising the real parsing/loading code at all, masking real coverage and regressions.
    """
    cases_dir = Path(__file__).parent / "cases"
    if cases_dir.exists():
        os.chdir(cases_dir)
        for pickle_file in cases_dir.glob("*/*.pickle"):
            pickle_file.unlink()


@pytest.fixture(autouse=True)
def _restore_settings():
    """fdsreader.settings holds plain module-level globals that some code mutates as a side
    effect (e.g. explorer/cli.py's _load() sets settings.ENABLE_CACHING = caching). Without this,
    one test's mutation silently leaks into every test that runs afterward in the same process -
    e.g. test_caching_is_off_unless_asked_for leaves ENABLE_CACHING False, which then made an
    unrelated later Simulation never set self._hash.
    """
    original = vars(settings).copy()
    yield
    for key in list(vars(settings).keys()):
        if key not in original:
            delattr(settings, key)
    for key, value in original.items():
        setattr(settings, key, value)
