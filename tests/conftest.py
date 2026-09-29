import os
from pathlib import Path

import pytest

from fdsreader import settings


def pytest_configure(config):
    """Change CWD to tests/cases/ so acceptance tests can use relative paths like './steckler_data'."""
    cases_dir = Path(__file__).parent / "cases"
    if cases_dir.exists():
        os.chdir(cases_dir)


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
