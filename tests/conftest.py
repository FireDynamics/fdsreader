import os
from pathlib import Path


def pytest_configure(config):
    """Change CWD to tests/cases/ so acceptance tests can use relative paths like './steckler_data'."""
    cases_dir = Path(__file__).parent / "cases"
    if cases_dir.exists():
        os.chdir(cases_dir)
