# Contributing to fdsreader

Thank you for your interest in contributing! This guide explains how to set up
your development environment and what to expect from the contribution process.

## Development setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/FireDynamics/fdsreader.git
cd fdsreader

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 3. Set up pre-commit hooks (runs ruff automatically before each commit)
pre-commit install

# 4. Extract test data
cd tests/cases && for f in *.tgz; do tar -xzvf "$f"; done && cd ../..
```

## Running tests

```bash
# Convenience script — detects uv/python3, extracts test data automatically
bash tests/run_tests.sh

# Or manually:
pytest tests/

# With coverage report
pytest tests/ --cov=fdsreader --cov-report=term-missing
```

All 46 tests must pass before opening a pull request.

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for issues
ruff check fdsreader/

# Auto-fix issues
ruff check fdsreader/ --fix

# Format code
ruff format fdsreader/
```

The CI will fail if ruff reports any errors. If you have pre-commit installed,
ruff runs automatically on every commit.

## Regenerating test data

Test data archives (`.tgz`) are generated with specific FDS versions and stored
in `tests/cases/`. The corresponding FDS input files (`.fds`) are in
`tests/cases/fds_inputs/` and can be used to regenerate the data.

```bash
FDS=/path/to/fds_openmp
BASE=tests/cases

mkdir -p $BASE/steckler_data_fds<version>
cd $BASE/steckler_data_fds<version>
$FDS $BASE/fds_inputs/input_steckler.fds

# Repeat for other cases, then archive:
cd $BASE
tar -czf steckler_data_fds<version>.tgz steckler_data_fds<version>/
```

See the [FDS version compatibility table](README.md#fds-version-compatibility)
for which versions have been tested.

## Open issues and known bugs

Before starting work please check the
[issue tracker](https://github.com/FireDynamics/fdsreader/issues) for known bugs.

### Critical bugs (good first issues)

| File | Line | Bug |
|------|------|-----|
| `fdsreader/utils/extent.py` | 14 | `ValueError` is created but never raised → silent data corruption |
| `fdsreader/utils/misc.py` | 19 | `log_error` decorator returns `None` when an exception is caught |
| `fdsreader/utils/data.py` | 66 | `open()` without context manager → potential file handle leak |

### FDS 6.10.1 compatibility

Geometry data (`geom_data`) cannot be read from FDS 6.10.1 outputs because
the `BGEOM` block was removed from the SMV format. A fix requires updating
`simulation.py` to handle the new format.

## Pull request checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] No ruff errors (`ruff check fdsreader/`)
- [ ] New features include tests
- [ ] Commit messages follow the existing style (see `git log --oneline`)
