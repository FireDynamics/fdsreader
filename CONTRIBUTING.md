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

All tests must pass before opening a pull request.

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

## Stale cache after pulling (editable installs)

`Simulation` caches a parsed simulation to a `.pickle` file next to the FDS
output and reuses it on the next load (`settings.ENABLE_CACHING`, on by
default). The cache is invalidated by comparing the installed package version
(`fdsreader.__version__`) against the one stored in the pickle — but for an
editable install (`pip install -e .`), that version is frozen at install time
and does **not** update automatically when you `git pull` or check out a
different commit. If you've changed anything that affects how a `Simulation`
or its data classes are structured, an old `.pickle` from before your change
can be loaded as if it were still valid and then fail later (e.g. an
`AttributeError`) once code touches the outdated structure.

After pulling changes into an editable install:

```bash
# Refresh the frozen version metadata
pip install -e . --force-reinstall --no-deps

# Or just delete any stale caches under the test data you're using
find tests/cases -name '*.pickle' -delete
```

## Open issues and known bugs

Before starting work please check the
[issue tracker](https://github.com/FireDynamics/fdsreader/issues) for known bugs.

### Critical bugs (good first issues)

| File | Line | Bug |
|------|------|-----|
| `fdsreader/utils/extent.py` | 14 | `ValueError` is created but never raised → silent data corruption |
| `fdsreader/utils/misc.py` | 19 | `log_error` decorator returns `None` when an exception is caught |
| `fdsreader/utils/data.py` | 66 | `open()` without context manager → potential file handle leak |
| `fdsreader/simulation.py` | 468 | `is_terrain = bool(texture_line[4])` — `texture_line[4]` is a `"0"`/`"1"` string from the SMV file; `bool("0")` is `True` in Python (any non-empty string is truthy), so `is_terrain` is always `True` regardless of the actual FDS value. Needs `bool(int(texture_line[4]))`. |
| `fdsreader/simulation.py` | 472 | `surface_id = line[0].split("%")[-1]` is never `.strip()`-ped, while `Surface.id()` is; the `s.id() == surface_id` match can silently fail on whitespace, leaving `geom.surface` as `None` even when a matching `SURF_ID` exists. |

### Other known gaps (not urgent, no action planned unless there's a concrete need)

- `tests/cases/*_fds6100.tgz` (bndf/devc/part/pl3d/steckler) are unused by any test — leftover from an earlier FDS-6.10.1 compatibility check, never wired up or cleaned up.
- `GeomBoundary._load_gbf`/`_load_gcf` (`fdsreader/geom/geometry.py`) and `SubGeomSlice._load_geom_data` (`fdsreader/slcf/geomslice.py`) duplicate near-identical Fortran-record-reading logic; a shared helper would reduce the risk of the two drifting (as already nearly happened once).
- `GeomBoundary.surf_ind`/`.geom_ind` (`fdsreader/geom/geometry.py`) are only available for FDS 6.10+ (`.gcf`) simulations; HVAC, `&RADF`, `&CTRL`, and detailed zone-model output aren't read by fdsreader at all.

## Pull request checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] No ruff errors (`ruff check fdsreader/`)
- [ ] New features include tests
- [ ] Commit messages follow the existing style (see `git log --oneline`)
