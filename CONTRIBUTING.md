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

### Other known gaps (not urgent, no action planned unless there's a concrete need)

- `tests/cases/*_fds6100.tgz` (bndf/devc/part/pl3d/steckler) are unused by any test — leftover from an earlier FDS-6.10.1 compatibility check, never wired up or cleaned up.
- `GeomBoundary._load_gbf`/`_load_gcf` (`fdsreader/geom/geometry.py`) and `SubGeomSlice._load_geom_data` (`fdsreader/slcf/geomslice.py`) duplicate near-identical Fortran-record-reading logic; a shared helper would reduce the risk of the two drifting (as already nearly happened once).
- The cell-centered coordinate calculation (`coords[:-1] + np.diff(coords) / 2`) is duplicated across `fdsreader/fds_classes/mesh.py` (twice), `fdsreader/bndf/obstruction.py`, and `fdsreader/slcf/slice.py`; a shared `utils` helper would collapse these into one maintained implementation.
- `Particle.filter_by_tag` (`fdsreader/part/particle.py`) and `Evacuation.filter_by_tag` (`fdsreader/evac/evacuation.py`) are near-identical, independently-maintained implementations; a bug found in one needs to be fixed in both by hand.
- `GeomBoundary.surf_ind`/`.geom_ind` (`fdsreader/geom/geometry.py`) are only available for FDS 6.10+ (`.gcf`) simulations; HVAC, `&RADF`, and detailed zone-model output aren't read by fdsreader at all. `&CTRL` and mass/species-history CSV output are supported (`Simulation.ctrl`/`.mass`).
- `Mesh.get_obstruction_mask_slice` raises `NotImplementedError` for a genuine 3D slice (`orientation == 0`) — masking a 3D slice's obstruction cells would need a 3-axis index range this function doesn't compute yet, and no test fixture contains a 3D `&SLCF` to verify an implementation against.

## Pull request checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] No ruff errors (`ruff check fdsreader/`)
- [ ] New features include tests
- [ ] Commit messages follow the existing style (see `git log --oneline`)
