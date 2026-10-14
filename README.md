# fdsreader

**Fast and easy-to-use Python reader for [Fire Dynamics Simulator (FDS)](https://pages.nist.gov/fds-smv/) output.**

[![PyPI](https://img.shields.io/pypi/v/fdsreader)](https://pypi.org/project/fdsreader/)
[![Python](https://img.shields.io/pypi/pyversions/fdsreader)](https://pypi.org/project/fdsreader/)
[![CI](https://github.com/FireDynamics/fdsreader/actions/workflows/ci.yml/badge.svg)](https://github.com/FireDynamics/fdsreader/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/FireDynamics/fdsreader/branch/master/graph/badge.svg)](https://codecov.io/gh/FireDynamics/fdsreader)
[![Docs](https://readthedocs.org/projects/fdsreader/badge/?version=latest)](https://fdsreader.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6783225.svg)](https://doi.org/10.5281/zenodo.6783225)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPLv3-blue)](https://github.com/FireDynamics/fdsreader/blob/master/LICENSE.LESSER)

fdsreader reads the binary and CSV files an FDS simulation writes and hands you the data
as NumPy arrays. Point it at a simulation directory and every output is available as a
Python object, loaded only when you first access it. Data from multi-mesh simulations can
be merged into one global array. For a quick look without writing any plotting code,
there is an [explorer](https://fdsreader.readthedocs.io/en/latest/guide/explorer.html) for Jupyter and the terminal.

## Installation

```sh
pip install fdsreader
```

The Jupyter [explorer](https://fdsreader.readthedocs.io/en/latest/guide/explorer.html) needs a plotting stack, which is an optional extra:
`pip install "fdsreader[notebook]"`. Its command line counterpart needs nothing beyond
numpy and is always installed.

fdsreader needs Python 3.10 or newer and reads output of FDS 6.7.5 and later; it is tested
against the latest FDS release. For complex geometry (`&GEOM`) from FDS 6.10 or newer, use fdsreader 1.12.1 or
later. Details are in the [compatibility table](https://fdsreader.readthedocs.io/en/latest/guide/compatibility.html);
if something does not read correctly, please [open an issue](https://github.com/FireDynamics/fdsreader/issues).

## Quickstart

```python
import fdsreader as fds

sim = fds.Simulation("./my_simulation")  # the directory containing the .smv file
print(sim)                               # what the simulation wrote

# Temperature slice closest to x = 0, merged over all meshes into one array
slc = sim.slices.filter_by_quantity("TEMPERATURE").get_nearest(x=0)
temperature = slc.to_global()            # shape (time, ...)
step = slc.get_nearest_timestep(30.0)
print(slc.times[step], temperature[step].max())

# Time series: devices by their ID from the input file, and the heat release rate
time = sim.devices["Time"].data
hgl_temp = sim.devices["HGL Temp"].data
hrr = sim.hrr["HRR"]
```

More examples for every data type are in [`examples/`](https://github.com/FireDynamics/fdsreader/tree/master/examples), the full API is in the
[documentation](https://fdsreader.readthedocs.io).

## Explorer

`fdsreader.explorer` looks at a simulation without writing plotting code for it: a time
bar driving a 2D slice, and any number of device or HRR quantities plotted against time.
In Jupyter that is an interactive widget; in a terminal you can either explore it
interactively or draw a single view and exit.

### In Jupyter

```python
from fdsreader.explorer import explore

explorer = explore()
```

That renders the whole interface — a directory browser, the time bar, the slice on the
left and the curves on the right. Browse to a case and press *Load simulation*.

```python
explore("./sample_data")            # skip the browser and load this case
explore(start_dir="/data/cases")    # open the browser somewhere specific
explore(interactive_canvas=False)   # static images instead of ipympl canvases
explore(caching=True)               # allow the .pickle cache (writable data only)
```

`explore()` returns the explorer, so the simulation and the current selection stay
available in later cells:

```python
sim = explorer.sim
explorer.current_time               # where the time bar is, in seconds
explorer.selection                  # the curves currently ticked
explorer.field.frame(400)           # the slice on screen, at time step 400
```

`ipympl` is optional; with it each plot gains a toolbar to zoom, pan and save. If the
plots come up as `Failed to load model class 'MPLCanvasModel'`, its browser-side
extension is not loading — pass `interactive_canvas=False`.

### In a terminal

The command line front end needs nothing but numpy, so it works over SSH on a machine
with no plotting stack and no browser.

`-i` opens a full-screen interactive view, where the slice and the curves are chosen from
lists rather than named on the command line:

```sh
fdsreader-explorer-cli ./sample_data -i
```

| Key | Action |
|---|---|
| `←` `→` or `h` `l` | step through time |
| `page up` / `page down` | ten steps |
| `home` / `end` | first / last step |
| `space` | play, pause |
| `+` `-` | play speed |
| `f` | pick the slice from a list |
| `c` | pick the curves from a list (`space` ticks, `enter` accepts) |
| `[` `]` | previous / next slice |
| `g` / `s` | colour scale over the whole run / this step |
| `?` | help |
| `q` | quit |

Without `-i` it draws one view and exits, which is what you want in a script:

```sh
fdsreader-explorer-cli ./sample_data                        # what is in it
fdsreader-explorer-cli ./sample_data --list                 # slices and curves by name
fdsreader-explorer-cli ./sample_data --slice 0 --time 4.4   # draw a slice
fdsreader-explorer-cli ./sample_data --curve TC_1 --curve HRR
fdsreader-explorer-cli ./sample_data --json                 # for scripting
```

```
TEMPERATURE [C] — x = 0 m  #0   ·   t = 4.401 s (step 630 of 749)

  10.00 |
        |
   8.82 |
        |
        |
        |          ...
        |        .....
   5.88 |       .....
        |      ...:...
        |       .....
        |        ....
        |        ...
   2.94 |       .:..
        |       -.
        |       -:-.
        |       .-*:
        |        +%+.
   0.00 |        +#:
        +------------------
         -2.5           2.5   y [m]
         scale 20 … 1412   ramp ' .:-=+*#%@'   grid 18x18
```

`--save FILE` writes the current view as an image or an animation instead
(`.png`/`.pdf`/`.svg`/`.gif`/`.mp4`, needs matplotlib).

### Notes

- The interactive terminal view uses `curses`, which is in the standard library everywhere except
  Windows; there, `pip install windows-curses` provides it.
- Slices are read one time step at a time, so stepping through a long run costs a few
  hundred kilobytes rather than loading the whole series into memory.

## Documentation

The [documentation](https://fdsreader.readthedocs.io) covers every data type, the
[settings](https://fdsreader.readthedocs.io/en/latest/guide/configuration.html) (lazy loading,
caching, error handling) and the full API.

## Citing

If fdsreader helps your research, please cite it via its Zenodo DOI
[10.5281/zenodo.6783225](https://doi.org/10.5281/zenodo.6783225). GitHub's
"Cite this repository" button gives you the reference in BibTeX or APA.

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/FireDynamics/fdsreader/blob/master/CONTRIBUTING.md)
for the development setup and how to run the tests.

## Authors and maintainers

fdsreader is developed at [Forschungszentrum Jülich, IAS-7](https://www.fz-juelich.de/en/ias/ias-7)
and maintained by Jan Vogelsang, Marc Fehr, Tristan Hehnen and Lukas Arnold.
For all authors see [CITATION.cff](https://github.com/FireDynamics/fdsreader/blob/master/CITATION.cff).

fdsreader is licensed under the [LGPLv3](https://github.com/FireDynamics/fdsreader/blob/master/LICENSE.LESSER).
