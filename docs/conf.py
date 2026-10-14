# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "fdsreader"
copyright = f"2020–{datetime.date.today().year}, Forschungszentrum Jülich, IAS-7"
author = "Jan Vogelsang, Marc Fehr, Tristan Hehnen, Kristian Börger, Lukas Arnold"

# The full version, including alpha/beta/rc tags
try:
    release = get_version("fdsreader")
except PackageNotFoundError:
    release = "unknown"
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autodocsumm",
    "sphinxcontrib.mermaid",
    "myst_nb",
]


# The example notebooks live next to their FDS input in examples/<type>/ and are copied to
# docs/examples/<type>/ for the build. They are committed with their outputs and only rendered,
# because Read the Docs cannot run FDS to produce the data they read. Only changed notebooks are
# copied, so file-watching builders (sphinx-autobuild) do not rebuild in a loop.
def _sync_example_notebooks():
    docs_dir = Path(__file__).parent
    source_dir, target_dir = docs_dir.parent / "examples", docs_dir / "examples"
    wanted = set()
    for notebook in source_dir.glob("*/*.ipynb"):
        target = target_dir / notebook.relative_to(source_dir)
        wanted.add(target)
        if not target.exists() or target.read_bytes() != notebook.read_bytes():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(notebook, target)
    for stale in target_dir.rglob("*.ipynb"):
        if stale not in wanted:
            stale.unlink()


_sync_example_notebooks()
nb_execution_mode = "off"

# Let diagrams take their natural height instead of squeezing them into 500px.
mermaid_height = "auto"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# The FDS version the compatibility workflow last tested against (bumped automatically by
# .github/workflows/fds_compat.yml), available in the docs as |fds_tested|.
with open(os.path.join(os.path.dirname(__file__), "..", ".github", "last_tested_fds_version.txt")) as f:
    fds_tested = f.read().strip().removeprefix("FDS-")
rst_epilog = f".. |fds_tested| replace:: {fds_tested}\n"

# fdsreader.explorer's Jupyter front end needs matplotlib and ipywidgets, which are an
# optional extra and are not installed for a docs build.
autodoc_mock_imports = ["matplotlib", "ipywidgets", "ipympl", "IPython"]

autoclass_content = "both"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
