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
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

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
    "autodocsumm",
    "sphinxcontrib.mermaid",
]

# Let diagrams take their natural height instead of squeezing them into 500px.
mermaid_height = "auto"

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
