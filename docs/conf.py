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
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "fdsreader"
copyright = "2020, FZJ IAS-7 (Prof. Dr. Lukas Arnold, Jan Vogelsang)"
author = "FZJ IAS-7 (Prof. Dr. Lukas Arnold, Jan Vogelsang)"

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
extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "autodocsumm"]

autoclass_content = "both"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
