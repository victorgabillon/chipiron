"""Module for conf."""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys

# Import the chess module.
sys.path.insert(0, os.path.abspath("../chipiron/"))
sys.path.insert(0, os.path.abspath(".."))


project = "chipiron"
copyright = "2024, Victor Gabillon"
author = "Victor Gabillon"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "app.py",
    "setup.py",
    "flaskapp.py",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

strip_signature_backslash = True


napoleon_use_ivar = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_attr_annotations = (
    False  # 👈 This disables duplicate field docs from class docstring
)
