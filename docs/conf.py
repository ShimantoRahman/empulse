# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../empulse"))
sys.path.insert(0, os.path.abspath(".."))

print(sys.path)
import empulse  # noqa: E402 F401

project = 'Empulse'
copyright = '2024, Shimanto Rahman'
author = 'Shimanto Rahman'
release = '0.0.13'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
#    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

autodoc_typehints = "none"
#napoleon_google_docstring = False
#napoleon_numpy_docstring = True
#napoleon_use_param = False
#napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']