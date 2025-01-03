# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../empulse"))
sys.path.insert(0, os.path.abspath(".."))

print(sys.path)
import empulse  # noqa: E402 F401

project = 'Empulse'
copyright = f"2024 - {datetime.now().year}, Shimanto Rahman (MIT License)"
author = 'Shimanto Rahman'
release = empulse.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "numpydoc",
    "sphinx_copybutton",
]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
autodoc_typehints = "none"
doctest_test_doctest_blocks = 'default'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}
intersphinx_disabled_reftypes = ["*"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# html_theme = 'sphinx_rtd_theme'
# html_theme_options = {"logo_only": True}
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    # -- General configuration ------------------------------------------------
    "sidebar_includehidden": True,
    # "use_edit_page_button": True,
    "external_links": [],
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ShimantoRahman/empulse",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/empulse/",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome",
        },
    ],

    # If "prev-next" is included in article_footer_items, then setting show_prev_next
    # to True would repeat prev and next links. See
    # https://github.com/pydata/pydata-sphinx-theme/blob/b731dc230bc26a3d1d1bb039c56c977a9b3d25d8/src/pydata_sphinx_theme/theme/pydata_sphinx_theme/layout.html#L118-L129
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 1,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "header_dropdown_text": "More",

    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "logo": {
        "alt_text": "empulse homepage",
        "image_relative": "_static/assets/empulse_logo_light.png",
        "image_light": "_static/assets/empulse_logo_light.png",
        "image_dark": "_static/assets/empulse_logo_dark.png",
    },
    "surface_warnings": True,
    # -- Template placement in theme layouts ----------------------------------
    "navbar_start": ["navbar-logo"],
    # Note that the alignment of navbar_center is controlled by navbar_align
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    # navbar_persistent is persistent right (even when on mobiles)
    "navbar_persistent": ["search-button"],
    "article_header_start": ["breadcrumbs"],
    "article_header_end": [],
    "article_footer_items": ["prev-next"],
    "content_footer_items": [],
    # Use html_sidebars that map page patterns to list of sidebar templates
    "primary_sidebar_end": [],
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
    # When specified as a dictionary, the keys should follow glob-style patterns, as in
    # https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
    # In particular, "**" specifies the default for all pages
    # Use :html_theme.sidebar_secondary.remove: for file-wide removal
    # "secondary_sidebar_items": {
    #     "**": [
    #         "page-toc",
    #         "sourcelink",
    #         # Sphinx-Gallery-specific sidebar components
    #         # https://sphinx-gallery.github.io/stable/advanced.html#using-sphinx-gallery-sidebar-components
    #         "sg_download_links",
    #         "sg_launcher_links",
    #     ],
    # },
    "show_version_warning_banner": True,
    "announcement": None,
}
html_short_title = "empulse"
html_logo = "./_static/assets/empulse_logo_light.png"
html_static_path = ['_static']
html_css_files = ["css/custom.css"]
html_js_files = [
    "js/custom-icon.js",
]
