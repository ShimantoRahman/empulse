# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import json
import os
import sys
from datetime import datetime

import requests

sys.path.insert(0, os.path.abspath("../empulse"))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("sphinxext"))

print(sys.path)
import empulse  # noqa: E402 F401

project = 'Empulse'
copyright = f"2024 - {datetime.now().year}, Shimanto Rahman (AGPLv3 License)"
author = 'Shimanto Rahman'
release = empulse.__version__


# -- Generate versions.json ---------------------------------------------------

def get_latest_github_release(repo):
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["tag_name"]

def generate_versions_json():
    if 'dev' in release:
        # If the version is a dev version, then we need to get the latest release from GitHub
        latest_version = get_latest_github_release("ShimantoRahman/empulse")
        versions = [
            {
                "name": "dev (latest)",
                "version": release,
                "url": "https://empulse.readthedocs.io/en/latest/",
            },
            {
                "name": f"{latest_version} (stable)",
                "version": latest_version,
                "url": "https://empulse.readthedocs.io/en/stable/",
                "preferred": True
            }
        ]
    else:
        versions = [
            {
                "name": "dev (latest)",
                "version": release,
                "url": "https://empulse.readthedocs.io/en/latest/"
            },
            {
                "name": f"{release} (stable)",
                "version": release,
                "url": "https://empulse.readthedocs.io/en/stable/",
                "preferred": True
            }
        ]

    with open('./_static/versions.json', 'w') as f:
        json.dump(versions, f, indent=2)

generate_versions_json()

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
    "sphinxcontrib.sass",
    "sphinx_codeautolink",
    "sphinx_design",
    # see sphinxext folder for custom extensions
    "override_pst_pagetoc",
    "themed_figure",
]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# numpydoc 1.9.0 crashes in ``Validator.name`` on ``property`` objects (no ``__module__``),
# which autodoc hands it for members like ``Metric.tp_benefit``. Make the accessor defensive
# so ``numpydoc_validation_checks`` can be used at all. Remove once the upstream fix lands.
import numpydoc.validate as _npv  # noqa: E402


@property  # type: ignore[misc]
def _safe_validator_name(self):  # noqa: ANN001, ANN202
    obj = self.obj
    module = getattr(obj, '__module__', None) or getattr(type(obj), '__module__', '')
    qualname = getattr(obj, '__qualname__', None) or getattr(obj, '__name__', repr(obj))
    return f'{module}.{qualname}' if module else qualname


_npv.Validator.name = _safe_validator_name

# Docstring validation gate. Only the checks listed here are enforced (build fails under -W).
# The set is what `python scripts/check_docstrings.py` reports at zero failures; grow it as
# more docstrings are cleaned up. Deliberately excluded, with reasons:
#   SA01  - the package documents related objects with `.. seealso::` directives, not the
#           numpydoc "See Also" section (see CLAUDE.md).
#   RT02  - the package follows scikit-learn's `self : ClassName` / `name : type` Returns
#           style, which RT02 rejects; scikit-learn's own docstrings fail it too.
#   ES01, EX01 - not every method warrants an extended summary or a runnable example.
#   RT01, PR01 - the strategy classes' abstract `build`/`to_latex` and the prebuilt-metric
#           instances (documented via a `.. rubric:: Methods` block) do not carry these.
numpydoc_validation_checks = {
    'GL02', 'GL03', 'GL05', 'GL06', 'GL07', 'GL09', 'GL10',
    'PR02', 'PR03', 'PR04', 'PR05', 'PR06', 'PR07', 'PR08', 'PR09', 'PR10',
    'RT03', 'RT04', 'RT05',
    'SA02', 'SA03', 'SA04',
    'SS01', 'SS02', 'SS03', 'SS04', 'SS05', 'SS06',
    'YD01',
}
# Skip validation for names matching these patterns. scikit-learn's inherited methods render
# on every estimator page via `:inherited-members:`; `__call__` is a special method whose
# docstring is very often legitimately inherited.
numpydoc_validation_exclude = {
    r'\.(get_params|set_params|score|get_metadata_routing|get_feature_names_out|fit_transform|set_output)$',
    r'\.set_[a-z_]+_request$',
    r'\.__call__$',
    # `MixtureComponent` is a NamedTuple; `index`/`count` are inherited from `tuple` and carry
    # CPython's own (GL02-failing) docstrings, not ours.
    r'\.MixtureComponent\.(index|count)$',
}
autodoc_typehints = "none"
doctest_test_doctest_blocks = 'default'
myst_heading_anchors = 3

suppress_warnings = [
    'myst.header',  # Suppress warnings about document headings
]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/latest/", None),
    "lightgbm": ("https://lightgbm.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "imblearn": ("https://imbalanced-learn.org/stable", None),
    "sympy": ("https://docs.sympy.org/latest/", None)
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
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
    "switcher": {
        "json_url": "https://empulse.readthedocs.io/en/latest/_static/versions.json",
        "version_match": release,
    },
    # check_switcher may be set to False if docbuild pipeline fails. See
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html#configure-switcher-json-url
    "check_switcher": True,
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
    "secondary_sidebar_items": {
        "**": ["page-toc"],
    },
    "show_version_warning_banner": True,
    "announcement": None,
}
html_short_title = "empulse"
html_logo = "./_static/assets/empulse_logo_light.png"
html_favicon = "./_static/assets//favicon.ico"
html_static_path = ['_static']
# dynamically construct custom.css through sphinxcontrib.sass
html_css_files = ["css/custom.css"]
sass_src_dir = "_static/scss"
sass_out_dir = "_static/css"
sass_targets = {
    "custom.scss": "custom.css"
}
html_js_files = [
    "js/custom-icon.js",
]
