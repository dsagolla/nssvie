# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from datetime import date

needs_sphinx = "4.3"
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "nssvie"
author = "Daniel Sagolla"
copyright = f"{date.today().year}, " + author
version = "0.0.1"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_math_dollar",
    "sphinx_copybutton",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_design",
]

# Automatically generate autosummary after each build
autosummary_generate = True

# Added after including sphinx_math_dollar. The following prevents msthjax to
# parse $ and $$.
mathjax3_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

add_function_parentheses = False

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_favicon = "_static/icons/favicon.png"

html_theme_options = {
    "logo": {
        "image_light": "icons/logo-nssvie-light.svg",
        "image_dark": "icons/logo-nssvie-dark.svg",
    },
    "use_edit_page_button": False,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/dsagolla/nssvie/",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/nssvie",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/daniel_sagolla",
            "icon": "fa-brands fa-twitter",
            "type": "fontawesome",
        },
        {
            "name": "Mastodon",
            "url": "https://mstdn.social/@dsagolla",
            "icon": "fa-brands fa-mastodon",
            "type": "fontawesome",
        },
    ],
    "page_sidebar_items": ["page-toc"],
    "show_prev_next": False,
    "footer_items": ["sphinx-version", "copyright"],
}

html_sidebars = {
    "**": [
        "search-field",
        "sidebar-nav-bs.html",
        "sidebar-ethical-ads"]
}
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = [
    "nssvie.css",
]

html_title = f"{project} Manual"
html_last_updated_fmt = "%b %d, %Y"
html_context = {"default_mode": "light"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
