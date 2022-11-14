# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from datetime import date

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
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_math_dollar",
    "sphinx_toggleprompt",
    "numpydoc",
    "sphinx-prompt",
    "sphinx.ext.autosummary"
]

# Copy button settings
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r'>>> |\.\.\. '

# Automatically generate autosummary after each build
autosummary_generate = True
# autosummary_imported_members = True

# Added after including sphinx_math_dollar. The following prevents msthjax to
# parse $ and $$.
mathjax3_config = {
    'tex2jax': {
        'inlineMath': [["\\(", "\\)"]],
        'displayMath': [["\\[", "\\]"]],
    },
}


exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]

# rst_epilog = """
# .. |psf| replace:: Python Software Foundation
# """
# rst_epilog = """
# .. |psf| replace:: Python Software Foundation
# """
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_favicon = "_static/icons/favicon.png"

html_theme_options = {
    "navbar_end": [
        "theme-switcher",
        "search-field.html",
        "navbar-icon-links.html",
    ],
    "page_sidebar_items": ["page-toc"],
    # "header_links_before_dropdown": 4,
    "use_edit_page_button": False,
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
    "pygment_light_style": "tango",
    "pygment_dark_style": "native",
    "logo": {
        "image_light": "icons/logo-nssvie-light.png",
        "image_dark": "icons/logo-nssvie-dark.png",
    },
}

html_context = {
    "default_mode": "auto",
    "github_url": "https://github.com",
    "github_user": "dsagolla",
    "github_repo": "nssvie",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
}

html_title = f"{project} Manual"
html_last_updated_fmt = '%b %d, %Y'

html_static_path = ["_static"]

html_js_files = ["js/custom-pydata.css"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# =====
# setup
# =====


def setup(app):
    """
    This function is used to employ a css file to the themes.
    Note: paths are relative to /docs/source/_static
    """

    app.add_css_file('css/custom-pydata.css')
    app.add_js_file('js/custom-pydata.js')
    # app.add_css_file('css/custom.css')
    # app.add_css_file('css/custom-anaconda-doc.css')