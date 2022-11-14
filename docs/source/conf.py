# =======
# Imports
# =======

from datetime import date

# -- General configuration ---------------------------------------------

extensions = [
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    # 'sphinx.ext.autodoc',
    # 'sphinx_math_dollar',
    # 'sphinx.ext.mathjax',
    # 'sphinx.ext.graphviz',
    # 'sphinx.ext.inheritance_diagram',
    # 'sphinx.ext.viewcode',
    'sphinx_toggleprompt',
    # 'sphinx.ext.autosectionlabel',
    # 'sphinx.ext.autosummary',
    # 'sphinx_automodapi.automodapi',
    # 'numpydoc',
    'sphinx-prompt',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:

source_suffix = [".rst", ".md"]

# The master toctree document.
main_doc = "index"

# -- Project information -----------------------------------------------

project = "nssvie"
author = "Daniel Sagolla"
copyright = f"{date.today().year}, " + author
version = "0.0.1"
# The full version, including alpha/beta/rc tags.
release = "0.0.1"

# -- Intersphinx Mapping------------------------------------------------

intersphinx_mapping = {}

# -- Sphinx Settings ---------------------------------------------------

# Check links and references
nitpicky = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

# Common definitions for the whole pages
rst_epilog = r"""
.. role:: synco
    :class: synco

.. |project| replace:: :synco:`nssvie`
"""


# Copy button settings
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "

# Automatically generate autosummary after each build
autosummary_generate = True
autosummary_imported_members = True

# Remove the module names from the signature
# add_module_names = False

# automodapi
numpydoc_show_class_members = False

# Added after including sphinx_math_dollar. The following prevents msthjax to
# parse $ and $$.
mathjax3_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    },
}

# LaTeX
# 'sphinx.ext.imgmath',
# imgmath_image_format = 'svg'

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_favicon = "_static/images/icons/favicon.ico"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "navbar_align": "left",
    "navbar_end": [
        "theme-switcher",
        # "search-field.html",
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
    # "pygment_light_style": "tango",
    # "pygment_dark_style": "native",
    "logo": {
        "image_light": "images/icons/logo-nssvie-light.png",
        "image_dark": "images/icons/logo-nssvie-dark.png",
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
    "index": [],  # Remove sidebars on landing page to save space
}

# html_sidebars = {
#     "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
# }

html_title = f"{project} Manual"
html_last_updated_fmt = "%b %d, %Y"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_js_files = ["js/custom-pydata.css"]
# html_logo = '_static/images/icons/logo-imate-light.png'
