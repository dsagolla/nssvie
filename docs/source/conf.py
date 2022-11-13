# =======
# Imports
# =======

from datetime import date

# -- Project information -----------------------------------------------------

project = 'nssvie'
author = 'Daniel Sagolla'
copyright = f'{date.today().year}, ' + author

# -- Sphinx Settings ----------------------------------------------------------

# Check links and references
nitpicky = True

# Why setting "root_doc": by using toctree in index.rst, two things are added
# to index.html main page: (1) a toc in that location of page, (2) toc in the
# sidebar menu. If we add :hidden: option to toctree, it removes toc from
# both page and sidebar menu. There is no way we can have only one of these,
# for instance, toc only in the page, but not in the menu. A solution to
# this is as follows:
#   1. Set "root_doc= 'content'". Then add those toc that should go into the
#      menu in the content.rst file.
#   2. Add those toc that should go into the page in index.rst file.
# This way, we can control which toc appears where.
#
# A problem: by setting "root_doc='content'", the sidebar logo links to
# contents.html page, not the main page. There is a logo_url variable but it
# does not seem to do anything. To fix this, I added a javascript (see in
# /docs/source/_static/js/custom-pydata.js) which overwrites
# <a href"path/contents.html"> to <a href="path/index.html>".
# root_doc = "contents"

# Common definitions for the whole pages
rst_epilog = r'''
.. role:: synco
   :class: synco

.. |project| replace:: :synco:`nssvie`
'''

# Figure, Tables, etc numbering
# numfig = True
# numfig_format = {
#     'figure': 'Figure %s',
#     'table': 'Table %s'
# }

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_toggleprompt',
    # 'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    # 'sphinx_automodapi.automodapi',
    'numpydoc',                               # either use napoleon or numpydoc
    'sphinx_design',
    'sphinx-prompt',
    'sphinx_copybutton',
]

# Copy button settings
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r'>>> |\.\.\. '

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
    'tex2jax': {
        'inlineMath': [["\\(", "\\)"]],
        'displayMath': [["\\[", "\\]"]],
    },
}

# LaTeX
# 'sphinx.ext.imgmath',
# imgmath_image_format = 'svg'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

# Options for theme
html_theme_options = {
    "navbar_end": [
        "theme-switcher",
        "search-field.html",
        "navbar-icon-links.html"
    ],
    "page_sidebar_items": ["page-toc", "edit-this-page"],
    # "header_links_before_dropdown": 4,
    "use_edit_page_button": True,
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
            "url": "https://twitter.com/dsagolla",
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
        "image_light": "images/icons/logo-ortho-light.png",
        "image_dark": "images/icons/logo-ortho-dark.png",
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

html_static_path = ['_static']

html_js_files = ["js/custom-pydata.css"]
# html_logo = '_static/images/icons/logo-imate-light.png'
html_favicon = '_static/images/icons/favicon.ico'

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
