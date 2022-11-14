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

extensions = []

source_suffix = {".rst": "restructuredtext"}

root_doc = "index"
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

html_theme_options = {
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "theme-switcher",
        "searchbox.html",
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
        "image_light": "icons/logo-nssvie-light.png",
        "image_dark": "icons/logo-nssvie-dark.png",
    },
}


html_static_path = ['_static']
