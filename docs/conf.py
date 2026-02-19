# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the python package to the path
sys.path.insert(0, os.path.abspath('../python'))

# -- Project information -----------------------------------------------------

project = 'Asala'
copyright = '2024-2026, Asala Contributors'
author = 'Asala Contributors'
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
    'sphinxcontrib.mermaid',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'furo'

# Theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,

    "light_css_variables": {
        "font-stack": "Manrope, -apple-system, BlinkMacSystemFont, Segoe UI, \
                      Roboto, Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, \
                      Helvetica Neue, Arial, sans-serif",
        "font-stack--monospace": "ui-monospace, SFMono-Regular, Menlo, Monaco, \
                                Consolas, Liberation Mono, Courier New, monospace",
    },
    "dark_css_variables": {
        "font-stack": "Manrope, -apple-system, BlinkMacSystemFont, Segoe UI, \
                      Roboto, Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, \
                      Helvetica Neue, Arial, sans-serif",
        "font-stack--monospace": "ui-monospace, SFMono-Regular, Menlo, Monaco, \
                                Consolas, Liberation Mono, Courier New, monospace",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'css/custom.css',
]

# Favicon
html_favicon = '_static/favicon.ico'

# -- Extension configuration -------------------------------------------------

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = [('Returns', 'params_style')]

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Mermaid configuration
mermaid_version = 'latest'
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme: 'default'});"

# Copy button configuration
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: |\s*\.\.\.\s*'
copybutton_prompt_is_regexp = True
