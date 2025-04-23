import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'manify'
copyright = '2025, Philippe Chlenski and Kaizhu Du'
author = 'Philippe Chlenski and Kaizhu Du'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     'sphinx.ext.autodoc',
#     'sphinx.ext.napoleon',
#     'sphinx_autodoc_typehints',
#     'sphinx.ext.autosummary',
#     'autoapi.extension',
# ]
extensions = [
    'sphinx.ext.napoleon',
    # 'sphinx_autodoc_typehints',
    'autoapi.extension',
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
]

# autosummary_generate = True
# autodoc_default_options = {
#     "members": True,
#     "undoc-members": True,
#     "show-inheritance": True,
# }

autoapi_type = 'python'
autoapi_dirs = [os.path.abspath('../../manify')]
autoapi_ignore = ["*/utils/*"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
