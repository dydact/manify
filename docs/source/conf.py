import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
project = 'manify'
copyright = '2025, Philippe Chlenski and Kaizhu Du'
author = 'Philippe Chlenski and Kaizhu Du'
release = '0.0.2'


extensions = [
    'sphinx.ext.napoleon',
    # 'sphinx_autodoc_typehints',
    'autoapi.extension',
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
]

autoapi_type = 'python'
autoapi_dirs = [os.path.abspath('../../manify')]
autoapi_ignore = ["*/utils/*"]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    "members": True,
    "undoc-members": False,  # Set to False to hide undocumented members
    "private-members": False,
    "show-inheritance": True,
    "exclude-members": "__init__",
}
autoclass_content = "class"
autoapi_options = [
    "members",           # pull in documented members
    "imported-members",  # if you want to re-export things
    "show-inheritance",
    # note: NO 'undoc-members' here
]
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

