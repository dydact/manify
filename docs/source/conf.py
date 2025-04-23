import os

project = "manify"
copyright = "2025, Philippe Chlenski and Kaizhu Du"
author = "Philippe Chlenski and Kaizhu Du"
release = "0.0.2"

extensions = ["sphinx.ext.napoleon", "autoapi.extension"]

napoleon_use_rtype = False

autoapi_type = "python"
autoapi_dirs = [os.path.abspath("../../manify")]
autoapi_options = ["members", "undoc-members", "imported-members", "show-inheritance", "show-module-summary"]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
