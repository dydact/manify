"""Generate the API reference pages from the codebase."""

from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath("."))

import mkdocs_gen_files

src_dir = "manify"
doc_dir = "reference"

# Custom titles mapping
module_titles = {
    "clustering": "Clustering",
    "curvature_estimation": "Curvature Estimation",
    "embedders": "Embedders",
    "manifolds": "Manifolds",
    "optimizers": "Optimizers",
    "predictors": "Predictors",
    "utils": "Utilities",
}

# Process all Python files
for path in sorted(Path(src_dir).rglob("*.py")):
    module_path = path.relative_to(src_dir).with_suffix("")
    doc_path = path.relative_to(src_dir).with_suffix(".md")
    full_doc_path = Path(doc_dir, doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Skip if parts is empty
    if not parts:
        continue

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(full_doc_path), exist_ok=True)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)

        # Create a nicer title for the page
        if len(parts) == 1 and parts[0] in module_titles:
            title = module_titles[parts[0]]
        else:
            title = parts[-1].replace("_", " ").title()

        fd.write(f"# {title}\n\n")
        fd.write(f"::: {src_dir}.{identifier}\n")
        fd.write("    options:\n")
        fd.write("        show_category_heading: false\n")
        fd.write("        heading_level: 2\n")
        fd.write("        members_order: source\n")
        fd.write("        filters: ['!^_', '!^Parameters$']\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Create root package documentation
with mkdocs_gen_files.open(f"{doc_dir}/index.md", "w") as fd:
    fd.write("# API Reference\n\n")
    fd.write("## Overview\n\n")

    # Only show modules, not individual classes
    fd.write(f"::: {src_dir}\n")
    fd.write("    options:\n")
    fd.write("        show_category_heading: false\n")
    fd.write("        members_order: source\n")
    fd.write("        filters: ['!^_', '!^Parameters$']\n")
    fd.write("        show_root_heading: false\n")
    fd.write("        heading_level: 3\n")
    fd.write("        members: false\n")  # This line prevents showing class members
