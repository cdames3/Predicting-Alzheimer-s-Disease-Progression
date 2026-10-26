"""Sphinx configuration for the project documentation.

Build locally with:
    sphinx-build -b html docs docs/_build/html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "Predicting Cognitive Decline from Longitudinal MRI"
author = "Christina Dames, Cinthia Ochoa Torre, Rayna Alexandre-Petrie"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",
]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}
# Heavy dependencies are mocked so the docs also build without them
# (for example on Read the Docs).
autodoc_mock_imports = ["torch", "wandb", "optuna", "nibabel"]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

exclude_patterns = ["_build"]
html_theme = "alabaster"
