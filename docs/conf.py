import os
import sys

# Make the package importable when building locally without installing it
sys.path.insert(0, os.path.abspath(".."))

project = "tnsd_access"
copyright = "2024, tnsd_access contributors"
author = "tnsd_access contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autodoc_mock_imports = ["mne", "numcodecs", "zarr", "tqdm", "dvc"]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "sphinx_rtd_theme"
html_static_path = []

templates_path = []
exclude_patterns = ["_build"]
