# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# EdelweissFE registers its input language keywords as an import side effect of its input
# file parser, and several of its modules look those keywords up at import time. autodoc
# imports modules standalone, which does not trigger the registration, so it is done here
# once from a clean import context.
from edelweissfe.utils.inputlanguage import InputLanguage

InputLanguage().ensureParserLoaded()

project = "EdelweissFD"
copyright = "2026, Alexander Dummer"
author = "Alexander Dummer"
release = "v26.11"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autosummary_generate = True

numpydoc_show_class_members = False

autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "edelweissfe": ("https://edelweiss-numerics.github.io/EdelweissFE", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

pygments_style = "nord"
