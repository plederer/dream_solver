# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dream_solver'
copyright = '2023, Philip Lederer'
author = 'Philip Lederer'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.mathjax', "myst_parser"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

from latex_macros import mymacros

mathjax3_config = {                  
    # "loader": {"load": ['[tex]/color']},
    "tex": { 
        "inlineMath": [['\\(', '\\)']],
        "displayMath": [["\\[", "\\]"]],
        # "packages" : {'[+]': ['color']},                       
        "macros": mymacros
    }                           
    }                 

myst_enable_extensions = ["amsmath","dollarmath"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']
