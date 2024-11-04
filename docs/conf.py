import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('.'))
from dream import SolverConfiguration
from dream.config import configuration
import sphinx_rtd_theme

# sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# print(sys.path)

# from configuration import SolverConfiguration


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dream'
copyright = '2023, Philip Lederer'
author = 'Philip Lederer'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.mathjax', "myst_parser", 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

mathjax3_config = {
    # "loader": {"load": ['[tex]/color']},
    "tex": {
        # "packages" : {'[+]': ['color']},
        "macros": {
            "bm": ["\\boldsymbol{#1}", 1],
            "div": "{\\operatorname{div}}",
            "tr": "{\\operatorname{tr}}",
            "dev": "{\\operatorname{dev}}",
            "curl": "{\\operatorname{curl}}",
            "grad": "{\\nabla}",
            "T": "{\\rm{T}}",
            "I": "{\\operatorname{I}}",
            "Re": "{\\operatorname{\\rm{Re}}}",
            "Pr": "{\\operatorname{\\rm{Pr}}}",
            "Ma": "{\\operatorname{\\rm{Ma}}}",
            "Fr": "{\\operatorname{\\rm{Fr}}}",
            "jump": ["{{[\\![#1]\\!]}}", 1],
            "sym": ["{{\\rm sym} (#1) }", 1],
            "skw": ["{{\\rm skw} (#1) }", 1],
            "vec": ["{\\bm{#1}}", 1],
            "mat": ["{#1}", 1],
        }
    }
}

myst_enable_extensions = ["amsmath", "dollarmath"]
suppress_warnings = ["myst.header"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# def test_autodoc_process_bases(app, what, name, obj, options, signature, return_annotation):
#     if isinstance(obj, configuration):
#         options['show_inheritance'] = True
#         return (f"{signature}", return_annotation)

# def setup(app):
#     app.connect("autodoc-process-signature", test_autodoc_process_bases)