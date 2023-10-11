mymacros = {
"bm": ["\\boldsymbol{#1}", 1],    
"pdt": ["{\\frac{\\partial #1}{\\partial t}}", 1],
"pdx":["{\\frac{\\partial #1}{\\partial x}}", 1],
"pdy":["{\\frac{\\partial #1}{\\partial y}}", 1],
"pdz":["{\\frac{\\partial #1}{\\partial z}}", 1],
"pdxi":["{\\frac{\\partial #1}{\\partial \\xi}}", 1],
"pdeta":["{\\frac{\\partial #1}{\\partial \\eta}}", 1],
"pdzeta":["{\\frac{\\partial #1}{\\partial \\zeta}}", 1],
"div":"{\\operatorname{div}}",
"eps":"{\\varepsilon}",
"tr":"{\\operatorname{tr}}",
"dev":"{\\operatorname{dev}}",
"curl":"{\\operatorname{curl}}",
"grad":"{\\nabla}",
"T":"{\\rm{T}}",
"I":"{\\operatorname{I}}",
"Re":"{\\operatorname{\\rm{Re}}}",
"Pr":"{\\operatorname{\\rm{Pr}}}",
"Ma":"{\\operatorname{\\rm{Ma}}}",
"Fr":"{\\operatorname{\\rm{Fr}}}",
"dl":["{#1^*}", 1],
"pdtdl":["{\\frac{\\partial #1}{\\partial \\dl{t}}}", 1],
"pdxdl":["{\\frac{\\partial #1}{\\partial \\dl{x}}}", 1],
"pdydl":["{\\frac{\\partial #1}{\\partial \\dl{y}}}", 1],
"pdzdl":["{\\frac{\\partial #1}{\\partial \\dl{z}}}", 1],
"jump":["{{[\\![#1]\\!]}}", 1],
"sym":["{{\\rm sym} (#1) }", 1],
"skw":["{{\\rm skw} (#1) }", 1],
"vec":["{\\bm{#1}}", 1],
"mat":["{#1}", 1],
"rhoref":"{\\rho_{\\rm{ref}}}",
"uref":"{u_{\\rm{ref}}}",
"Tref":"{T_{\\rm{ref}}}",
"muref":"{\\mu_{\\rm{ref}}}",
"Lref":"{L_{\\rm{ref}}}",
"kref":"{k_{\\rm{ref}}}",
"uinf":"{u_\\infty}",
"cinf":"{c_\\infty}",
"muinf":"{\\mu_\\infty}",
"rhoinf":"{\\rho_\\infty}",
"Tinf":"{T_\\infty}",
"pinf":"{p_\\infty}",
"VEL":"{\\vec{u}}",
"HEAT":"{\\vec{q}}",
"TAU":"{\\mat{\\tau}}",
"EPS":"{\\mat{\\varepsilon}}",
"CVAR":"{\\vec{U}}",
"PVAR":"{\\vec{V}}",
"CHVAR":"{\\vec{W}}"}


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

mathjax3_config = {                  
    # "loader": {"load": ['[tex]/color']},
    "tex": { 
        # "packages" : {'[+]': ['color']},                       
        "macros": mymacros
    }                           
    }                 

myst_enable_extensions = ["amsmath","dollarmath"]
suppress_warnings = ["myst.header"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']
