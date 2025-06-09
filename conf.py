import sphinx_rtd_theme
import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('.'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dream'
copyright = '2025'
author = 'Philip Lederer, Jan Ellmenreich, Edmond Shehadi'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinxcontrib.bibtex",
              'sphinxcontrib.tikz',
              "sphinx.ext.autodoc",
              'sphinx.ext.autosummary',
              "sphinx.ext.mathjax",
              "sphinx.ext.todo",
              "sphinxcontrib.jquery",
              "IPython.sphinxext.ipython_console_highlighting",
              "IPython.sphinxext.ipython_directive",
              "nbsphinx",
              "myst_parser",
              ]

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', "**.ipynb_checkpoints"]

tikz_proc_suite = 'pdf2svg'

# autodoc settings
add_module_names = False
autodoc_member_order = 'bysource'

# autosummary settings
autosummary_generate = True

highlight_language = 'python3'
pygments_style = "sphinx"

# bibtex settings
bibtex_bibfiles = ['introduction/literature.bib']
bibtex_default_style = 'unsrt'

tikz_latex_preamble = r"\usepackage{bm}"


# html_static_path = [os.path.abspath('.') + '/_static']

html_js_files = ['webgui.js', 'webgui_jupyter_widgets.js']

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
            "mat": ["{\\bm{#1}}", 1],
            "facets": r"\mathcal{F}_h",
            "mesh": r"\mathcal{T}_h",
        }
    }
}

myst_enable_extensions = ["amsmath", "dollarmath", "fieldlist"]

#, "colon_fence","html_admonition", "html_image", "attrs_inline"]
#myst_allow_html = True 


suppress_warnings = ["myst.header"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# nbsphinx_execute = 'always'


html_static_path = ['_static']

nbsphinx_timeout = 600
html_sourcelink_suffix = ''
nbsphinx_allow_errors = False

# bsphinx_prolog = r"""
# {% set docname = env.doc2path(env.docname, base='').replace('i-tutorials/', '') %}

# .. raw:: html

#     <style>
#         .p-Widget {
#             height: 400px;
#         }
#         .dg.main {
#             margin-left: 0px;
#         }
#         div.p-Widget div div div div.dg ul li {
#             list-style: none;
#             margin-left: 0px;
#         }
#         div.p-Widget div div div div.dg ul li div.dg {
#             margin-bottom: 0px;
#         }
#     </style>

# .. only:: html
#     .. role:: raw-html(raw)
#         :format: html

#     .. nbinfo::

#         This page was generated from `{{ docname }}`__.

#     __ ../../jupyter-files/{{docname}}
# """

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    # 'prev_next_buttons_location': 'bottom',
    # 'style_external_links': False,
    # 'vcs_pageview_mode': '',
    # 'style_nav_header_background': 'white',
    # 'flyout_display': 'hidden',
    # 'version_selector': True,
    # 'language_selector': True,
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_logo = 'graphics/dream_logo.svg'

def setup(app):
    app.add_css_file("custom.css")

# html_static_path = ['_static']


rst_prolog = """
.. role:: py(code)
  :language: python3
"""
