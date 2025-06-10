""" Setup script for dream package.
This script is used to load the dynamic dipendencies of the package,
especially the ngsolve package, since we don't want to force pip install
if it is available in the path.
"""
from setuptools import setup

install_requires = [
    "numpy>=1.26.4",
    "matplotlib>=3.9.1",
    "webgui_jupyter_widgets>=0.2.31",
    "jupyter>=1.1.1"
]

try:
    import ngsolve
except ImportError:
    install_requires.append("ngsolve>=6.2.2501")
  
extras_require = {
    "pandas": "pandas>=2.2.2",
    "matplotlib": "matplotlib>=3.9.1",
    "webgui": "webgui_jupyter_widgets>=0.2.31"
}

setup(
    install_requires=install_requires,
    extras_require=extras_require,
)
