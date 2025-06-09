compressible
========================================
.. currentmodule:: dream.compressible

.. note:: We currently only support two-dimensional domains.

.. autosummary::
    :toctree:
    :recursive:

    solver
    conservative
    eos
    riemann_solver
    viscosity
    scaling
    config
    
.. include:: equations.md
    :parser: myst_parser.sphinx_

Examples
--------
.. toctree::
    :maxdepth: 1

    ../../examples/euler_around_naca.ipynb
    ../../examples/isentropic_vortex.ipynb
