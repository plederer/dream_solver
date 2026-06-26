compressible_flow
========================================
.. currentmodule:: dream.compressible_flow

The ``dream.compressible_flow`` module provides a high-order HDG/DG solver for the
compressible Navier-Stokes equations. It covers everything from equation-of-state and viscosity
models through Riemann solvers and non-reflecting boundary conditions to a wide selection of
implicit, explicit, and IMEX time integration schemes.

.. note:: We currently only support two-dimensional domains.

.. include:: equations.md
    :parser: myst_parser.sphinx_

API Reference
-------------
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

Examples
--------
.. toctree::
    :maxdepth: 1

    ../../examples/quick_start_compressible.ipynb
    ../../examples/euler_around_naca.ipynb
    ../../examples/isentropic_vortex.ipynb
