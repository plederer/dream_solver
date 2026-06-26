scalar_transport
========================================
.. currentmodule:: dream.scalar_transport

The ``dream.scalar_transport`` module solves the linear scalar convection–diffusion equation
using Hybridised Discontinuous Galerkin (HDG) or Discontinuous Galerkin (DG) methods in space,
combined with a range of implicit, explicit, and IMEX time integration schemes.
The main entry point is :py:class:`~dream.scalar_transport.solver.ScalarTransportSolver`.

.. note:: We currently only support a linear formulation.

.. include:: equations.md
    :parser: myst_parser.sphinx_

API Reference
-------------

.. autosummary::
    :toctree:
    :recursive:

    solver
    spatial
    time
    riemann_solver
    config

Examples
--------
.. toctree::
    :maxdepth: 1

    ../../examples/quick_start_scalar_transport.ipynb
    ../../examples/wave1d.ipynb
