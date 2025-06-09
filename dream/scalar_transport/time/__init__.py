r""" Definitions of the temporal discretizations for the scalar transport equation.

At the continuous level, we solve the current PDE:

.. math::
    \partial_t u + f(u) + g(u, \nabla u) = 0,

where :math:`f(u) = \nabla \cdot (\bm{b} u)` is the inviscid flux and :math:`g(u, \nabla u) = \nabla \cdot (\kappa \nabla u)` is the viscous flux.

:note: The spatial fluxes are discretized, assuming they are on the left-hand side, see above.
"""

from .implicit import (ImplicitEuler, 
                       BDF2,
                       SDIRK22,
                       SDIRK33)
from .explicit import (ExplicitEuler,
                       SSPRK3,
                       CRK4)
from .imex import (IMEXRK_ARS443)

__all__ = ['ImplicitEuler',
           'BDF2',
           'SDIRK22',
           'SDIRK33',
           'ExplicitEuler',
           'SSPRK3',
           'CRK4',
           'IMEXRK_ARS443']




