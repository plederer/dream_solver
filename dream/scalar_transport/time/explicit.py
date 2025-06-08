""" Definitions of explicit time marching schemes for a scalar transport equation. """
from __future__ import annotations
from dream.config import Integrals
from dream.time import TimeSchemes

import ngsolve as ngs
import logging
import typing

logger = logging.getLogger(__name__)


class ExplicitSchemes(TimeSchemes):
    time_levels = ('n+1',)

    def assemble(self) -> None:

        compile = self.root.optimizations.compile

        # Check that a mass matrix is indeed defined in the bilinear form dictionary.
        if "mass" not in self.root.fem.blf['U']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        self.blf = ngs.BilinearForm(self.root.fem.fes)
        self.lf = ngs.LinearForm(self.root.fem.fes)
        self.rhs = self.root.fem.gfu.vec.CreateVector()
        
        # Step 1: precompute the mass matrix. Note, this is scaled by dt.
        mass = ngs.BilinearForm(self.root.fem.fes, symmetric=True)
        if compile.realcompile:
            mass += self.root.fem.blf['U']['mass'].Compile(**compile)
        else:
            mass += self.root.fem.blf['U']['mass']

        # Assemble the mass matrix.
        mass.Assemble()
        # Invert and store the mass matrix.
        self.minv = self.root.linear_solver.inverse(mass, self.root.fem.fes)

        # Remove the mass matrix item from the bilinear form dictionary, before proceeding.
        self.root.fem.blf['U'].pop('mass')

        # Process all items in the relevant bilinear and linear forms.
        self.add_sum_of_integrals(self.blf, self.root.fem.blf)

        # Add the integrals to the linear functional, if they exist (all symbolic at this point).
        self.add_sum_of_integrals(self.lf, self.root.fem.lf)
        
        # Before proceeding, check whether we require a linear functional or not. 
        if self.lf.integrators:
            self.lf.Assemble()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        gfus['n+1'] = u
        
        blf['U']['mass'] = ngs.InnerProduct(u/self.dt, v) * ngs.dx

    def update_solution(self, t: float):
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        logger.info(f"time: {t:6e}")
        self.update_solution(t)
        yield None



class ExplicitEuler(ExplicitSchemes):
    r""" Class responsible for implementing an explicit (forwards-)Euler time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). Namely,

    .. math::
        \widetilde{\bm{M}} \bm{u}^{n+1} = \widetilde{\bm{M}} \bm{u}^{n} - \bm{B} \bm{u}^{n} ,

    where :math:`\widetilde{\bm{M}} = \frac{1}{\delta t} \int_{D} u v\, d\bm{x}` is the weighted mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "explicit_euler"

    def update_solution(self, t: float):
        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        if self.lf.integrators:
            self.rhs -= self.lf.vec
        
        self.root.fem.gfu.vec.data -= self.minv * self.rhs


class SSPRK3(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 3rd-order strong-stability-preserving Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), see Section 4.1, Equation 4.2 in :cite:`gottlieb2001strong`. This is implemented as

    .. math::
        \bm{y}_{1}   &=             \bm{u}^{n} -                                      \widetilde{\bm{M}}^{-1} \bm{B} \bm{u}^{n},\\[2ex]
        \bm{y}_{2}   &= \frac{3}{4} \bm{u}^{n} + \frac{1}{4} \bm{y}_{1} - \frac{1}{4} \widetilde{\bm{M}}^{-1} \bm{B} \bm{y}_{1},\\[2ex]
        \bm{u}^{n+1} &= \frac{1}{3} \bm{u}^{n} + \frac{2}{3} \bm{y}_{2} - \frac{2}{3} \widetilde{\bm{M}}^{-1} \bm{B} \bm{y}_{2},

    where :math:`\widetilde{\bm{M}} = \frac{1}{\delta t} \int_{D} u v\, d\bm{x}` is the weighted mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "ssprk3"

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Reserve space for the solution at the old time step (at t^n).
        self.U0 = self.root.fem.gfu.vec.CreateVector()

        # Time stamps for the stage values between t = [n,n+1].
        self.c1 = 0.0
        self.c2 = 1.0
        self.c3 = 0.5

        # Define the SSP-RK3 coefficients using alpha and beta (see reference [1]).
        self.alpha20 = 0.75
        self.alpha21 = 0.25
        self.beta21 = 0.25

        self.alpha30 = 1.0/3.0
        self.alpha32 = 2.0/3.0
        self.beta32 = 2.0/3.0

    def update_solution(self, t: float):

        # Extract the current solution.
        self.U0.data = self.root.fem.gfu.vec

        # Stage: 1.
        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        if self.lf.integrators:
            self.rhs -= self.lf.vec

        self.root.fem.gfu.vec.data = self.U0 - self.minv * self.rhs

        # Stage: 2.
        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        if self.lf.integrators:
            self.rhs -= self.lf.vec

        self.root.fem.gfu.vec.data *= self.alpha21
        self.root.fem.gfu.vec.data += self.alpha20 * self.U0 - self.beta21 * self.minv * self.rhs

        # Stage: 3.
        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        if self.lf.integrators:
            self.rhs -= self.lf.vec

        self.root.fem.gfu.vec.data *= self.alpha32
        self.root.fem.gfu.vec.data += self.alpha30 * self.U0 - self.beta32 * self.minv * self.rhs



class CRK4(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 4th-order (classic) Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). This is implemented as

    .. math::
        \bm{k}_{1}   &= -\widetilde{\bm{M}}^{-1} \bm{B} \bm{u}^{n},\\[2ex]
        \bm{k}_{2}   &= -\widetilde{\bm{M}}^{-1} \bm{B} \big( \bm{u}^{n} + \bm{k}_1 / 2 \big),\\[2ex] 
        \bm{k}_{3}   &= -\widetilde{\bm{M}}^{-1} \bm{B} \big( \bm{u}^{n} + \bm{k}_2 / 2 \big),\\[2ex]
        \bm{k}_{4}   &= -\widetilde{\bm{M}}^{-1} \bm{B} \big( \bm{u}^{n} + \bm{k}_3     \big),\\[2ex]
        \bm{u}^{n+1} &= \bm{u}^{n} + \big( \bm{k}_1 + 2\bm{k}_2 + 2\bm{k}_3 + \bm{k}_4 \big) / 6,

    where :math:`\widetilde{\bm{M}} = \frac{1}{\delta t} \int_{D} u v\, d\bm{x}` is the weighted mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "crk4"

    def assemble(self) -> None:
        super().assemble()

        # Define the CRK4 coefficients.
        self.a21 = 0.5
        self.a32 = 0.5
        self.a43 = 1.0

        self.b1 = 1.0/6.0
        self.b2 = 1.0/3.0
        self.b3 = 1.0/3.0
        self.b4 = 1.0/6.0

        self.c1 = 0.0
        self.c2 = 0.5
        self.c3 = 0.5
        self.c4 = 1.0

        # Reserve space for the tentative solution.
        self.K1 = self.root.fem.gfu.vec.CreateVector()
        self.K2 = self.root.fem.gfu.vec.CreateVector()
        self.K3 = self.root.fem.gfu.vec.CreateVector()
        self.K4 = self.root.fem.gfu.vec.CreateVector()
        self.Us = self.root.fem.gfu.vec.CreateVector()

    def update_solution(self, t: float):

        # Stage: 1.
        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        if self.lf.integrators:
            self.rhs -= self.lf.vec
        
        self.K1.data = -self.minv * self.rhs

        # Stage: 2.
        self.Us.data = self.root.fem.gfu.vec + self.a21 * self.K1
        self.blf.Apply(self.Us, self.rhs)
        if self.lf.integrators:
            self.rhs -= self.lf.vec

        self.K2.data = -self.minv * self.rhs

        # Stage: 3.
        self.Us.data = self.root.fem.gfu.vec + self.a32 * self.K2
        self.blf.Apply(self.Us, self.rhs)
        if self.lf.integrators:
            self.rhs -= self.lf.vec

        self.K3.data = -self.minv * self.rhs

        # Stage: 4.
        self.Us.data = self.root.fem.gfu.vec + self.K3
        self.blf.Apply(self.Us, self.rhs)
        if self.lf.integrators:
            self.rhs -= self.lf.vec

        self.K4.data = -self.minv * self.rhs

        # Reconstruct the solution at t^{n+1}.
        self.root.fem.gfu.vec.data += self.b1 * self.K1 \
                                    + self.b2 * self.K2 \
                                    + self.b3 * self.K3 \
                                    + self.b4 * self.K4






