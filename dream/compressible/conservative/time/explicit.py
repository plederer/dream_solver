""" Definitions of explicit time integration schemes for conservative methods. """
from __future__ import annotations

import ngsolve as ngs
import logging
import typing
from dream.time import TimeSchemes
from dream.config import Integrals

logger = logging.getLogger(__name__)


class ExplicitSchemes(TimeSchemes):

    time_levels = ('n', 'n+1')

    def assemble(self) -> None:

        # Check that a mass matrix is indeed defined in the bilinear form dictionary.
        if "mass" not in self.root.fem.blf['U']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        compile = self.root.optimizations.compile

        # NOTE, we assume that self.lf is not needed here (for efficiency).
        self.blf = ngs.BilinearForm(self.root.fem.fes)
        self.rhs = self.root.fem.gfu.vec.CreateVector()
        self.minv = ngs.BilinearForm(self.root.fem.fes, symmetric=True)

        # Step 1: precompute and store the inverse mass matrix. Note, this is scaled by dt.
        if compile.realcompile:
            self.minv += self.root.fem.blf['U']['mass'].Compile(**compile)
        else:
            self.minv += self.root.fem.blf['U']['mass']

        # Invert the mass matrix.
        self.minv.Assemble()
        self.minv = self.root.linear_solver.inverse(self.minv, self.root.fem.fes)

        # Remove the mass matrix item from the bilinear form dictionary, before proceeding.
        self.root.fem.blf['U'].pop('mass')

        # Process all items in the relevant bilinear and linear forms.
        self.add_sum_of_integrals(self.blf, self.root.fem.blf)

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        gfus['n+1'] = u

        # Add the mass matrix.
        blf['U']['mass'] = ngs.InnerProduct(u/self.dt, v) * ngs.dx

    def get_current_level(self, space: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[space]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def update_solution(self, t: float):
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        logger.info(f"time: {t:6e}")
        self.update_solution(t)
        yield None


class ExplicitEuler(ExplicitSchemes):
    r""" Class responsible for implementing an explicit (forwards-)Euler time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). Assuming a standard DG formulation,

    .. math::
        \widetilde{\bm{M}} \bm{U}^{n+1} = \widetilde{\bm{M}} \bm{U}^{n} - \bm{f} \big( \bm{U}^{n} \big),

    where :math:`\widetilde{\bm{M}} = \bm{M} / \delta t` is the weighted mass matrix, :math:`\bm{M}` is the mass matrix and :math:`\bm{f}` arises from the spatial discretization of the PDE.
    """
    name: str = "explicit_euler"

    def update_solution(self, t: float):

        # Extract the current solution.
        Un = self.root.fem.gfu

        self.blf.Apply(Un.vec, self.rhs)
        Un.vec.data += self.minv * self.rhs


class SSPRK3(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 3rd-order strong-stability-preserving Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), see Section 4.1, Equation 4.2 in :cite:`gottlieb2001strong`. Assuming a standard DG formulation,

    .. math::
        \bm{y}_{1}   &=             \bm{U}^{n} -                                      \widetilde{\bm{M}}^{-1} \bm{f} \big( \bm{U}^{n} \big),\\[2ex]
        \bm{y}_{2}   &= \frac{3}{4} \bm{U}^{n} + \frac{1}{4} \bm{y}_{1} - \frac{1}{4} \widetilde{\bm{M}}^{-1} \bm{f} \big( \bm{y}_{1} \big),\\[2ex]
        \bm{U}^{n+1} &= \frac{1}{3} \bm{U}^{n} + \frac{2}{3} \bm{y}_{2} - \frac{2}{3} \widetilde{\bm{M}}^{-1 }\bm{f} \big( \bm{y}_{2} \big),

    where :math:`\widetilde{\bm{M}} = \bm{M} / \delta t` is the weighted mass matrix, :math:`\bm{M}` is the mass matrix and :math:`\bm{f}` arises from the spatial discretization of the PDE.
    """
    name: str = "ssprk3"

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Number of stages.
        self.RKnStage = 3

        # Reserve space for the solution at the old time step (at t^n).
        self.Un = self.root.fem.gfu
        self.U0 = self.Un.vec.CreateVector()

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
        Un = self.root.fem.gfu
        self.U0.data = Un.vec

        # First stage.
        self.blf.Apply(Un.vec, self.rhs)
        Un.vec.data = self.U0 + self.minv * self.rhs

        # Second stage.
        self.blf.Apply(Un.vec, self.rhs)

        # NOTE, avoid 1-liners with dependency on the same read/write data. Can be bugged in NGSolve.
        Un.vec.data *= self.alpha21
        Un.vec.data += self.alpha20 * self.U0 + self.beta21 * self.minv * self.rhs

        # Third stage.
        self.blf.Apply(Un.vec, self.rhs)
        # NOTE, avoid 1-liners with dependency on the same read/write data. Can be bugged in NGSolve.
        Un.vec.data *= self.alpha32
        Un.vec.data += self.alpha30 * self.U0 + self.beta32 * self.minv * self.rhs


class CRK4(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 4th-order (classic) Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). Assuming a standard DG formulation,

    .. math::
        \bm{k}_{1}   &= -\widetilde{\bm{M}}^{-1} \bm{f} \big( \bm{U}^{n}                \big),\\[2ex]
        \bm{k}_{2}   &= -\widetilde{\bm{M}}^{-1} \bm{f} \big( \bm{U}^{n} + \bm{k}_1 / 2 \big),\\[2ex] 
        \bm{k}_{3}   &= -\widetilde{\bm{M}}^{-1} \bm{f} \big( \bm{U}^{n} + \bm{k}_2 / 2 \big),\\[2ex]
        \bm{k}_{4}   &= -\widetilde{\bm{M}}^{-1} \bm{f} \big( \bm{U}^{n} + \bm{k}_3     \big),\\[2ex]
        \bm{U}^{n+1} &= \bm{U}^{n} + \big( \bm{k}_1 + 2\bm{k}_2 + 2\bm{k}_3 + \bm{k}_4  \big) / 6,

    where :math:`\widetilde{\bm{M}} = \bm{M} / \delta t` is the weighted mass matrix, :math:`\bm{M}` is the mass matrix and :math:`\bm{f}` arises from the spatial discretization of the PDE.
    """

    name: str = "crk4"

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Number of stages.
        self.RKnStage = 4

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

        # First stage.
        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        self.K1.data = self.minv * self.rhs

        # Second stage.
        self.Us.data = self.root.fem.gfu.vec + self.a21 * self.K1
        self.blf.Apply(self.Us, self.rhs)
        self.K2.data = self.minv * self.rhs

        # Third stage.
        self.Us.data = self.root.fem.gfu.vec + self.a32 * self.K2
        self.blf.Apply(self.Us, self.rhs)
        self.K3.data = self.minv * self.rhs

        # Fourth stage.
        self.Us.data = self.root.fem.gfu.vec + self.K3
        self.blf.Apply(self.Us, self.rhs)
        self.K4.data = self.minv * self.rhs

        # Reconstruct the solution at t^{n+1}.
        self.root.fem.gfu.vec.data += self.b1 * self.K1 \
            + self.b2 * self.K2 \
            + self.b3 * self.K3 \
            + self.b4 * self.K4
