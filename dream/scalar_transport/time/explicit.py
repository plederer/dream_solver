""" Definitions of explicit time marching schemes for a scalar transport equation. """
from __future__ import annotations
from dream.config import Integrals, Log
from dream.time import TimeSchemes

import ngsolve as ngs
import typing
import numpy as np


class ExplicitSchemes(TimeSchemes):

    def assemble(self) -> None:

        condense = self.root.fem.static_condensation
        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.add_sum_of_integrals(self.blf, self.root.fem.blf, 'explicit form')

        self.lf = None
        if any([space for space, forms in self.root.fem.lf.items() if forms]):
            self.lf = ngs.LinearForm(self.root.fem.fes)
            self.add_sum_of_integrals(self.lf, self.root.fem.lf, 'linear form')
            self.lf.Assemble()

        # Precompute the  mass matrix.
        u, v = self.root.fem.TnT['u']
        self.mass = ngs.BilinearForm(self.root.fem.fes)
        self.mass += ngs.InnerProduct(u, v) * ngs.dx
        self.mass.Assemble()
        self.mass = self.mass.mat
        self.minv = self.mass.Inverse(self.root.fem.fes.FreeDofs(), inverse="sparsecholesky")

        # TODO: Use matrix-free implementation
        # self.mass = self.root.fem.spaces['u'].Mass(1.0)

        # Can be avoided, but for readability and given the simplicity of the PDE, we allocate a rhs anyway.
        self.rhs = self.root.fem.gfu.vec.CreateVector()

    def add_symbolic_temporal_forms(self, blf, lf):
        """ Not used in explicit schemes, since the temporal forms are not needed. """
        ...

    def is_diverged(self, vec) -> bool:
        return np.isnan(vec).any()


class ExplicitEuler(ExplicitSchemes):
    r""" Class responsible for implementing an explicit (forwards-)Euler time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). Namely,

    .. math::
        \widetilde{\bm{M}} \bm{u}^{n+1} = \widetilde{\bm{M}} \bm{u}^{n} - \bm{B} \bm{u}^{n} ,

    where :math:`\widetilde{\bm{M}} = \frac{1}{\delta t} \int_{D} u v\, d\bm{x}` is the weighted mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "explicit_euler"

    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:

        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec

        self.root.fem.gfu.vec.data -= self.dt.Get() * self.minv * self.rhs

        self.t.Set(t0 + self.dt.Get())

        yield {'t': self.t.Get()}


class SSPRK3(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 3rd-order strong-stability-preserving Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), see Section 4.1, Equation 4.2 in :cite:`gottlieb2001strong`. This is implemented as

    .. math::
        \bm{y}_{1}   &=             \bm{u}^{n} -                                      \widetilde{\bm{M}}^{-1} \bm{B} \bm{u}^{n},\\[2ex]
        \bm{y}_{2}   &= \frac{3}{4} \bm{u}^{n} + \frac{1}{4} \bm{y}_{1} - \frac{1}{4} \widetilde{\bm{M}}^{-1} \bm{B} \bm{y}_{1},\\[2ex]
        \bm{u}^{n+1} &= \frac{1}{3} \bm{u}^{n} + \frac{2}{3} \bm{y}_{2} - \frac{2}{3} \widetilde{\bm{M}}^{-1} \bm{B} \bm{y}_{2},

    where :math:`\widetilde{\bm{M}} = \frac{1}{\delta t} \int_{D} u v\, d\bm{x}` is the weighted mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "ssprk3"
    number_of_stages: int = 3
    time_of_stages: tuple[float] = (0.0, 1.0, 0.5, 1.0)

    def assemble(self) -> None:
        super().assemble()

        # Reserve space for the solution at the old time step (at t^n).
        self.rhs = self.root.fem.gfu.vec.CreateVector()
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

    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:

        # Extract the current solution.
        Un = self.root.fem.gfu
        self.U0.data = Un.vec

        # First stage.
        self.blf.Apply(Un.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        Un.vec.data = self.U0 - self.dt.Get() * self.minv * self.rhs

        self.set_stage_time(1, t0)
        yield {'t': self.t.Get(), 'stage': 1}

        # Second stage.
        self.blf.Apply(Un.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec

        # NOTE, avoid 1-liners with dependency on the same read/write data.
        Un.vec.data *= self.alpha21
        Un.vec.data += self.alpha20 * self.U0 - self.beta21 * self.dt.Get() * self.minv * self.rhs

        self.set_stage_time(2, t0)
        yield {'t': self.t.Get(), 'stage': 2}

        # Third stage.
        self.blf.Apply(Un.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec

        # NOTE, avoid 1-liners with dependency on the same read/write data.
        Un.vec.data *= self.alpha32
        Un.vec.data += self.alpha30 * self.U0 - self.beta32 * self.dt.Get() * self.minv * self.rhs

        self.set_stage_time(3, t0)
        yield {'t': self.t.Get(), 'stage': 3}

        if self.is_diverged(self.root.fem.gfu.vec):
            yield {"is_diverged": True}


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
    number_of_stages: int = 4
    time_of_stages: tuple[float] = (0.0, 0.5, 0.5, 1.0, 1.0)

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
        self.rhs = self.root.fem.gfu.vec.CreateVector()  # can be avoided (I think)
        self.K1 = self.root.fem.gfu.vec.CreateVector()
        self.K2 = self.root.fem.gfu.vec.CreateVector()
        self.K3 = self.root.fem.gfu.vec.CreateVector()
        self.K4 = self.root.fem.gfu.vec.CreateVector()
        self.Us = self.root.fem.gfu.vec.CreateVector()

    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:

        # First stage.
        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        self.K1.data = -self.dt.Get() * self.minv * self.rhs

        self.set_stage_time(1, t0)
        yield {'t': self.t.Get(), 'stage': 1}

        # Second stage.
        self.Us.data = self.root.fem.gfu.vec + self.a21 * self.K1
        self.blf.Apply(self.Us, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        self.K2.data = -self.dt.Get() * self.minv * self.rhs

        self.set_stage_time(2, t0)
        yield {'t': self.t.Get(), 'stage': 2}

        # Third stage.
        self.Us.data = self.root.fem.gfu.vec + self.a32 * self.K2
        self.blf.Apply(self.Us, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        self.K3.data = -self.dt.Get() * self.minv * self.rhs

        self.set_stage_time(3, t0)
        yield {'t': self.t.Get(), 'stage': 3}

        # Fourth stage.
        self.Us.data = self.root.fem.gfu.vec + self.K3
        self.blf.Apply(self.Us, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        self.K4.data = -self.dt.Get() * self.minv * self.rhs

        self.set_stage_time(4, t0)
        yield {'t': self.t.Get(), 'stage': 4}

        # Reconstruct the solution at t^{n+1}.
        self.root.fem.gfu.vec.data += self.b1 * self.K1 \
            + self.b2 * self.K2 \
            + self.b3 * self.K3 \
            + self.b4 * self.K4
