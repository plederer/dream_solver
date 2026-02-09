""" Definitions of explicit time integration schemes for conservative methods. """
from __future__ import annotations

import numpy as np
import ngsolve as ngs
import typing
from dream.time import TimeSchemes, time_generator
from dream.config import Integrals, Log


class ExplicitSchemes(TimeSchemes):

    @property
    def minv(self):
        return self.dt.Get() * self._minv

    def assemble(self) -> None:

        self.blf = ngs.BilinearForm(self.root.fem.fes, nonassemble=True)

        self.lf = None
        if any([space for space, forms in self.root.fem.lf.items() if forms]):
            self.lf = ngs.LinearForm(self.root.fem.fes)
            self.add_sum_of_integrals(self.lf, self.root.fem.lf, 'linear form')
            self.lf.Assemble()

        # Step 1: precompute and store the inverse mass matrix. Note, this is scaled by dt.
        self._minv = self.root.fem.spaces['U'].Mass(1.0).Inverse()

        # Process all items in the relevant bilinear and linear forms.
        rhs = self.parse_sum_of_integrals(self.root.fem.blf)
        self.add_sum_of_integrals(self.blf, rhs, 'explicit bilinear form')

        if self.root.timestep_controller is not None:
            self.root.timestep_controller.initialize()

    def add_symbolic_temporal_forms(self, blf, lf):
        """ Not used in explicit schemes, since the temporal forms are not needed. """
        ...

    def is_diverged(self, vec) -> bool:
        return np.isnan(vec).any()

    @time_generator("time level")
    def solve_current_time_level(self, t0: float):

        for i in range(1, self.number_of_stages + 1):
            yield from self.solve_stage(i, t0)

        self.update_final_stage_solution()

        if self.is_diverged(self.root.fem.gfu.vec):
            yield {'is_diverged': True}


class ExplicitEuler(ExplicitSchemes):
    r""" Class responsible for implementing an explicit (forwards-)Euler time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). Assuming a standard DG formulation,

    .. math::
        \widetilde{\bm{M}} \bm{U}^{n+1} = \widetilde{\bm{M}} \bm{U}^{n} - \bm{f} \big( \bm{U}^{n} \big),

    where :math:`\widetilde{\bm{M}} = \bm{M} / \delta t` is the weighted mass matrix, :math:`\bm{M}` is the mass matrix and :math:`\bm{f}` arises from the spatial discretization of the PDE.
    """
    name: str = "explicit_euler"
    number_of_stages: int = 1
    time_of_stages: tuple[float] = (0.0, 1.0)

    def assemble(self) -> None:
        super().assemble()
        self.rhs = self.root.fem.gfu.vec.CreateVector()

    @time_generator(r"stage {0}")
    def solve_stage(self, stage: int, t0: float) -> typing.Generator[Log, None, None]:

        if stage > 1:
            raise TypeError(f"Stage {stage} does not exist.")

        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)

        if self.lf is not None:
            self.rhs.data -= self.lf.vec

        self.root.fem.gfu.vec.data -= self.minv * self.rhs

        self.set_stage_time(stage, t0)
        yield {'t': self.t.Get(), 'stage': stage}


class RK_ARS22(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 2-stage, 2nd-order Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), see Section 2.6, Equation in :cite:`ascher1997implicit`. Assuming a standard DG formulation,

    """
    name: str = "rk_ars22"
    number_of_stages: int = 2
    time_of_stages: tuple[float] = (0.0, 1.0 - ngs.sqrt(2.0)/2.0, 1.0)

    def assemble(self) -> None:
        super().assemble()

        # Reserve space for the solution at the old time step (at t^n).
        self.U0 = self.root.fem.gfu.vec.CreateVector()
        self.K1 = self.root.fem.gfu.vec.CreateVector()
        self.K2 = self.root.fem.gfu.vec.CreateVector()

        # Butcher tableau coefficients.
        alpha = ngs.sqrt(2.0)/2.0
        gamma = 1.0 - alpha
        delta = 1.0 - 1.0/(2.0*gamma)

        # Time stamps for the stage values between t = [n,n+1].
        self.c1 = 0.0
        self.c2 = gamma
        self.c3 = 1.0

        # Define A-coefficients in the Butcher tableau.
        self.a21 = gamma
        self.a31 = delta
        self.a32 = 1.0 - delta

        # Define the b-coefficients in the Butcher tableau.
        self.b1 = self.a31
        self.b2 = self.a32

        if self.lf is not None:
            raise NotImplementedError(f"RHS term has not been implimented in this scheme (yet).")

    @time_generator(r"stage {0}")
    def solve_stage(self, stage: int, t0: float) -> typing.Generator[Log, None, None]:

        if stage == 1:

            self.U0.data = self.root.fem.gfu.vec
            self.blf.Apply(self.root.fem.gfu.vec, self.K1)
            self.root.fem.gfu.vec.data = self.U0 - self.minv * (self.a21 * self.K1)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        elif stage == 2:

            self.blf.Apply(self.root.fem.gfu.vec, self.K2)
            self.root.fem.gfu.vec.data = self.U0 - self.minv * (self.a31 * self.K1 + self.a32 * self.K2)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        else:
            raise TypeError(f"Stage {stage} does not exist.")


class RK_ARS33(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 3-stage, 3rd-order Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), see Section 2.7, Equation in :cite:`ascher1997implicit`. Assuming a standard DG formulation,

    """
    name: str = "rk_ars33"
    number_of_stages: int = 3
    time_of_stages: tuple[float] = (0.0, 0.4358665215, 0.7179332608, 1.0)

    def assemble(self) -> None:
        super().assemble()

        # Reserve space for the solution at the old time step (at t^n).
        self.U0 = self.root.fem.gfu.vec.CreateVector()
        self.K1 = self.root.fem.gfu.vec.CreateVector()
        self.K2 = self.root.fem.gfu.vec.CreateVector()
        self.K3 = self.root.fem.gfu.vec.CreateVector()

        # Time stamps for the stage values between t = [n,n+1].
        self.c = [0.0, 0.4358665215, 0.7179332608, 1.0]

        # Define A-coefficients in the Butcher tableau.
        self.a21 = 0.4358665215
        self.a31 = 0.3212788860
        self.a32 = 0.3966543747
        self.a41 = -0.1058582960
        self.a42 = 0.5529291479
        self.a43 = 0.5529291479

        # Define the b-coefficients in the Butcher tableau.
        self.b1 = 0.0
        self.b2 = 1.2084966490
        self.b3 = -0.6443631710
        self.b4 = 0.4358665215

        if self.lf is not None:
            raise NotImplementedError(f"RHS term has not been implimented in this scheme (yet).")

    @time_generator(r"stage {0}")
    def solve_stage(self, stage: int, t0: float) -> typing.Generator[Log, None, None]:

        if stage == 1:

            self.U0.data = self.root.fem.gfu.vec
            self.blf.Apply(self.root.fem.gfu.vec, self.K1)
            self.root.fem.gfu.vec.data = self.U0 - self.minv * (self.a21 * self.K1)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        elif stage == 2:

            self.blf.Apply(self.root.fem.gfu.vec, self.K2)
            self.root.fem.gfu.vec.data = self.U0 \
                - self.minv * (self.a31 * self.K1 + self.a32 * self.K2)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        elif stage == 3:

            self.blf.Apply(self.root.fem.gfu.vec, self.K3)
            self.root.fem.gfu.vec.data = self.U0 \
                - self.minv * (self.a41 * self.K1 + self.a42 * self.K2 + self.a43 * self.K3)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        else:
            raise TypeError(f"Stage {stage} does not exist.")

    def update_final_stage_solution(self) -> None:

        self.blf.Apply(self.root.fem.gfu.vec, self.K1)  # K1 is used to store K4, since b1 = 0
        self.root.fem.gfu.vec.data = self.U0 \
            - self.minv * (self.b2 * self.K2 + self.b3 * self.K3 + self.b4 * self.K1)


class RK_ARS43(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 4-stage, 3rd-order Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), see Section 2.8, Equation in :cite:`ascher1997implicit`. Assuming a standard DG formulation,

    """
    name: str = "rk_ars43"
    number_of_stages: int = 4
    time_of_stages: tuple[float] = (0.0, 1.0/2.0, 2.0/3.0, 1.0/2.0, 1.0)

    def assemble(self) -> None:
        super().assemble()

        # Reserve space for the solution at the old time step (at t^n).
        self.U0 = self.root.fem.gfu.vec.CreateVector()
        self.K1 = self.root.fem.gfu.vec.CreateVector()
        self.K2 = self.root.fem.gfu.vec.CreateVector()
        self.K3 = self.root.fem.gfu.vec.CreateVector()
        self.K4 = self.root.fem.gfu.vec.CreateVector()

        # Time stamps for the stage values between t = [n,n+1].
        self.c = [0.0, 1.0/2.0, 2.0/3.0, 1.0/2.0, 1.0]

        # Define A-coefficients in the Butcher tableau.
        self.a21 = 1.0/2.0
        self.a31 = 11.0/18.0
        self.a32 = 1.0/18.0
        self.a41 = 5.0/6.0
        self.a42 = -5.0/6.0
        self.a43 = 1.0/2.0
        self.a51 = 1.0/4.0
        self.a52 = 7.0/4.0
        self.a53 = 3.0/4.0
        self.a54 = -7.0/4.0

        # Define the b-coefficients in the Butcher tableau.
        self.b1 = self.a51
        self.b2 = self.a52
        self.b3 = self.a53
        self.b4 = self.a54

        if self.lf is not None:
            raise NotImplementedError(f"RHS term has not been implimented in this scheme (yet).")

    @time_generator(r"stage {0}")
    def solve_stage(self, stage: int, t0: float) -> typing.Generator[Log, None, None]:

        if stage == 1:

            self.U0.data = self.root.fem.gfu.vec
            self.blf.Apply(self.root.fem.gfu.vec, self.K1)
            self.root.fem.gfu.vec.data = self.U0 - self.minv * (self.a21 * self.K1)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        elif stage == 2:

            self.blf.Apply(self.root.fem.gfu.vec, self.K2)
            self.root.fem.gfu.vec.data = self.U0 \
                - self.minv * (self.a31 * self.K1 + self.a32 * self.K2)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        elif stage == 3:

            self.blf.Apply(self.root.fem.gfu.vec, self.K3)
            self.root.fem.gfu.vec.data = self.U0 \
                - self.minv * (self.a41 * self.K1 + self.a42 * self.K2 + self.a43 * self.K3)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        elif stage == 4:

            self.blf.Apply(self.root.fem.gfu.vec, self.K4)
            self.root.fem.gfu.vec.data = self.U0 \
                - self.minv * (self.a51 * self.K1 + self.a52 * self.K2
                               + self.a53 * self.K3 + self.a54 * self.K4)

            self.set_stage_time(stage, t0)
            yield {'t': self.t.Get(), 'stage': stage}

        else:
            raise TypeError(f"Stage {stage} does not exist.")


class SSPRK3(ExplicitSchemes):
    r""" Class responsible for implementing an explicit 3rd-order strong-stability-preserving Runge-Kutta time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), see Section 4.1, Equation 4.2 in :cite:`gottlieb2001strong`. Assuming a standard DG formulation,

    .. math::
        \bm{y}_{1}   &=             \bm{U}^{n} -                                      \widetilde{\bm{M}}^{-1} \bm{f} \big( \bm{U}^{n} \big),\\[2ex]
        \bm{y}_{2}   &= \frac{3}{4} \bm{U}^{n} + \frac{1}{4} \bm{y}_{1} - \frac{1}{4} \widetilde{\bm{M}}^{-1} \bm{f} \big( \bm{y}_{1} \big),\\[2ex]
        \bm{U}^{n+1} &= \frac{1}{3} \bm{U}^{n} + \frac{2}{3} \bm{y}_{2} - \frac{2}{3} \widetilde{\bm{M}}^{-1 }\bm{f} \big( \bm{y}_{2} \big),

    where :math:`\widetilde{\bm{M}} = \bm{M} / \delta t` is the weighted mass matrix, :math:`\bm{M}` is the mass matrix and :math:`\bm{f}` arises from the spatial discretization of the PDE.
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

    @time_generator("time level")
    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:

        # Extract the current solution.
        Un = self.root.fem.gfu
        self.U0.data = Un.vec

        # First stage.
        self.blf.Apply(Un.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        Un.vec.data = self.U0 - self.minv * self.rhs

        self.set_stage_time(1, t0)
        yield {'t': self.t.Get(), 'stage': 1}

        # Second stage.
        self.blf.Apply(Un.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec

        # NOTE, avoid 1-liners with dependency on the same read/write data.
        Un.vec.data *= self.alpha21
        Un.vec.data += self.alpha20 * self.U0 - self.beta21 * self.minv * self.rhs

        self.set_stage_time(2, t0)
        yield {'t': self.t.Get(), 'stage': 2}

        # Third stage.
        self.blf.Apply(Un.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec

        # NOTE, avoid 1-liners with dependency on the same read/write data.
        Un.vec.data *= self.alpha32
        Un.vec.data += self.alpha30 * self.U0 - self.beta32 * self.minv * self.rhs

        self.set_stage_time(3, t0)
        yield {'t': self.t.Get(), 'stage': 3}

        if self.is_diverged(self.root.fem.gfu.vec):
            yield {"is_diverged": True}


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
    number_of_stages: int = 4
    time_of_stages: tuple[float] = (0.0, 0.5, 0.5, 1.0)

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

    @time_generator("time level")
    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:

        # First stage.
        self.blf.Apply(self.root.fem.gfu.vec, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        self.K1.data = -self.minv * self.rhs

        self.set_stage_time(1, t0)
        yield {'t': self.t.Get(), 'stage': 1}

        # Second stage.
        self.Us.data = self.root.fem.gfu.vec + self.a21 * self.K1
        self.blf.Apply(self.Us, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        self.K2.data = -self.minv * self.rhs

        self.set_stage_time(2, t0)
        yield {'t': self.t.Get(), 'stage': 2}

        # Third stage.
        self.Us.data = self.root.fem.gfu.vec + self.a32 * self.K2
        self.blf.Apply(self.Us, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        self.K3.data = -self.minv * self.rhs

        self.set_stage_time(3, t0)
        yield {'t': self.t.Get(), 'stage': 3}

        # Fourth stage.
        self.Us.data = self.root.fem.gfu.vec + self.K3
        self.blf.Apply(self.Us, self.rhs)
        if self.lf is not None:
            self.rhs.data -= self.lf.vec
        self.K4.data = -self.minv * self.rhs

        self.set_stage_time(4, t0)
        yield {'t': self.t.Get(), 'stage': 4}

        # Reconstruct the solution at t^{n+1}.
        self.root.fem.gfu.vec.data += self.b1 * self.K1 \
            + self.b2 * self.K2 \
            + self.b3 * self.K3 \
            + self.b4 * self.K4

        if self.is_diverged(self.root.fem.gfu.vec):
            yield {'is_diverged': True}
