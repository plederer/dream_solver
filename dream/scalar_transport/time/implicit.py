""" Definitions of implicit time marching schemes for a scalar transport equation. """
from __future__ import annotations
from dream.config import Integrals, Log
from dream.time import TimeSchemes

import ngsolve as ngs
import typing


class ImplicitSchemes(TimeSchemes):

    def assemble(self) -> None:

        condense = self.root.fem.static_condensation
        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.add_sum_of_integrals(self.blf, self.root.fem.blf, 'implicit bilinear form')
        self.blf.Assemble()

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

        # TODO: Use matrix-free implementation
        # self.mass = self.root.fem.spaces['u'].Mass(1.0)

        # Can be avoided, but for readability and given the simplicity of the PDE, we allocate a rhs anyway.
        self.rhs = self.root.fem.gfu.vec.CreateVector()

    def solve_stage(self, stage: int, t0: float) -> typing.Generator[Log, None, None]:

        if stage != 1:
            raise TypeError(f"Stage {stage} does not exist.")

        for log in self.solve_current_time_level(t0):
            log['stage'] = stage
            yield log


class ImplicitEuler(ImplicitSchemes):
    r""" Class responsible for implementing an implicit (backwards-)Euler time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). Namely,

    .. math::
        \widetilde{\bm{M}} \bm{u}^{n+1} + \bm{B} \bm{u}^{n+1} = \widetilde{\bm{M}} \bm{u}^{n},

    where :math:`\widetilde{\bm{M}} = \frac{1}{\delta t} \int_{D} u v\, d\bm{x}` is the modified mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "implicit_euler"
    number_of_steps: int = 2
    number_of_stages: int = 1
    time_of_stages: tuple[float] = (0.0, 1.0)

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        u, v = self.root.fem.TnT['u']
        blf['u']['time'] = ngs.InnerProduct(u/self.dt, v) * ngs.dx

    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:
        self.t.Set(t0 + self.dt.Get())

        # Compute the right-hand side, first.
        self.rhs.data = 1/self.dt.Get() * self.mass * self.root.fem.gfu.vec

        if self.lf is not None:
            self.rhs.data -= self.lf.vec

        self.root.fem.solver.solve_linear_system(self.blf, self.root.fem.gfu, self.rhs, operator="=")

        yield {'t': self.t.Get()}


class BDF2(ImplicitSchemes):
    r""" Class responsible for implementing an implicit 2nd-order backward differentiation formula that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), using also the previous solution (:math:`t = t^{n-1}`). Namely,

    .. math::
        \widetilde{\bm{M}} \bm{u}^{n+1} + \bm{B} \bm{u}^{n+1} = \widetilde{\bm{M}} \Big( \frac{4}{3} \bm{u}^{n} - \frac{1}{3} \bm{u}^{n-1}\Big),

    where :math:`\widetilde{\bm{M}} = \frac{3}{2\delta t} \int_{D} u v\, d\bm{x}` is the weighted mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "bdf2"
    number_of_steps: int = 3
    number_of_stages: int = 1
    time_of_stages: tuple[float] = (0.0, 1.0)

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        u, v = self.root.fem.TnT['u']
        blf['u']['time'] = ngs.InnerProduct(1.5*u/self.dt, v) * ngs.dx

    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:
        self.t.Set(t0 + self.dt.Get())

        un = self.gfus['u']['n']
        un_1 = self.gfus['u']['n-1']

        # Compute the right-hand side, first.
        self.rhs.data = 0.5/self.dt.Get() * self.mass * (4*un.vec - un_1.vec)

        if self.lf is not None:
            self.rhs.data -= self.lf.vec

        self.root.fem.solver.solve_linear_system(self.blf, self.root.fem.gfu, self.rhs, operator="=")

        yield {'t': self.t.Get()}


class DIRKSchemes(TimeSchemes):
    r""" Interface class responsible for configuring a generic diagonally-implicit Runge-Kutta scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), using s-stages. Namely,

    .. math::
        \widetilde{\bm{M}} \bm{u}^{n+1}                        &= \widetilde{\bm{M}} \bm{u}^{n}     - \frac{1}{a_{ii}} \sum_{i=1}^{s}   b_i    \bm{B} \bm{y}_{i},\\
        \widetilde{\bm{M}}_{i} \bm{y}_{i} + \bm{B} \bm{y}_{i}  &= \widetilde{\bm{M}}_{i} \bm{u}^{n} - \frac{1}{a_{ii}} \sum_{j=1}^{i-1} a_{ij} \bm{B} \bm{y}_{j},

    where 

    - :math:`\widetilde{\bm{M}}_{i} = \frac{1}{a_{ii}\delta t} \int_{D} u v\, d\bm{x}` is the *ith* stage-weighted mass matrix.
    - :math:`\widetilde{\bm{M}}` is based on :math:`a_{ii}=1`.
    - :math:`\bm{y}_{i}` is the solution at the *ith* stage.
    - :math:`a_{ij}` and :math:`b_{i}` are taken from the Butcher of a specific scheme.

    Finally, :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """

    def assemble(self) -> None:

        condense = self.root.fem.static_condensation

        # NOTE, we assume that self.lf is not needed here (for efficiency).
        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.blfs = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.mass = ngs.BilinearForm(self.root.fem.fes, symmetric=True)
        self.rhs = self.root.fem.gfu.vec.CreateVector()
        self.mu0 = self.root.fem.gfu.vec.CreateVector()

        # Check that a mass matrix is defined in the bilinear form dictionary.
        if "mass" not in self.root.fem.blf['u']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        # Precompute the weighted mass matrix, with weights: 1/(dt*aii).
        self.mass += self.root.fem.blf['u']['mass']

        # Assemble the mass matrix once.
        self.mass.Assemble()

        # Add both spatial and mass-matrix terms in blf.
        self.add_sum_of_integrals(self.blf, self.root.fem.blf, 'implicit bilinear form')

        # Skip the mass matrix contribution in blfs and only use the space for "U".
        integrals = self.parse_sum_of_integrals(self.root.fem.blf, include_spaces=['u'], exclude_terms=('mass',))
        self.add_sum_of_integrals(self.blfs, integrals, "implicit bilinear form for splitting")

        self.blf.Assemble()

    # Generic function to store (S-)DIRK coefficient, vector pairs.
    def setup_stage_definitions(self,
                                stage_data: list[tuple[typing.Any, list[tuple[float, typing.Any]]]]
                                ) -> None:
        r"""
        stage_data:
            list of (apply_target, coeffs) for each stage, starting at stage 1.
            apply_target: vector to pass to Apply() or None.
            coeffs: list of (coeff, vector) pairs to sum into rhs.
        """
        self.stage_definitions = \
            {
                i + 1: {"apply_target": apply_target, "coeffs": coeffs}
                for i, (apply_target, coeffs) in enumerate(stage_data)
            }

    # Generic function that solves a DIRK scheme.
    def solve_stage(self, stage: int, t0: float) -> typing.Generator[Log, None, None]:

        # Extract the stage information, if possible.
        try:
            stage_info = self.stage_definitions[stage]
        except KeyError:
            raise TypeError(f"Stage {stage} does not exist.")

        # Stage 1 is a special case, we handle it separately.
        if stage == 1:
            self.mu0.data = self.mass.mat * self.root.fem.gfu.vec

        # Distinguish between singly-diagonal (SDIRK) and variable (DIRK) coefficients.
        if self.variable_aii is not None:
            self.aii.Set(self.variable_aii[stage])  # should be set before the solving routine.
            ovaii = 1.0/self.variable_aii[stage]
            self.rhs.data = ovaii * self.mu0
        else:
            self.rhs.data = self.mu0

        # Compute previous-stage residuals, if the apply_target exists.
        if stage_info["apply_target"] is not None:
            self.set_stage_time(stage - 1, t0)
            self.blfs.Apply(self.root.fem.gfu.vec, stage_info["apply_target"])

        # Build the right-hand side, scaled with its respective coefficients.
        for aij, xj in stage_info["coeffs"]:
            self.rhs.data += aij * xj

        self.set_stage_time(stage, t0)
        # Solve the resulting nonlinear system.

        self.root.fem.solver.solve_linear_system(self.blf, self.root.fem.gfu, self.rhs, operator="=")
        yield {'t': self.t.Get(), 'stage': stage}

    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:
        for i in range(1, self.number_of_stages + 1):
            yield from self.solve_stage(i, t0)


class SDIRK22(DIRKSchemes):
    r""" Updates the solution via a 2-stage 2nd-order (stiffly-accurate) singly diagonally-implicit Runge-Kutta (SDIRK). Taken from Section 2.6 in :cite:`ascher1997implicit`. Its corresponding Butcher tableau is:

    .. math::
        \begin{array}{c|cc}
                \alpha & \alpha     & 0      \\
                1    &   1 - \alpha & \alpha \\
            \hline
                     & 1 - \alpha    & \alpha
        \end{array}

    where :math:`\alpha = (2 - \sqrt{2})/2`.

    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{u}^{n+1} = \bm{y}_{2}`.
    """
    name: str = "sdirk22"
    number_of_stages: int = 2
    time_of_stages: tuple[float] = (0.0, 1.0 - ngs.sqrt(2.0)/2.0, 1.0)

    def initialize_butcher_tableau(self):

        alpha = ngs.sqrt(2.0)/2.0

        self.aii = 1.0 - alpha
        self.a21 = alpha

        # Time stamps for the stage values between t = [n,n+1].
        self.c = [1.0 - alpha, 1.0]

        # This is possible, because the method is stiffly-accurate.
        self.b1 = self.a21
        self.b2 = self.aii

    def configure_scheme(self) -> None:

        # This is an SDIRK scheme, aii is constant.
        self.variable_aii = None

        # Scale the coefficients by their (-ve) diagonal counterpart.
        a21 = -self.a21 / self.aii
        self.setup_stage_definitions([
            (None, []),                 # stage 1
            (self.x1, [(a21, self.x1)])  # stage 2
        ])

    def assemble(self) -> None:
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()

        # Initialize the multistage configuration parameters for the scheme.
        self.configure_scheme()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        u, v = self.root.fem.TnT['u']

        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['u']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx


class SDIRK33(DIRKSchemes):
    r""" Updates the solution via a 3-stage 3rd-order (stiffly-accurate) singly diagonally-implicit Runge-Kutta (SDIRK). Taken from Section 2.7 in :cite:`ascher1997implicit`. Its corresponding Butcher tableau is: 

    .. math::
        \begin{array}{c|ccc}
                0.4358665215 & 0.4358665215 &  0            & 0            \\
                0.7179332608 & 0.2820667392 &  \phantom{-}0.4358665215 & 0 \\
            1            & 1.2084966490 & -0.6443631710 & 0.4358665215 \\
            \hline
                             & 1.2084966490 & -0.6443631710 & 0.4358665215
        \end{array}

    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{u}^{n+1} = \bm{y}_{3}`.
    """
    name: str = "sdirk33"
    number_of_stages: int = 3
    time_of_stages: tuple[float] = (0.0, 0.4358665215, 0.7179332608, 1.0)

    def initialize_butcher_tableau(self):

        self.aii = 0.4358665215
        self.a21 = 0.2820667392
        self.a31 = 1.2084966490
        self.a32 = -0.6443631710

        # Time stamps for the stage values between t = [n,n+1].
        self.c = [0.4358665215, 0.7179332608, 1.0]

        # This is possible, because the method is stiffly-accurate.
        self.b1 = self.a31
        self.b2 = self.a32
        self.b3 = self.aii

    def configure_scheme(self) -> None:

        # This is an SDIRK scheme, aii is constant.
        self.variable_aii = None

        # Scale the coefficients by their (-ve) diagonal counterpart.
        a21 = -self.a21 / self.aii
        a31 = -self.a31 / self.aii
        a32 = -self.a32 / self.aii
        self.setup_stage_definitions([
            (None, []),                                 # stage 1
            (self.x1, [(a21, self.x1)]),                # stage 2
            (self.x2, [(a31, self.x1), (a32, self.x2)])  # stage 3
        ])

    def assemble(self) -> None:
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()

        # Initialize the multistage configuration parameters for the scheme.
        self.configure_scheme()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['u']
        gfus = self.gfus['u'].copy()
        gfus['n+1'] = u

        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['u']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx
