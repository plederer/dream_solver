""" Definitions of the implicit-explicit (IMEX) time schemes for the scalar transport equation. 
"""
from __future__ import annotations
from dream.config import Integrals, Log
from dream.time import TimeSchemes

import ngsolve as ngs
import typing


class IMEXRKSchemes(TimeSchemes):

    def assemble(self) -> None:

        self.root.io.log.logger.warning(
            "Experimental IMEXRK schemes are currently under development. Use with caution!")

        # This cannot be an inviscid formulation: it must contain inviscid and viscous terms.
        if self.root.is_inviscid is True:
            raise ValueError("IMEXRK Schemes are based on inviscid and viscous operator splitting.")

        condense = self.root.fem.static_condensation

        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.implicit = ngs.BilinearForm(self.root.fem.fes)
        self.explicit = ngs.BilinearForm(self.root.fem.fes)
        self.mass = ngs.BilinearForm(self.root.fem.fes, symmetric=True)
        self.rhs = self.root.fem.gfu.vec.CreateVector()
        self.mu0 = self.root.fem.gfu.vec.CreateVector()

        # Precompute the  mass matrix.
        u, v = self.root.fem.TnT['u']
        self.mass = ngs.BilinearForm(self.root.fem.fes)
        self.mass += ngs.InnerProduct(u, v) * ngs.dx
        self.mass.Assemble()
        self.mass = self.mass.mat

        # TODO: Use matrix-free implementation
        # self.mass = self.root.fem.spaces['u'].Mass(1.0)

        # blf: all spaces, treat convection explicitly (exclude from implicit form).
        blf = self.parse_sum_of_integrals(self.root.fem.blf, exclude_terms=('convection',))
        self.add_sum_of_integrals(self.blf, blf)

        # blfs: stage-previous implicit terms — diffusion only for volume space U.
        diffusion = self.parse_sum_of_integrals(self.root.fem.blf, exclude_terms=('mass', 'convection'))
        self.add_sum_of_integrals(self.implicit, diffusion)

        # blfe: explicit terms — convection only for volume space U.
        convection = self.parse_sum_of_integrals(self.root.fem.blf, exclude_terms=('mass', 'diffusion'))
        self.add_sum_of_integrals(self.explicit, convection)

        self.lfc = None
        self.lfd = None
        if any([space for space, forms in self.root.fem.lf.items() if forms]):
            self.lfc = ngs.LinearForm(self.root.fem.fes)
            self.lfd = ngs.LinearForm(self.root.fem.fes)

            convection = self.parse_sum_of_integrals(self.root.fem.lf, include_terms=('convection',))
            diffusion = self.parse_sum_of_integrals(self.root.fem.lf, include_terms=('diffusion',))

            self.add_sum_of_integrals(self.lfc, convection)
            self.add_sum_of_integrals(self.lfd, diffusion)

            self.lfc.Assemble()
            self.lfd.Assemble()

        self.blf.Assemble()


class IMEXRK_ARS443(IMEXRKSchemes):
    r""" Interface class responsible for configuring an additive 4-stage implicit, 4-stage explicit, 3rd-order implicit-explicit Runge-Kutta scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), see Section 2.8 in :cite:`ascher1997implicit`. Note, currently, the splitting assumes that 

    - Inviscid/convection terms are handled *explicitly*.
    - Viscous/diffusion terms are handled *implicitly*.

    At the continuous level, the splitting is done as such

    .. math::
        \partial_t u + g(u, \nabla u) = -f(u),

    where :math:`g` and :math:`f` are the viscous and inviscid fluxes, respectively. Recall, both spatial fluxes, are discretized assuming they are on the left-hand side of the equality -- hence the -ve sign in front of :math:`f(u)`.

    Discretly, the formulation can be expressed as

    .. math::
        \widetilde{\bm{M}} \bm{y}_{i} + \bm{B}_d \bm{y}_{i} = \widetilde{\bm{M}} \bm{u}^{n} 
                                                            -\frac{1}{a_{ii}} \sum_{j=1}^{i-1} a_{ij} \bm{B}_d \bm{y}_{j}
                                                            -\frac{1}{a_{ii}} \sum_{j=1}^{i} \hat{a}_{i+1,j} \bm{B}_c \bm{y}_{j-1},

    where 

    - :math:`\bm{y}_{0} = \bm{u}^{n}`. 
    - :math:`\bm{y}_{i}` is the solution at the *ith* stage.
    - :math:`\bm{u}^{n+1} = \bm{y}_{s}` since this is a stiffly-accurate method.
    - :math:`\widetilde{\bm{M}} = \frac{1}{a_{ii}\delta t} \int_{D} u v\, d\bm{x}` is the weighted mass matrix and :math:`a_{ii}` is constant (SDIRK).
    - :math:`\bm{B}_{d}` and :math:`\bm{B}_{c}` are the matrices of the diffusion and convection bilinear forms, respectively. See :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms`.
    - :math:`a_{ij}` and :math:`\hat{a}_{ij}` are the *implicit* and *explicit* coefficients, respectively, based on the below Butcher tableau.

    .. math::
        \begin{array}{c|cccc}
                0              & 0 & \phantom{-}0           &  \phantom{-}0           & 0           & 0          \\
            \frac{1}{2}    & 0 & \phantom{-}\frac{1}{2} &  \phantom{-}0           & 0           & 0          \\
                \frac{2}{3}    & 0 & \phantom{-}\frac{1}{6} &  \phantom{-}\frac{1}{2} & 0           & 0          \\
            \frac{1}{2}    & 0 &           -\frac{1}{2} &  \phantom{-}\frac{1}{2} & \frac{1}{2} & 0          \\
            1              & 0 & \phantom{-}\frac{3}{2} &            -\frac{3}{2} & \frac{1}{2} & \frac{1}{2}\\
            \hline
                \mathbf{SDIRK} & 0 & \phantom{-}\frac{3}{2} &            -\frac{3}{2} & \frac{1}{2} & \frac{1}{2}
        \end{array}
        \qquad \qquad
        \begin{array}{c|ccc}
                0            & 0             &  \phantom{-}0            & 0           & \phantom{-}0\\
            \frac{1}{2}  & \frac{1}{2}   &  \phantom{-}0            & 0           & \phantom{-}0\\
                \frac{2}{3}  & \frac{11}{18} &  \phantom{-}\frac{1}{18} & 0           & \phantom{-}0\\
            \frac{1}{2}  & \frac{5}{6}   &            -\frac{5}{6}  & \frac{1}{2} & \phantom{-}0\\
            1            & \frac{1}{4}   &  \phantom{-}\frac{7}{4}  & \frac{3}{4} & -\frac{7}{4}\\
            \hline
                \mathbf{ERK} & \frac{1}{4}   &  \phantom{-}\frac{7}{4}  & \frac{3}{4} & -\frac{7}{4}
        \end{array}

    :note: The SDIRK coefficients are padded (zero on first row/column). In reality their indices ignore the padding, e.g. :math:`a_{21} = 1/6`, :math:`a_{22} = 1/2`.
    """
    name: str = "imex_rk_ars443"

    def initialize_butcher_tableau(self):

        # Implicit RK coefficients.
        self.aii = 1.0/2.0
        self.a21 = 1.0/6.0
        self.a31 = -1.0/2.0
        self.a32 = 1.0/2.0
        self.a41 = 3.0/2.0
        self.a42 = -3.0/2.0
        self.a43 = 1.0/2.0

        self.b1 = self.a41
        self.b2 = self.a42
        self.b3 = self.a43
        self.b4 = self.aii

        self.c1 = 1.0/2.0
        self.c2 = 2.0/3.0
        self.c3 = 1.0/2.0
        self.c4 = 1.0

        # Explicit RK coefficients.
        self.ae21 = 1.0/2.0
        self.ae31 = 11.0/18.0
        self.ae32 = 1.0/18.0
        self.ae41 = 5.0/6.0
        self.ae42 = -5.0/6.0
        self.ae43 = 1.0/2.0
        self.ae51 = 1.0/4.0
        self.ae52 = 7.0/4.0
        self.ae53 = 3.0/4.0
        self.ae54 = -7.0/4.0

        self.be1 = self.ae51
        self.be2 = self.ae52
        self.be3 = self.ae53
        self.be4 = self.ae54

        self.ce2 = self.c1
        self.ce3 = self.c2
        self.ce4 = self.c3
        self.ce5 = self.c4

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.x3 = self.root.fem.gfu.vec.CreateVector()

        self.f1 = self.root.fem.gfu.vec.CreateVector()
        self.f2 = self.root.fem.gfu.vec.CreateVector()
        self.f3 = self.root.fem.gfu.vec.CreateVector()
        self.f4 = self.root.fem.gfu.vec.CreateVector()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['u']

        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['u']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx

    def solve_stage(self, U: ngs.GridFunction, rhs: ngs.BaseVector):
        if self.root.fem.static_condensation is True:
            rhs.data += self.blf.harmonic_extension_trans * rhs
            U.vec.data = self.binv * rhs
            U.vec.data += self.blf.harmonic_extension * U.vec
            U.vec.data += self.blf.inner_solve * rhs
        else:
            U.vec.data = self.binv * rhs

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        # Initial vector: M*U^n + lf.vec.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec
        if self.lfd is not None:
            self.mu0.data += self.lfd.vec  # NOTE, this is hardcoded for DG only.

        # Abbreviation for singly-diagonal coefficients.
        ovaii = 1.0/self.aii

        # Stage: 1.
        self.explicit.Apply(self.root.fem.gfu.vec, self.f1)
        if self.lfc is not None:
            self.f1.data -= self.lfc.vec  # NOTE, this is hardcoded for DG only.

        # Abbreviation for 1st-stage scaled-coefficients.
        ae21 = ovaii*self.ae21

        self.rhs.data = self.mu0 - ae21 * self.f1
        self.solve_stage(self.root.fem.gfu, self.rhs)

        # Stage: 2.
        self.explicit.Apply(self.root.fem.gfu.vec, self.f2)
        if self.lfc is not None:
            self.f2.data -= self.lfc.vec  # NOTE, this is hardcoded for DG only.

        self.implicit.Apply(self.root.fem.gfu.vec, self.x1)
        if self.lfd is not None:
            self.x1.data -= self.lfd.vec  # NOTE, this is hardcoded for DG only.

        # Abbreviation for 2nd-stage scaled-coefficients.
        ae31 = ovaii*self.ae31
        ae32 = ovaii*self.ae32
        a21 = ovaii*self.a21

        self.rhs.data = self.mu0       \
            - ae31 * self.f1 \
            - ae32 * self.f2 \
            - a21 * self.x1
        self.solve_stage(self.root.fem.gfu, self.rhs)

        # Stage: 3.
        self.explicit.Apply(self.root.fem.gfu.vec, self.f3)
        if self.lfc is not None:
            self.f3.data -= self.lfc.vec  # NOTE, this is hardcoded for DG only.

        self.implicit.Apply(self.root.fem.gfu.vec, self.x2)
        if self.lfd is not None:
            self.x2.data -= self.lfd.vec  # NOTE, this is hardcoded for DG only.

        # Abbreviation for 3rd-stage scaled-coefficients.
        ae41 = ovaii*self.ae41
        ae42 = ovaii*self.ae42
        ae43 = ovaii*self.ae43
        a31 = ovaii*self.a31
        a32 = ovaii*self.a32

        self.rhs.data = self.mu0       \
            - ae41 * self.f1 \
            - ae42 * self.f2 \
            - ae43 * self.f3 \
            - a31 * self.x1 \
            - a32 * self.x2
        self.solve_stage(self.root.fem.gfu, self.rhs)

        # Stage: 4.
        self.explicit.Apply(self.root.fem.gfu.vec, self.f4)
        if self.lfc is not None:
            self.f4.data -= self.lfc.vec  # NOTE, this is hardcoded for DG only.

        self.implicit.Apply(self.root.fem.gfu.vec, self.x3)
        if self.lfd is not None:
            self.x3.data -= self.lfd.vec  # NOTE, this is hardcoded for DG only.

        # Abbreviation for 4th-stage scaled-coefficients.
        ae51 = ovaii*self.ae51
        ae52 = ovaii*self.ae52
        ae53 = ovaii*self.ae53
        ae54 = ovaii*self.ae54
        a41 = ovaii*self.a41
        a42 = ovaii*self.a42
        a43 = ovaii*self.a43

        self.rhs.data = self.mu0       \
            - ae51 * self.f1 \
            - ae52 * self.f2 \
            - ae53 * self.f3 \
            - ae54 * self.f4 \
            - a41 * self.x1  \
            - a42 * self.x2  \
            - a43 * self.x3
        self.solve_stage(self.root.fem.gfu, self.rhs)

        yield {}
