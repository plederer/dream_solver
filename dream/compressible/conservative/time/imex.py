""" Definitions of IMEX time integration schemes for conservative methods """
from __future__ import annotations

import ngsolve as ngs
import typing
from dream.time import TimeSchemes
from dream.config import Integrals, Log


class IMEXRKSchemes(TimeSchemes):

    def assemble(self) -> None:

        condense = self.root.fem.static_condensation

        # NOTE, we assume that self.lf is not needed here (for efficiency).
        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.blfs = ngs.BilinearForm(self.root.fem.fes)
        self.blfe = ngs.BilinearForm(self.root.fem.fes)
        self.mass = ngs.BilinearForm(self.root.fem.fes, symmetric=True)
        self.rhs = self.root.fem.gfu.vec.CreateVector()
        self.mu0 = self.root.fem.gfu.vec.CreateVector()

        # Check that a mass matrix is defined in the bilinear form dictionary.
        if "mass" not in self.root.fem.blf['U']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        # Precompute the weighted mass matrix, with weights: 1/(dt*aii).
        self.mass += self.root.fem.blf['U']['mass']

        # Assemble the mass matrix once.
        self.mass.Assemble()

        # The first assumption is that this cannot be an inviscid formulation: it must have inviscid + viscous terms.
        if self.root.dynamic_viscosity.is_inviscid is True:
            raise ValueError("IMEXRK Schemes are based on inviscid and viscous operator splitting.")

        # We have only two possibilities for the IMEXRK splitting: pure HDG or hybrid DG-HDG.
        if self.root.fem.method.name == "dg_hdg":

            # For the hybrid DG-HDG scheme, we only need to skip the 'convection' term in blf,
            # since everything else is handled within the DG_HDG class.
            integrals = self.parse_sum_of_integrals(self.root.fem.blf, exclude_terms=('convection',))
            self.add_sum_of_integrals(self.blf, integrals, "imex bilinear form")
        
        elif self.root.fem.method.name == "hdg":

            # This is a pure HDG-IMEX scheme, we only skip the volume convection term in
            # the volume equations (tested by V), while we retain the inviscid terms in the
            # facet equations (tested by Vhat) which are computed implicitly. In the below,
            # we handle the splitting manually, instead of calling add_sum_of_integrals.
            # The reason is we retain an implicit treatment of the 'convection' term on the
            # facets (Uhat-space), but treat the 'convection' term explicitly on the volume (U-space).

            # Determine which spaces to iterate over.

            integrals = self.parse_sum_of_integrals(self.root.fem.blf, include_spaces=['U'], exclude_terms=('convection',))
            integrals.update(self.parse_sum_of_integrals(self.root.fem.blf, exclude_spaces=['U']))
            self.add_sum_of_integrals(self.blf, integrals, "imex bilinear form")

        else:
            # As of now, only HDG and DG-HDG discretizations are possible with IMEXRK schemes.
            raise ValueError("IMEXRK currently can be used either with HDG or DG-HDG discretizations.")

        # Skip the mass matrix and convection contribution in blfs and only use the space for "U".
        integrals = self.parse_sum_of_integrals(self.root.fem.blf, include_spaces=['U'], exclude_terms=('mass', 'convection'))
        self.add_sum_of_integrals(self.blfs, integrals, "imex bilinear form for splitting")

        # Add only the convection part in blfe, as this is handled explicitly in time.
        integrals = self.parse_sum_of_integrals(self.root.fem.blf, include_spaces=['U'], exclude_terms=('mass', 'diffusion'))
        self.add_sum_of_integrals(self.blfe, integrals, "imex bilinear form for explicit convection")

        # Initialize the nonlinear solver here. Notice, it uses a reference to blf, rhs and gfu.
        self.root.fem.solver.initialize_nonlinear_routine(self.blf, self.root.fem.gfu, self.rhs)


class IMEXRK_ARS443(IMEXRKSchemes):

    name: str = "imex_rk_ars443"
    time_levels = ('n', 'n+1')

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

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        gfus['n+1'] = u

        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['U']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec

        # Abbreviation.
        ovaii = 1.0/self.aii

        # Stage: 1.
        self.blfe.Apply(self.root.fem.gfu.vec, self.f1)

        ae21 = ovaii*self.ae21

        self.rhs.data = self.mu0       \
            - ae21 * self.f1
        for log in self.root.fem.solver.solve_nonlinear_system():
            yield {'stage': 1, **log}

        # Stage: 2.
        self.blfe.Apply(self.root.fem.gfu.vec, self.f2)
        self.blfs.Apply(self.root.fem.gfu.vec, self.x1)

        ae31 = ovaii*self.ae31
        ae32 = ovaii*self.ae32
        a21 = ovaii*self.a21

        self.rhs.data = self.mu0       \
            - ae31 * self.f1 \
            - ae32 * self.f2 \
            - a21 * self.x1
        for log in self.root.fem.solver.solve_nonlinear_system():
            yield {'stage': 2, **log}

        # Stage: 3.
        self.blfe.Apply(self.root.fem.gfu.vec, self.f3)
        self.blfs.Apply(self.root.fem.gfu.vec, self.x2)

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
        for log in self.root.fem.solver.solve_nonlinear_system():
            yield {'stage': 3, **log}

        # Stage: 4.
        self.blfe.Apply(self.root.fem.gfu.vec, self.f4)
        self.blfs.Apply(self.root.fem.gfu.vec, self.x3)

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
            - a41 * self.x1 \
            - a42 * self.x2 \
            - a43 * self.x3
        for log in self.root.fem.solver.solve_nonlinear_system():
            yield {'stage': 4, **log}

        # NOTE,
        # No need to explicitly update the gfu, since the last stage
        # corresponds to the value at time: t^{n+1}.
