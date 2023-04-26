from __future__ import annotations

from ngsolve import *

from .interface import TestAndTrialFunction, VectorIndices, MixedMethods, RiemannSolver
from .primitive import PrimitiveFormulation, Indices
from .. import conditions as co
from .. import viscosity as visc


class PrimitiveFormulation2D(PrimitiveFormulation):

    _indices = Indices(PRESSURE=0, VELOCITY=VectorIndices(X=1, Y=2), TEMPERATURE=3)

    def _initialize_FE_space(self) -> ProductSpace:

        order = self.cfg.order
        mixed_method = self.cfg.mixed_method
        periodic = self.cfg.periodic

        V = L2(self.mesh, order=order)
        VHAT = FacetFESpace(self.mesh, order=order)
        Q = VectorL2(self.mesh, order=order)

        if periodic:
            VHAT = Periodic(VHAT)

        space = V**4 * VHAT**4

        if mixed_method is MixedMethods.NONE:
            pass
        elif mixed_method is MixedMethods.GRADIENT:
            space *= Q**4
        else:
            raise NotImplementedError(f"Mixed method {mixed_method} not implemented for {self}!")

        return space

    def _initialize_TnT(self) -> TestAndTrialFunction:
        return TestAndTrialFunction(*zip(*self.fes.TnT()))

    def primitive_convective_jacobian_x(self, U) -> CF:

        rho = self.density(U)
        c = self.speed_of_sound(U)
        u = self.velocity(U)[0]

        A = CF((
            u, rho, 0, 0,
            0, u, 0, 1/rho,
            0, 0, u, 0,
            0, rho*c**2, 0, u),
            dims=(4, 4))

        return A

    def primitive_convective_jacobian_y(self, U) -> CF:
        rho = self.density(U)
        c = self.speed_of_sound(U)
        v = self.velocity(U)[1]

        B = CF((v, 0, rho, 0,
                0, v, 0, 0,
                0, 0, v, 1/rho,
                0, 0, rho*c**2, v),
               dims=(4, 4))

        return B

    def primitive_convective_jacobian(self, U, unit_vector: CF) -> CF:
        A = self.primitive_convective_jacobian_x(U)
        B = self.primitive_convective_jacobian_y(U)
        return A * unit_vector[0] + B * unit_vector[1]
