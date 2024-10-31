from __future__ import annotations
import typing
import ngsolve as ngs

from dream import bla
from dream.config import MultipleConfiguration, any
from dream.compressible.config import CompressibleState

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


class RiemannSolver(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def equations(self):
        return self.cfg.pde.equations

    def get_convective_stabilisation_matrix(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        NotImplementedError()


class Upwind(RiemannSolver):

    name = "upwind"

    def get_convective_stabilisation_matrix(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        unit_vector = bla.as_vector(unit_vector)
        return self.equations.get_conservative_convective_jacobian(U, unit_vector, 'outgoing')


class LaxFriedrich(RiemannSolver):

    name = "lax_friedrich"
    aliases = ('lf', )

    def get_convective_stabilisation_matrix(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        unit_vector = bla.as_vector(unit_vector)

        u = self.equations.velocity(U)
        c = self.equations.speed_of_sound(U)

        lambda_max = bla.abs(bla.inner(u, unit_vector)) + c
        return lambda_max * ngs.Id(unit_vector.dim + 2)


class Roe(RiemannSolver):

    name = "roe"

    def get_convective_stabilisation_matrix(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        unit_vector = bla.as_vector(unit_vector)

        lambdas = self.equations.characteristic_velocities(U, unit_vector, type_="absolute")
        return self.equations.transform_characteristic_to_conservative(bla.diagonal(lambdas), U, unit_vector)


class HLL(RiemannSolver):

    name = "hll"

    def get_convective_stabilisation_matrix(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        unit_vector = bla.as_vector(unit_vector)

        u = self.equations.velocity(U)
        c = self.equations.speed_of_sound(U)

        un = bla.inner(u, unit_vector)
        s_plus = bla.max(un + c)

        return s_plus * ngs.Id(unit_vector.dim + 2)


class HLLEM(RiemannSolver):

    name = "hllem"

    @any(default=1e-8)
    def theta_0(self, value):
        """ Defines a threshold value used to stabilize contact waves, when the eigenvalue tends to zero.

        This can occur if the flow is parallel to the element or domain boundary!
        """
        return float(value)

    def get_convective_stabilisation_matrix(self, U: CompressibleState, unit_vector: bla.VECTOR) -> ngs.CF:
        unit_vector = bla.as_vector(unit_vector)

        u = self.equations.velocity(U)
        c = self.equations.speed_of_sound(U)

        un = bla.inner(u, unit_vector)
        un_abs = bla.abs(un)
        s_plus = bla.max(un + c)

        theta = bla.max(un_abs/(un_abs + c), self.theta_0)
        THETA = bla.diagonal([1] + unit_vector.dim * [theta] + [1])
        THETA = self.equations.transform_characteristic_to_conservative(THETA, U, unit_vector)

        return s_plus * THETA

    theta_0: float
