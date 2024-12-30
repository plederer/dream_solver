from __future__ import annotations
import typing
import ngsolve as ngs

from dream import bla
from dream.config import InterfaceConfiguration, configuration
from dream.compressible.config import flowstate

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


class RiemannSolver(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def get_convective_stabilisation_matrix(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        NotImplementedError()


class Upwind(RiemannSolver):

    name = "upwind"

    def get_convective_stabilisation_matrix(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
            .. math::
                \bm{\tau}_c := \bm{A}_n^+
        """
        unit_vector = bla.as_vector(unit_vector)
        return self.cfg.pde.get_conservative_convective_jacobian(U, unit_vector, 'outgoing')


class LaxFriedrich(RiemannSolver):

    name = "lax_friedrich"
    aliases = ('lf', )

    def get_convective_stabilisation_matrix(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
            .. math::
                \bm{\tau}_c := (|u_n| + c) \bm{I}
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.pde.velocity(U)
        c = self.cfg.pde.speed_of_sound(U)

        lambda_max = bla.abs(bla.inner(u, unit_vector)) + c
        return lambda_max * ngs.Id(unit_vector.dim + 2)


class Roe(RiemannSolver):

    name = "roe"

    def get_convective_stabilisation_matrix(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
            .. math::
                \bm{\tau}_c := |A_n|
        """
        unit_vector = bla.as_vector(unit_vector)

        lambdas = self.cfg.pde.characteristic_velocities(U, unit_vector, "absolute")
        return self.cfg.pde.transform_characteristic_to_conservative(bla.diagonal(lambdas), U, unit_vector)


class HLL(RiemannSolver):

    name = "hll"

    def get_convective_stabilisation_matrix(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
            .. math::
                \bm{\tau}_c := \max(u_n + c, 0) \bm{I}
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.pde.velocity(U)
        c = self.cfg.pde.speed_of_sound(U)

        un = bla.inner(u, unit_vector)
        s_plus = bla.max(un + c)

        return s_plus * ngs.Id(unit_vector.dim + 2)


class HLLEM(RiemannSolver):

    name = "hllem"

    @configuration(default=1e-8)
    def theta_0(self, value):
        r""" Defines a threshold value used to stabilize contact waves, when the eigenvalue tends to zero.
        This can occur if the flow is parallel to the element or domain boundary!

        .. math::
            \theta_0
        """
        return float(value)
    
    theta_0: float

    def get_convective_stabilisation_matrix(self, U: flowstate, unit_vector: bla.VECTOR) -> ngs.CF:
        r""" Returns the convective stabilisation matrix for the upwind scheme.
        
            .. math::
                \begin{align*}
                \theta &:= \max\left(\frac{|u_n|}{|u_n| + c}, \theta_0\right), &
                \Theta &:=  \text{diag}(1, \theta, \ldots, \theta, 1), &
                \bm{\tau}_c &:= \bm{P} \bm{\Theta} \bm{P}^{-1}
                \end{align*}
        """
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.pde.velocity(U)
        c = self.cfg.pde.speed_of_sound(U)

        un = bla.inner(u, unit_vector)
        un_abs = bla.abs(un)
        s_plus = bla.max(un + c)

        theta = bla.max(un_abs/(un_abs + c), self.theta_0)
        THETA = bla.diagonal([1] + unit_vector.dim * [theta] + [1])
        THETA = self.cfg.pde.transform_characteristic_to_conservative(THETA, U, unit_vector)

        return s_plus * THETA

