from __future__ import annotations

import ngsolve as ngs

from dream import bla
from dream.config import MultipleConfiguration, standard_configuration
from dream.compressible.state import CompressibleState


class RiemannSolver(MultipleConfiguration, is_interface=True):

    def __init__(self, cfg: CompressibleFlowConfiguration = None, **kwargs):
        super().__init__(**kwargs)

        if cfg is None:
            cfg = CompressibleFlowConfiguration()

        self.cfg = cfg

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR) -> ngs.CF:
        NotImplementedError()


class LaxFriedrich(RiemannSolver):

    aliases = ('lf', )

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR) -> ngs.CF:
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.equations.velocity(state)
        c = self.cfg.equations.speed_of_sound(state)

        lambda_max = bla.abs(bla.inner(u, unit_vector)) + c
        return lambda_max * ngs.Id(unit_vector.dim + 2)


class Roe(RiemannSolver):

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR) -> ngs.CF:
        unit_vector = bla.as_vector(unit_vector)

        lambdas = self.cfg.equations.characteristic_velocities(state, unit_vector, type_="absolute")
        return self.cfg.equations.transform_characteristic_to_conservative(bla.diagonal(lambdas), state, unit_vector)


class HLL(RiemannSolver):

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR) -> ngs.CF:
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.equations.velocity(state)
        c = self.cfg.equations.speed_of_sound(state)

        un = bla.inner(u, unit_vector)
        s_plus = bla.max(un + c)

        return s_plus * ngs.Id(unit_vector.dim + 2)


class HLLEM(RiemannSolver):

    @standard_configuration(default=1e-8)
    def theta_0(self, value):
        """ Defines a threshold value used to stabilize contact waves, when the eigenvalue tends to zero.

        This can occur if the flow is parallel to the element or domain boundary!
        """
        return float(value)

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR) -> ngs.CF:
        unit_vector = bla.as_vector(unit_vector)

        u = self.cfg.equations.velocity(state)
        c = self.cfg.equations.speed_of_sound(state)

        un = bla.inner(u, unit_vector)
        un_abs = bla.abs(un)
        s_plus = bla.max(un + c)

        theta = bla.max(un_abs/(un_abs + c), self.theta_0)
        THETA = bla.diagonal([1] + unit_vector.dim * [theta] + [1])
        THETA = self.cfg.equations.transform_characteristic_to_conservative(THETA, state, unit_vector)

        return s_plus * THETA

    theta_0: float
