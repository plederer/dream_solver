r""" Dimensionless incompressible Navier-Stokes equations

We consider the dimensionless incompressible Navier-Stokes equations

.. math::
    \frac{\partial \vec{u}}{\partial t} + \vec{u} \cdot \nabla \vec{u} - \frac{1}{Re} \div{(\mat{\tau})} + \nabla p = 0

"""
from __future__ import annotations

import ngsolve as ngs
from dream.config import (dream_configuration,
                          ngsdict,
                          quantity)
from dream.mesh import (Condition,
                        Periodic,
                        Initial)


class flowfields(ngsdict):

    u = quantity('velocity', r"$u$")
    p = quantity('pressure', r"$p$")
    tau = quantity('deviatoric_stress_tensor', r"$\tau$")
    eps = quantity('strain_rate_tensor', r"\varepsilon")
    grad_u = quantity('velocity_gradient', r"\nabla u")


# Define conditions


class Inflow(Condition):

    name: str = "inflow"

    def __init__(self, velocity: flowfields | ngs.CF):
        self.velocity = velocity
        super().__init__()

    @dream_configuration
    def velocity(self) -> flowfields:
        """ Returns the inflow velocity """
        return self._velocity

    @velocity.setter
    def velocity(self, velocity: ngsdict) -> None:
        if isinstance(velocity, flowfields):
            self._velocity = velocity.u
        elif isinstance(velocity, ngs.CF):
            self._velocity = velocity
        else:
            self._velocity = ngs.CF(tuple(velocity))


class Outflow(Condition):

    name: str = "outflow"


class Wall(Condition):

    name: str = "wall"


class Force(Condition):

    name: str = "force"

    def __init__(self, force: flowfields | ngs.CF):
        self.force = force
        super().__init__()

    @dream_configuration
    def force(self) -> flowfields:
        """ Returns the force vector """
        return self._force

    @force.setter
    def force(self, force: ngsdict) -> None:
        if isinstance(force, flowfields):
            self._force = force.u
        elif isinstance(force, ngs.CF):
            self._force = force
        else:
            self._force = ngs.CF(tuple(force))


BCS = [Inflow, Outflow, Wall, Periodic]
DCS = [Initial, Force]
