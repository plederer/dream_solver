r""" Definitions of viscous constitutive relations for compressible flow 

This module defines the dynamic viscosity and heat conductivity for the 
compressible flow solvers, which are used to compute the viscous stress 
tensor and the heat flux in the Navier-Stokes equations.

We derive the dimensionless viscous stress tensor :math:`\mat{\tau}` and 
heat flux :math:`\vec{q}` from the dimensional one, namely:

.. math::

    \overline{\mat{\tau}} &= 2 \overline{\mu} \left[ \frac{\overline{\nabla} \overline{\vec{u}} + (\overline{\nabla} \overline{\vec{u}})^\T}{2} - \frac{1}{3} \overline{\div}(\overline{\vec{u}}) \right], \\
    \rho_{ref} u^2_{ref} \mat{\tau} &= 2\mu \overline{\mu}_\infty  \frac{u_{ref}}{L_{ref}} \left[ \frac{\nabla \vec{u} + (\nabla \vec{u})^\T}{2} - \frac{1}{3} \div(\vec{u}) \right], \\
    \mat{\tau} &= 2 \mu \frac{\overline{\mu}_\infty}{\rho_{ref} u_{ref} L_{ref}} \left[ \frac{\nabla \vec{u} + (\nabla \vec{u})^\T}{2} - \frac{1}{3} \div(\vec{u}) \right], \\
    \mat{\tau} &= \frac{2 \mu}{\Re_{ref}} \left[ \frac{\nabla \vec{u} + (\nabla \vec{u})^\T}{2} - \frac{1}{3} \div(\vec{u}) \right],

and

.. math::

    \overline{\vec{q}} &= -\overline{k} \overline{\nabla} \overline{T}, \\
    \rho_{ref} u^3_{ref} \vec{q} &= -\overline{k}_\infty \frac{T_{ref}}{L_{ref}} k \nabla T, \\
    \vec{q} &= -\frac{\overline{k}_\infty}{\overline{\mu}_\infty} \frac{T_{ref}}{u^2_{ref}} \frac{\overline{\mu}_\infty}{\rho_{ref} u_{ref}L_{ref}} k \nabla T, \\
    \vec{q} &= -\frac{\overline{k}_\infty}{\overline{\mu}_\infty c_p} \frac{\overline{\mu}_\infty}{\rho_{ref} u_{ref}L_{ref}} k \nabla T, \\
    \vec{q} &= -\frac{k}{\Pr_\infty \Re_{ref}} \nabla T.

To set the dimensionless dynamic viscosity :math:`\mu` and heat 
conductivity :math:`k`, the following definitions are available:

Inviscid
    .. math::
        \mu := 0, \quad k := 0.

Constant viscosity and heat conductivity
    .. math::
        \mu := 1, \quad k:= 1.

Sutherland viscosity
    The dimensionless dynamic viscosity :math:`\mu` is derived from the dimensional one as follows:

    .. math::
        \overline{\mu}(\overline{T}) &= \overline{\mu}_\infty \left( \frac{\overline{T}}{\overline{T}_\infty} \right)^{3/2} \frac{\overline{T}_\infty + \overline{S}}{\overline{T} + \overline{S}}, \\
        \overline{\mu}_\infty \mu(\overline{T}) &= \overline{\mu}_\infty \left( \frac{\overline{T}}{\overline{T}_\infty} \right)^{3/2} \frac{T_{ref}}{T_{ref}} \frac{\overline{T}_\infty + \overline{S}}{\overline{T} + \overline{S}}, \\
        \mu(T) &= \left( \frac{T}{T_\infty} \right)^{3/2} \frac{T_\infty + S}{T + S}.

    with :math:`S = \overline{S}/T_{ref}`.

    For the heat conductivity :math:`k` the same relation holds:

    .. math::
        k(T) = \left( \frac{T}{T_\infty} \right)^{3/2} \frac{T_\infty + S}{T + S}.

See :class:`dream.compressible.scaling` for the definition of the reference Reynolds number :math:`\Re_{ref}`.
"""
from __future__ import annotations
import typing

from dream.config import Configuration, dream_configuration
from dream.compressible.config import flowfields

if typing.TYPE_CHECKING:
    from .solver import CompressibleFlowSolver


class DynamicViscosity(Configuration, is_interface=True):

    root: CompressibleFlowSolver

    @property
    def is_inviscid(self) -> bool:
        return isinstance(self, Inviscid)

    def viscosity(self, U: flowfields):
        raise NotImplementedError()


class Inviscid(DynamicViscosity):

    name: str = "inviscid"

    def viscosity(self, U: flowfields):
        """ Returns the dynamic viscosity and heat conductivity for an inviscid flow. 

        .. math::
            \mu = k = 0

        :raise: TypeError 
            If the dynamic viscosity is requested in an inviscid setting.
        """
        raise TypeError("Inviscid Setting! Dynamic Viscosity not defined!")


class Constant(DynamicViscosity):

    name: str = "constant"

    def viscosity(self, U: flowfields):
        """ Returns the dynamic viscosity and heat conductivity for a constant viscosity flow.

        .. math::
            \mu = k = 1
        """
        return 1


class Sutherland(DynamicViscosity):

    name: str = "sutherland"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "sutherland_temperature": 110.4,
        }

        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def sutherland_temperature(self) -> float:
        """ The Sutherland temperature :math:`\overline{S}` in Kelvin.

            :getter: Returns the Sutherland temperature.
            :setter: Sets the Sutherland temperature. Defaults to :math:`110.4 \, K`
        """
        return self._sutherland_temperature

    @sutherland_temperature.setter
    def sutherland_temperature(self, value: float) -> None:
        self._sutherland_temperature = value

    def viscosity(self, U: flowfields):
        r""" Returns the dynamic viscosity and heat conductivity for a Sutherland flow.

        .. math::
            \mu(T) = k(T) = \left( \frac{T}{T_\infty} \right)^{3/2} \frac{T_\infty + S}{T + S}
        """

        if U.T is not None:
            Tinf = self.root.scaling.temperature
            T0 = self.sutherland_temperature/self.root.scaling.reference_temperature

            return (U.T/Tinf)**(3/2) * (Tinf + T0)/(U.T + T0)
