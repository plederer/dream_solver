from __future__ import annotations
import enum
import abc
from typing import TYPE_CHECKING
from ngsolve import CF

if TYPE_CHECKING:
    from .formulations import Formulation


class DynamicViscosity(enum.Enum):
    INVISCID = None
    CONSTANT = "constant"
    SUTHERLAND = "sutherland"


def viscosity_factory(formulation: Formulation) -> _DynamicViscosity:

    mu = formulation.cfg.dynamic_viscosity

    if mu is DynamicViscosity.INVISCID:
        return Inviscid(formulation)

    elif mu is DynamicViscosity.CONSTANT:
        return Constant(formulation)

    elif mu is DynamicViscosity.SUTHERLAND:
        return Sutherland(formulation)

    else:
        raise NotImplementedError(f"Viscosity: {mu}")


class _DynamicViscosity(abc.ABC):

    def __init__(self, formulation: Formulation) -> None:
        self.formulation = formulation
        self.solver_configuration = formulation.cfg

    @abc.abstractmethod
    def get(self, U, Q) -> CF: ...

    @abc.abstractmethod
    def get_gradient(self, U, Q) -> CF: ...


class Inviscid(_DynamicViscosity):

    def get(self, U, Q):
        raise ValueError("Invalid for inviscid fluid!")

    def get_gradient(self, U, Q):
        raise ValueError("Invalid for inviscid fluid!")


class Constant(_DynamicViscosity):

    def get(self, U, Q) -> CF:
        return CF((1))

    def get_gradient(self, U, Q) -> CF:
        dim = self.formulation.mesh.dim
        return CF(tuple(0 for dir in range(dim)))


class Sutherland(_DynamicViscosity):

    def __init__(self, formulation: Formulation) -> None:
        self.formulation = formulation
        self.solver_configuration = formulation.cfg

        self._S0 = 110.4
        self._T0 = 293.15
        self._mu_0 = 1.716e-5

    def get(self, U, Q) -> CF:

        gamma = self.formulation.cfg.heat_capacity_ratio
        M = self.formulation.cfg.Mach_number
        T_farfield = self.solver_configuration.farfield_temperature

        T_ = self.formulation.temperature(U)
        S_ = self._S0/(T_farfield * (gamma - 1) * M**2)
        T_farfield_ = 1/((gamma - 1) * M**2)

        mu = (T_/T_farfield_)**(3/2) * (T_farfield_ + S_)/(T_ + S_)

        return mu

    def get_gradient(self, U, Q) -> CF:
        raise NotImplementedError("Currently not implemented")
