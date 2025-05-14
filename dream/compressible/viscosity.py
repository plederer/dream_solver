""" Definitions of viscous constitutive relations for compressible flow """
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
        raise TypeError("Inviscid Setting! Dynamic Viscosity not defined!")


class Constant(DynamicViscosity):

    name: str = "constant"

    def viscosity(self, U: flowfields):
        return 1


class Sutherland(DynamicViscosity):

    name: str = "sutherland"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "measurement_temperature": 110.4,
            "measurement_viscosity": 1.716e-5,
        }

        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def measurement_temperature(self) -> float:
        return self._measurement_temperature

    @measurement_temperature.setter
    def measurement_temperature(self, value: float) -> None:
        self._measurement_temperature = value

    @dream_configuration
    def measurement_viscosity(self) -> float:
        return self._measurement_viscosity

    @measurement_viscosity.setter
    def measurement_viscosity(self, value: float) -> None:
        self._measurement_viscosity = value

    def viscosity(self, U: flowfields):

        if U.T is not None:

            INF = flowfields(rho=self.root.scaling.density(), c=self.root.scaling.speed_of_sound(self.root.mach_number))
            INF.T = self.root.temperature(INF)

            T0 = self.measurement_temperature/self.root.scaling.dimensionful_values.T * INF.T

            return (U.T/INF.T)**(3/2) * (INF.T + T0)/(U.T + T0)
