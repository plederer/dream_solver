from __future__ import annotations
import typing

from dream.config import InterfaceConfiguration, configuration
from dream.compressible.config import flowstate

if typing.TYPE_CHECKING:
    from .solver import CompressibleFlowSolver


class DynamicViscosity(InterfaceConfiguration, is_interface=True):

    cfg: CompressibleFlowSolver

    @property
    def is_inviscid(self) -> bool:
        return isinstance(self, Inviscid)

    def viscosity(self, U: flowstate):
        raise NotImplementedError()


class Inviscid(DynamicViscosity):

    name: str = "inviscid"
    aliases = ('euler', )

    def viscosity(self, U: flowstate):
        raise TypeError("Inviscid Setting! Dynamic Viscosity not defined!")


class Constant(DynamicViscosity):

    name: str = "constant"

    def viscosity(self, U: flowstate):
        return 1


class Sutherland(DynamicViscosity):

    name: str = "sutherland"

    @configuration(default=110.4)
    def measurement_temperature(self, value: float) -> float:
        return value

    @configuration(default=1.716e-5)
    def measurement_viscosity(self, value: float) -> float:
        return value

    def viscosity(self, U: flowstate):

        if U.T is not None:

            INF = flowstate(rho=self.cfg.scaling.density(), c=self.cfg.scaling.speed_of_sound(self.cfg.mach_number))
            INF.T = self.cfg.temperature(INF)

            T0 = self.measurement_temperature/self.cfg.scaling.dimensionful_values.T * INF.T

            return (U.T/INF.T)**(3/2) * (INF.T + T0)/(U.T + T0)

