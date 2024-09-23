from __future__ import annotations
import typing

from dream.config import MultipleConfiguration, any
from dream.compressible.config import CompressibleState

if typing.TYPE_CHECKING:
    from dream.compressible import CompressibleEquations


class DynamicViscosity(MultipleConfiguration, is_interface=True):

    @property
    def is_inviscid(self) -> bool:
        return isinstance(self, Inviscid)

    def viscosity(self, U: CompressibleState, equations: CompressibleEquations = None):
        raise NotImplementedError()

    def format(self):
        formatter = self.formatter.new()
        formatter.entry('Dynamic Viscosity', str(self))
        return formatter.output


class Inviscid(DynamicViscosity):

    name: str = "inviscid"
    aliases = ('euler', )

    def viscosity(self, U: CompressibleState, equations: CompressibleEquations = None):
        raise TypeError("Inviscid Setting! Dynamic Viscosity not defined!")


class Constant(DynamicViscosity):

    name: str = "constant"

    def viscosity(self, U: CompressibleState, equations: CompressibleEquations = None):
        return 1


class Sutherland(DynamicViscosity):

    name: str = "sutherland"

    @any(default=110.4)
    def measurement_temperature(self, value: float) -> float:
        return value

    @any(default=1.716e-5)
    def measurement_viscosity(self, value: float) -> float:
        return value

    def viscosity(self, U: CompressibleState, equations: CompressibleEquations):

        T = U.T

        if U.is_set(T):

            REF = equations.get_reference_state()
            INF = equations.get_farfield_state()

            T0 = self.measurement_temperature/REF.T

            return (T/INF.T)**(3/2) * (INF.T + T0)/(T + T0)

    def format(self):
        formatter = self.formatter.new()
        formatter.output += super().format()
        formatter.entry('Law Reference Temperature', self.measurement_temperature)
        formatter.entry('Law Reference Viscosity', self.measurement_viscosity)

        return formatter.output
