from __future__ import annotations

from dream.config import MultipleConfiguration, standard_configuration
from dream.compressible.state import CompressibleState


class DynamicViscosity(MultipleConfiguration, is_interface=True):

    @property
    def is_inviscid(self) -> bool:
        return isinstance(self, Inviscid)

    def viscosity(self, state: CompressibleState, *args):
        raise NotImplementedError()

    def format(self):
        formatter = self.formatter.new()
        formatter.entry('Dynamic Viscosity', str(self))
        return formatter.output


class Inviscid(DynamicViscosity):

    def viscosity(self, state: CompressibleState, *args):
        raise TypeError("Inviscid Setting! Dynamic Viscosity not defined!")


class Constant(DynamicViscosity):

    def viscosity(self, state: CompressibleState, *args):
        return 1


class Sutherland(DynamicViscosity):

    @standard_configuration(default=110.4)
    def measurement_temperature(self, value: float) -> float:
        return value

    @standard_configuration(default=1.716e-5)
    def measurement_viscosity(self, value: float) -> float:
        return value

    def viscosity(self, state: CompressibleState, equations: CompressibleEquations):

        T = state.temperature

        if state.is_set(T):

            REF = equations.get_reference_state()
            INF = equations.get_farfield_state()

            Tinf = INF.temperature
            T0 = self.measurement_temperature/REF.temperature

            return (T/Tinf)**(3/2) * (Tinf + T0)/(T + T0)

    def format(self):
        formatter = self.formatter.new()
        formatter.output += super().format()
        formatter.entry('Law Reference Temperature', self.measurement_temperature)
        formatter.entry('Law Reference Viscosity', self.measurement_viscosity)

        return formatter.output
