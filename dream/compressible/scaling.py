from __future__ import annotations

import ngsolve as ngs
from dream import bla
from dream.config import DescriptorConfiguration, any
from dream.compressible.state import ScalingState


class Scaling(DescriptorConfiguration, is_interface=True):

    @any(default={'L': 1, 'rho': 1.293, 'u': 102.9, 'c': 343, 'T': 293.15, 'p': 101325})
    def dimensional_infinity_values(self, state: ScalingState):
        return ScalingState(**state)

    def density(self) -> float:
        return 1.0

    def velocity_magnitude(self, Mach_number: float):
        raise NotImplementedError()

    def speed_of_sound(self, Mach_number: float):
        raise NotImplementedError()

    def velocity(self, direction: tuple[float, ...], Mach_number: float):
        mag = self.velocity_magnitude(Mach_number)
        return mag * bla.unit_vector(direction)

    def format(self):
        formatter = self.formatter.new()
        formatter.entry('Scaling', str(self))
        return formatter.output

    dimensional_infinity_values: ScalingState


class Aerodynamic(Scaling):

    name = "aerodynamic"

    def _check_Mach_number(self, Mach_number: float):
        Ma = Mach_number
        if isinstance(Ma, ngs.Parameter):
            Ma = Ma.Get()

        if Ma <= 0.0:
            raise ValueError("Aerodynamic scaling requires Mach number > 0")

    def velocity_magnitude(self, Mach_number: float):
        return 1.0

    def speed_of_sound(self, Mach_number: float):
        self._check_Mach_number(Mach_number)
        return 1/Mach_number


class Acoustic(Scaling):

    name = "acoustic"

    def velocity_magnitude(self, Mach_number: float):
        return Mach_number

    def speed_of_sound(self, Mach_number: float):
        return 1.0


class Aeroacoustic(Scaling):

    name = "aeroacoustic"

    def velocity_magnitude(self, Mach_number: float):
        Ma = Mach_number
        return Ma/(1 + Ma)

    def speed_of_sound(self, Mach_number: float):
        Ma = Mach_number
        return 1/(1 + Ma)
