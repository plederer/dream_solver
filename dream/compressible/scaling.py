from __future__ import annotations

import ngsolve as ngs
from dream import bla
from dream.config import MultipleConfiguration, any
from dream.compressible.config import ReferenceState


class Scaling(MultipleConfiguration, is_interface=True):

    @any(default=None)
    def reference_values(self, state: ReferenceState):
        reference = ReferenceState({'L': 1, 'rho': 1.293, 'u': 1, 'c': 343, 'T': 293.15, 'p': 101325})
        if state is not None:
            reference.update(state)
        return reference

    def density(self) -> float:
        return 1.0

    def velocity_magnitude(self, mach_number: float):
        raise NotImplementedError()

    def speed_of_sound(self, mach_number: float):
        raise NotImplementedError()

    def velocity(self, direction: tuple[float, ...], mach_number: float):
        mag = self.velocity_magnitude(mach_number)
        return mag * bla.unit_vector(direction)

    reference_values: ReferenceState


class Aerodynamic(Scaling):

    name = "aerodynamic"

    def _check_Mach_number(self, mach_number: float):

        if isinstance(mach_number, ngs.Parameter):
            mach_number = mach_number.Get()

        if mach_number <= 0.0:
            raise ValueError("Aerodynamic scaling requires Mach number > 0")

    def velocity_magnitude(self, mach_number: float):
        return 1.0

    def speed_of_sound(self, mach_number: float):
        self._check_Mach_number(mach_number)
        return 1/mach_number


class Acoustic(Scaling):

    name = "acoustic"

    def velocity_magnitude(self, mach_number: float):
        return mach_number

    def speed_of_sound(self, mach_number: float):
        return 1.0


class Aeroacoustic(Scaling):

    name = "aeroacoustic"

    def velocity_magnitude(self, mach_number: float):
        Ma = mach_number
        return Ma/(1 + Ma)

    def speed_of_sound(self, mach_number: float):
        Ma = mach_number
        return 1/(1 + Ma)
