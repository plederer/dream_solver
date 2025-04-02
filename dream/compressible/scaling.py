from __future__ import annotations
import typing
import ngsolve as ngs

from dream import bla
from dream.config import InterfaceConfiguration, configuration
from dream.compressible.config import dimensionfulfields

if typing.TYPE_CHECKING:
    from .solver import CompressibleFlowSolver


class Scaling(InterfaceConfiguration, is_interface=True):

    cfg: CompressibleFlowSolver

    @configuration(default=None)
    def dimensionful_values(self, fields: dimensionfulfields):
        """ Set the dimensionful values for the scaling. 

            :setter: Sets the dimensionful values, defaults to {'L': 1, 'rho': 1.293, 'u': 1, 'c': 343, 'T': 293.15, 'p': 101325}
            :getter: Returns the dimensionful values
        """
        reference = dimensionfulfields({'L': 1, 'rho': 1.293, 'u': 1, 'c': 343, 'T': 293.15, 'p': 101325})
        if fields is not None:
            reference.update(fields)
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

    dimensionful_values: dimensionfulfields


class Aerodynamic(Scaling):

    name = "aerodynamic"

    @property
    def reference_reynolds_number(self) -> ngs.CF:
        r""" Returns the reference Reynolds number. 

            .. math:: 
                \Re_{ref} = \Re_\infty
        """
        return self.cfg.reynolds_number/self.velocity_magnitude(self.cfg.mach_number)

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

    @property
    def reference_reynolds_number(self) -> ngs.CF:
        r""" Returns the reference Reynolds number. 

            .. math:: 
                \Re_{ref} = \frac{\Re_\infty}{\Ma_\infty}
        """
        return self.cfg.reynolds_number/self.velocity_magnitude(self.cfg.mach_number)

    def velocity_magnitude(self, mach_number: float):
        return mach_number

    def speed_of_sound(self, mach_number: float):
        return 1.0


class Aeroacoustic(Scaling):

    name = "aeroacoustic"

    @property
    def reference_reynolds_number(self) -> ngs.CF:
        r""" Returns the reference Reynolds number. 

            .. math:: 
                \Re_{ref} = \frac{\Re_\infty (1+\Ma_\infty)}{\Ma_\infty}
        """
        return self.cfg.reynolds_number/self.velocity_magnitude(self.cfg.mach_number)

    def velocity_magnitude(self, mach_number: float):
        Ma = mach_number
        return Ma/(1 + Ma)

    def speed_of_sound(self, mach_number: float):
        Ma = mach_number
        return 1/(1 + Ma)
