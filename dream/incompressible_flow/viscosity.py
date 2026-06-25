from __future__ import annotations
import typing
import ngsolve as ngs

# Import the necessary modules from dream
from dream.config import Configuration, dream_configuration
from dream.incompressible_flow.config import flowfields

if typing.TYPE_CHECKING:
    from dream.incompressible_flow.solver import IncompressibleFlowSolver


class DynamicViscosity(Configuration, is_interface=True):

    root: IncompressibleFlowSolver

    @property
    def is_linear_model(self):
        return isinstance(self, Constant)

    def shear_rate(self, u: flowfields):
        eps = self.root.strain_rate_tensor(u)
        return 2*ngs.sqrt(0.5 * ngs.InnerProduct(eps, eps))

    def kinematic_viscosity(self, u: flowfields):
        raise NotImplementedError("Overload this method in derived class!")


class Constant(DynamicViscosity):

    name: str = "constant"

    def kinematic_viscosity(self, u: flowfields):
        return 1.0


class Powerlaw(DynamicViscosity):

    name: str = "powerlaw"

    def __init__(self, mesh, root=None, **default):

        self._powerlaw_exponent = ngs.Parameter(2.0)
        self._viscosity_ratio = ngs.Parameter(1.0)

        DEFAULT = {
            'powerlaw_exponent': self._powerlaw_exponent.Get(),
            'viscosity_ratio': self._viscosity_ratio.Get()
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def powerlaw_exponent(self) -> ngs.Parameter:
        """ Returns the power law exponent """
        return self._powerlaw_exponent

    @powerlaw_exponent.setter
    def powerlaw_exponent(self, power_law_exponent: ngs.Parameter) -> None:

        if isinstance(power_law_exponent, ngs.Parameter):
            power_law_exponent = power_law_exponent.Get()

        if not 1 <= power_law_exponent <= 2:
            raise ValueError("Invalid power law exponent. Value has to be in the range of 1 <= r <= 2!")

        self._powerlaw_exponent.Set(power_law_exponent)

    @dream_configuration
    def viscosity_ratio(self) -> ngs.Parameter:
        """ Returns the viscosity ratio """
        return self._viscosity_ratio

    @viscosity_ratio.setter
    def viscosity_ratio(self, viscosity_ratio: ngs.Parameter) -> None:

        if isinstance(viscosity_ratio, ngs.Parameter):
            viscosity_ratio = viscosity_ratio.Get()

        if viscosity_ratio <= 0:
            raise ValueError("Invalid viscosity ratio. Value has to be > 0!")

        self._viscosity_ratio.Set(viscosity_ratio)

    def kinematic_viscosity(self, u: flowfields):
        return self.viscosity_ratio * self.shear_rate(u)**(self.powerlaw_exponent - 2)

