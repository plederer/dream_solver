import ngsolve as ngs

from dream.config import standard_configuration, parameter_configuration
from dream.compressible.state import CompressibleState


class CompressibleFlowConfiguration(form.PDEConfiguration):

    label: str = "compressible_flow"

    bcs = CompressibleBC
    dcs = CompressibleDC

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.equations = CompressibleEquations(self)

    def dimensionless_infinity_values(self, direction: tuple[float, ...]) -> CompressibleState:
        return self.equations.get_farfield_state(direction)

    @standard_configuration(Conservative)
    def formulation(self, formulation) -> CompressibleFormulation:
        return formulation

    @parameter_configuration(default=0.3)
    def Mach_number(self, Mach_number: float):

        if Mach_number < 0:
            raise ValueError("Invalid Mach number. Value has to be >= 0!")

        return Mach_number

    @parameter_configuration(default=1)
    def Reynolds_number(self, Reynolds_number: float):
        """ Represents the ratio between inertial and viscous forces """
        if Reynolds_number <= 0:
            raise ValueError("Invalid Reynold number. Value has to be > 0!")

        return Reynolds_number

    @parameter_configuration(default=0.72)
    def Prandtl_number(self, Prandtl_number: float):
        if Prandtl_number <= 0:
            raise ValueError("Invalid Prandtl_number. Value has to be > 0!")

        return Prandtl_number

    @multiple_configuration(default=IdealGas)
    def equation_of_state(self, equation_of_state):
        return equation_of_state

    @multiple_configuration(default=Inviscid)
    def dynamic_viscosity(self, dynamic_viscosity):
        return dynamic_viscosity

    @multiple_configuration(default=Aerodynamic)
    def scaling(self, scaling):
        return scaling

    @multiple_configuration(default=LaxFriedrich)
    def riemann_solver(self, riemann_solver):
        riemann_solver['cfg'] = self
        return riemann_solver

    @multiple_configuration(default=Inactive)
    def mixed_method(self, mixed_method):
        return mixed_method

    @Reynolds_number.getter_check
    def Reynolds_number(self):
        if self.dynamic_viscosity.is_inviscid:
            raise ValueError("Inviscid solver configuration: Reynolds number not applicable")

    @Prandtl_number.getter_check
    def Prandtl_number(self):
        if self.dynamic_viscosity.is_inviscid:
            raise ValueError("Inviscid solver configuration: Prandtl number not applicable")

    Mach_number: ngs.Parameter
    Reynolds_number: ngs.Parameter
    Prandtl_number: ngs.Parameter
    equation_of_state: IdealGas
    dynamic_viscosity: Constant | Inviscid | Sutherland
    scaling: Aerodynamic | Acoustic | Aeroacoustic
    riemann_solver: LaxFriedrich | Roe | HLL | HLLEM
    mixed_method: Inactive | StrainHeat | Gradient
