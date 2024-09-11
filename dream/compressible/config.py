from __future__ import annotations

from dream import bla
from dream.config import variable, State
from dream.pde import Formulation
from dream.mesh import (Boundary,
                        Domain,
                        BoundaryConditions,
                        DomainConditions,
                        Periodic,
                        Initial,
                        Force,
                        Perturbation,
                        SpongeLayer,
                        PSpongeLayer,
                        GridDeformation,
                        DreamMesh)


class CompressibleFormulation(Formulation, is_interface=True):
    ...


class CompressibleState(State):

    rho = variable(bla.as_scalar, 'density')
    u = variable(bla.as_vector, 'velocity')
    rho_u = variable(bla.as_vector, 'momentum')
    p = variable(bla.as_scalar, 'pressure')
    T = variable(bla.as_scalar, 'temperature')
    rho_E = variable(bla.as_scalar, 'energy')
    E = variable(bla.as_scalar, 'specific_energy')
    rho_Ei = variable(bla.as_scalar, 'inner_energy')
    Ei = variable(bla.as_scalar, 'specific_inner_energy')
    rho_Ek = variable(bla.as_scalar, 'kinetic_energy')
    Ek = variable(bla.as_scalar, 'specific_kinetic_energy')
    rho_H = variable(bla.as_scalar, 'enthalpy')
    H = variable(bla.as_scalar, 'specific_enthalpy')
    c = variable(bla.as_scalar, 'speed_of_sound')


class CompressibleStateGradient(State):

    rho = variable(bla.as_vector, 'density')
    u = variable(bla.as_matrix, 'velocity')
    rho_u = variable(bla.as_matrix, 'momentum')
    p = variable(bla.as_vector, 'pressure')
    T = variable(bla.as_vector, 'temperature')
    rho_E = variable(bla.as_vector, 'energy')
    E = variable(bla.as_vector, 'specific_energy')
    rho_Ei = variable(bla.as_vector, 'inner_energy')
    Ei = variable(bla.as_vector, 'specific_inner_energy')
    rho_Ek = variable(bla.as_vector, 'kinetic_energy')
    Ek = variable(bla.as_vector, 'specific_kinetic_energy')
    rho_H = variable(bla.as_vector, 'enthalpy')
    H = variable(bla.as_vector, 'specific_enthalpy')
    c = variable(bla.as_vector, 'speed_of_sound')

    eps = variable(bla.as_matrix, 'strain_rate_tensor')


class ReferenceState(State):

    L = variable(bla.as_scalar, "length")
    rho = variable(bla.as_scalar, "density")
    rho_u = variable(bla.as_scalar, "momentum")
    u = variable(bla.as_scalar, "velocity")
    c = variable(bla.as_scalar, "speed_of_sound")
    T = variable(bla.as_scalar, "temperature")
    p = variable(bla.as_scalar, "pressure")


class FarField(Boundary):

    def __init__(self, state: CompressibleState, theta_0: float = 0):
        super().__init__(state)
        self.theta_0 = theta_0


class Outflow(Boundary):

    def __init__(self, pressure: CompressibleState | float):
        if not isinstance(pressure, CompressibleState):
            pressure = CompressibleState(pressure=pressure)
        super().__init__(pressure)


class CharacteristicRelaxationOutflow(Boundary):

    def __init__(self,
                 state: CompressibleState,
                 sigma: float = 0.25,
                 reference_length: float = 1,
                 tangential_convective_fluxes: bool = True) -> None:

        super().__init__(state)
        self.sigma = sigma
        self.reference_length = reference_length
        self.tang_conv_flux = tangential_convective_fluxes


class CharacteristicRelaxationInflow(Boundary):

    def __init__(self,
                 state: CompressibleState,
                 sigma: float = 0.25,
                 reference_length: float = 1,
                 tangential_convective_fluxes: bool = True) -> None:

        super().__init__(state)
        self.sigma = sigma
        self.reference_length = reference_length
        self.tang_conv_flux = tangential_convective_fluxes


class InviscidWall(Boundary):
    ...


class Symmetry(Boundary):
    ...


class IsothermalWall(Boundary):

    def __init__(self, temperature: float | CompressibleState) -> None:
        if not isinstance(temperature, CompressibleState):
            temperature = CompressibleState(temperature=temperature)
        super().__init__(temperature)


class AdiabaticWall(Boundary):
    ...


class CompressibleBC(BoundaryConditions):
    farfield = FarField
    outflow = Outflow
    characteristic_outflow = CharacteristicRelaxationOutflow
    characteristic_inflow = CharacteristicRelaxationInflow
    inviscid_wall = InviscidWall
    symmetry = Symmetry
    isothermal_wall = IsothermalWall
    adiabatic_wall = AdiabaticWall
    periodic = Periodic


# ------- Domain Conditions ------- #


class PML(Domain):
    ...


class CompressibleDC(DomainConditions):
    initial = Initial
    force = Force
    perturbation = Perturbation
    sponge_layer = SpongeLayer
    psponge_layer = PSpongeLayer
    grid_deformation = GridDeformation
    pml = PML
