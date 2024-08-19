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
                        GridDeformation)
from dream.compressible.state import CompressibleState

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