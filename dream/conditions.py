from __future__ import annotations
import enum
import dataclasses
from collections import UserDict
from typing import Optional, TYPE_CHECKING, ItemsView
from ngsolve import CF, Parameter, InnerProduct

if TYPE_CHECKING:
    from .configuration import SolverConfiguration


def extract_boundaries_from_pattern(pattern, facets):
    if isinstance(pattern, str):
        pattern = pattern.split("|")
    intersection = set(facets).intersection(pattern)

    return tuple(intersection)


@dataclasses.dataclass
class DomainConditionsContainer:
    initial: Optional[Initial] = None
    sponge: Optional[SpongeLayer] = None
    pml: Optional[PML] = None


class DomainConditions(UserDict):

    def __init__(self, domains, solver_configuration: SolverConfiguration) -> None:
        super().__init__({domain: DomainConditionsContainer() for domain in set(domains)})
        self.solver_configuration = solver_configuration

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(self)

    @property
    def ngs_pattern(self) -> str:
        return "|".join(self.domains)

    def set_initial(self,
                    density: float,
                    velocity: tuple[float, ...],
                    temperature: Optional[float] = None,
                    pressure: Optional[float] = None,
                    energy: Optional[float] = None,
                    domain: Optional[str] = None):

        if domain is None:
            domains = self.domains
        else:
            domains = extract_boundaries_from_pattern(domain, self.domains)

        gamma = self.solver_configuration.heat_capacity_ratio

        for domain in domains:
            self[domain].initial = Initial(density, velocity, temperature, pressure, energy, gamma)

    def set_sponge_layer(self,
                         domain: str,
                         weight_function: CF,
                         density: float,
                         velocity: tuple[float, ...],
                         temperature: float = None,
                         pressure: float = None,
                         energy: float = None,
                         weight_function_order: int = 3):

        domains = extract_boundaries_from_pattern(domain, self.domains)

        gamma = self.solver_configuration.heat_capacity_ratio

        for domain in domains:
            self[domain].sponge = SpongeLayer(weight_function, density,
                                              velocity, temperature, pressure, energy,
                                              weight_function_order, gamma)

    def __getitem__(self, key: str) -> DomainConditionsContainer:
        return super().__getitem__(key)

    def items(self) -> ItemsView[str, DomainConditionsContainer]:
        return super().items()


class BoundaryConditions(UserDict):

    def __init__(self, boundaries, solver_configuration: SolverConfiguration) -> None:
        super().__init__({boundary: None for boundary in set(boundaries)})
        self.solver_configuration = solver_configuration

    @property
    def boundaries(self) -> tuple[str, ...]:
        return tuple(self)

    @property
    def pattern(self) -> str:
        return "|".join([boundary for boundary, bc in self.items() if isinstance(bc, Condition)])

    def set_dirichlet(self,
                      boundary: str,
                      density: float,
                      velocity: tuple[float, ...],
                      temperature: float = None,
                      pressure: float = None,
                      energy: float = None):

        gamma = self.solver_configuration.heat_capacity_ratio
        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = Dirichlet(density, velocity, temperature, pressure, energy, gamma)

    def set_farfield(self,
                     boundary: str,
                     density: float,
                     velocity: tuple[float, ...],
                     temperature: float = None,
                     pressure: float = None,
                     energy: float = None):

        gamma = self.solver_configuration.heat_capacity_ratio
        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = FarField(density, velocity, temperature, pressure, energy, gamma)

    def set_outflow(self,
                    boundary: str,
                    pressure: float):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = Outflow(pressure)

    def set_nonreflecting_outflow(self,
                                  boundary: str,
                                  pressure: float,
                                  type: str = "perfect",
                                  sigma: float = 0.25,
                                  reference_length: float = 1,
                                  tangential_convective_fluxes: bool = True,
                                  tangential_viscous_fluxes: bool = True,
                                  normal_viscous_fluxes: bool = False):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = NRBC_Outflow(pressure, type, sigma, reference_length,
                                          tangential_convective_fluxes, tangential_viscous_fluxes,
                                          normal_viscous_fluxes)

    def set_nonreflecting_inflow(self,
                                 boundary: str,
                                 density: float,
                                 velocity: tuple[float, ...],
                                 temperature: float = None,
                                 pressure: float = None,
                                 energy: float = None,
                                 type: str = "perfect",
                                 sigma: float = 0.25,
                                 reference_length: float = 1,
                                 tangential_convective_fluxes: bool = True,
                                 tangential_viscous_fluxes: bool = True,
                                 normal_viscous_fluxes: bool = False):

        gamma = self.solver_configuration.heat_capacity_ratio
        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = NRBC_Inflow(density,
                                         velocity,
                                         temperature,
                                         pressure,
                                         energy,
                                         type,
                                         sigma,
                                         reference_length,
                                         gamma,
                                         tangential_convective_fluxes,
                                         tangential_viscous_fluxes,
                                         normal_viscous_fluxes)

    def set_inviscid_wall(self, boundary: str):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = InviscidWall()

    def set_periodic(self, boundary: str):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = Periodic()

    def set_symmetry(self, boundary: str):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = InviscidWall()

    def set_isothermal_wall(self, boundary: str, temperature: float):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)
        for boundary in boundaries:
            self[boundary] = IsothermalWall(temperature)

    def set_adiabatic_wall(self, boundary: str):

        boundaries = extract_boundaries_from_pattern(boundary, self.boundaries)
        for boundary in boundaries:
            self[boundary] = AdiabaticWall()


def convert_to_pressure(density, temperature=None, velocity=None, energy=None, gamma=1.4) -> CF:
    if temperature is not None:
        return (gamma - 1) / gamma * temperature * density

    elif velocity is not None and energy is not None:
        return (gamma - 1) * (energy - density/2 * InnerProduct(velocity, velocity))

    else:
        raise ValueError("Either temperature, or velocity and energy is needed!")


def convert_to_temperature(density, pressure=None, velocity=None, energy=None, gamma=1.4) -> CF:
    if pressure is not None:
        return gamma/(gamma - 1) * pressure/density

    elif velocity is not None and energy is not None:
        return gamma/density * (energy - density/2 * InnerProduct(velocity, velocity))

    else:
        raise ValueError("Either pressure, or velocity and energy is needed!")


def convert_to_energy(density, velocity, pressure=None, temperature=None, gamma=1.4) -> CF:
    if pressure is not None:
        return pressure/(gamma - 1) + density/2 * InnerProduct(velocity, velocity)

    elif temperature is not None:
        return density*temperature/gamma + density/2 * InnerProduct(velocity, velocity)

    else:
        raise ValueError("Either temperature or pressure is needed!")


def convert_to_pressure_temperature_energy(density, velocity, temperature=None, pressure=None, energy=None, gamma=1.4):

    if temperature is not None:
        temperature = CF(temperature)
        pressure = convert_to_pressure(density, temperature, velocity, energy, gamma)
        energy = convert_to_energy(density, velocity, pressure, temperature, gamma)

    elif pressure is not None:
        pressure = CF(pressure)
        temperature = convert_to_temperature(density, pressure, velocity, energy, gamma)
        energy = convert_to_energy(density, velocity, pressure, temperature, gamma)

    elif energy is not None:
        energy = CF(energy)
        temperature = convert_to_temperature(density, pressure, velocity, energy, gamma)
        pressure = convert_to_pressure(density, temperature, velocity, energy, gamma)

    else:
        raise ValueError("Either temperature, pressure or energy is needed!")

    return pressure, temperature, energy


class Condition:

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__class__.__name__


class Dirichlet(Condition):

    def __init__(self,
                 density: float,
                 velocity: tuple[float, ...],
                 temperature: Optional[float] = None,
                 pressure: Optional[float] = None,
                 energy: Optional[float] = None,
                 gamma: float = 1.4) -> None:

        density = CF(density)
        velocity = CF(velocity)

        pressure, temperature, energy = convert_to_pressure_temperature_energy(
            density, velocity, temperature, pressure, energy, gamma)

        self.density = density
        self.velocity = velocity
        self.momentum = density * velocity
        self.temperature = temperature
        self.pressure = pressure
        self.energy = energy


class FarField(Condition):

    def __init__(self,
                 density: float,
                 velocity: tuple[float, ...],
                 temperature: Optional[float] = None,
                 pressure: Optional[float] = None,
                 energy: Optional[float] = None,
                 gamma: float = 1.4) -> None:

        density = CF(density)
        velocity = CF(velocity)

        pressure, temperature, energy = convert_to_pressure_temperature_energy(
            density, velocity, temperature, pressure, energy, gamma)

        self.density = density
        self.velocity = velocity
        self.momentum = density * velocity
        self.temperature = temperature
        self.pressure = pressure
        self.energy = energy


class Outflow(Condition):

    def __init__(self, pressure) -> None:
        self.pressure = CF(pressure)


class NRBC_Outflow(Condition):
    class TYPE(enum.Enum):
        PERFECT = "perfect"
        PARTIALLY = "partially"
        PIROZZOLI = "pirozzoli"

    def __init__(self,
                 pressure,
                 type: str = "perfect",
                 sigma: int = 0.25,
                 reference_length: float = 1,
                 tangential_convective_fluxes: bool = True,
                 tangential_viscous_fluxes: bool = True,
                 normal_viscous_fluxes: bool = False) -> None:

        self.pressure = CF(pressure)
        self.type = self.TYPE(type)
        self.sigma = Parameter(sigma)
        self.reference_length = Parameter(reference_length)
        self.tang_conv_flux = tangential_convective_fluxes
        self.tang_visc_flux = tangential_viscous_fluxes
        self.norm_visc_flux = normal_viscous_fluxes


class NRBC_Inflow(Condition):

    class TYPE(enum.Enum):
        PERFECT = "perfect"
        PARTIALLY = "partially"

    def __init__(self,
                 density: float,
                 velocity: tuple[float, ...],
                 temperature: Optional[float] = None,
                 pressure: Optional[float] = None,
                 energy: Optional[float] = None,
                 type: str = "perfect",
                 sigma: float = 0.25,
                 reference_length: float = 1,
                 gamma: float = 1.4,
                 tangential_convective_fluxes: bool = True,
                 tangential_viscous_fluxes: bool = True,
                 normal_viscous_fluxes: bool = False) -> None:

        self.type = self.TYPE(type)
        self.sigma = Parameter(sigma)
        self.reference_length = Parameter(reference_length)

        density = CF(density)
        velocity = CF(velocity)

        pressure, temperature, energy = convert_to_pressure_temperature_energy(
            density, velocity, temperature, pressure, energy, gamma)

        self.density = density
        self.velocity = velocity
        self.momentum = density * velocity
        self.temperature = temperature
        self.pressure = pressure
        self.energy = energy

        self.tang_conv_flux = tangential_convective_fluxes
        self.tang_visc_flux = tangential_viscous_fluxes
        self.norm_visc_flux = normal_viscous_fluxes


class InviscidWall(Condition):
    ...


class IsothermalWall(Condition):
    def __init__(self, temperature) -> None:
        self.temperature = CF(temperature)


class AdiabaticWall(Condition):

    def __init__(self) -> None:
        pass


class Initial(Condition):

    def __init__(self,
                 density: float,
                 velocity: tuple[float, ...],
                 temperature: Optional[float] = None,
                 pressure: Optional[float] = None,
                 energy: Optional[float] = None,
                 gamma: float = 1.4) -> None:

        density = CF(density)
        velocity = CF(velocity)

        pressure, temperature, energy = convert_to_pressure_temperature_energy(
            density, velocity, temperature, pressure, energy, gamma)

        self.density = density
        self.velocity = velocity
        self.momentum = density * velocity
        self.temperature = temperature
        self.pressure = pressure
        self.energy = energy


class SpongeLayer(Condition):

    def __init__(self,
                 weight_function: CF,
                 density: float,
                 velocity: tuple[float, ...],
                 temperature: Optional[float] = None,
                 pressure: Optional[float] = None,
                 energy: Optional[float] = None,
                 weight_function_order: int = 3,
                 gamma: float = 1.4) -> None:

        density = CF(density)
        velocity = CF(velocity)

        pressure, temperature, energy = convert_to_pressure_temperature_energy(
            density, velocity, temperature, pressure, energy, gamma)

        self.weight_function = weight_function
        self.weight_function_order = weight_function_order
        self.density = density
        self.velocity = velocity
        self.momentum = density * velocity
        self.temperature = temperature
        self.pressure = pressure
        self.energy = energy


class PML(Condition):
    ...


class Periodic:
    ...
