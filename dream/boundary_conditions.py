from __future__ import annotations
import enum
from collections import UserDict
from typing import Optional, TYPE_CHECKING
from ngsolve import CF, Parameter, InnerProduct

if TYPE_CHECKING:
    from .configuration import SolverConfiguration


def extract_from_ngs_pattern(pattern, facets):
    if isinstance(pattern, str):
        pattern = pattern.split("|")
    intersection = set(facets).intersection(pattern)

    return tuple(intersection)


class InitialCondition(UserDict):

    def __init__(self, domains, solver_configuration: SolverConfiguration) -> None:
        super().__init__({domain: None for domain in set(domains)})
        self.solver_configuration = solver_configuration

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(self)

    @property
    def ngs_pattern(self) -> str:
        return "|".join(self.domains)

    def set(self, density, velocity, temperature=None, pressure=None, energy=None, domain=None):

        if domain is None:
            domains = self.domains
        else:
            domains = extract_from_ngs_pattern(domain, self.domains)

        gamma = self.solver_configuration.heat_capacity_ratio

        for domain in domains:
            self[domain] = Initial(density, velocity, temperature, pressure, energy, gamma)


class BoundaryConditions(UserDict):

    def __init__(self, boundaries, solver_configuration: SolverConfiguration) -> None:
        super().__init__({boundary: None for boundary in set(boundaries)})
        self.solver_configuration = solver_configuration

    @property
    def boundaries(self) -> tuple[str, ...]:
        return tuple(self)

    @property
    def ngs_pattern(self) -> str:
        return "|".join(self.boundaries)

    def set_dirichlet(self,
                      boundary,
                      density: float,
                      velocity: tuple[float, ...],
                      temperature: float = None,
                      pressure: float = None,
                      energy: float = None):

        gamma = self.solver_configuration.heat_capacity_ratio
        boundaries = extract_from_ngs_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = Dirichlet(density, velocity, temperature, pressure, energy, gamma)

    def set_farfield(self,
                     boundary,
                     density: float,
                     velocity: tuple[float, ...],
                     temperature: float = None,
                     pressure: float = None,
                     energy: float = None):

        gamma = self.solver_configuration.heat_capacity_ratio
        boundaries = extract_from_ngs_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = FarField(density, velocity, temperature, pressure, energy, gamma)

    def set_outflow(self,
                    boundary,
                    pressure: float):

        boundaries = extract_from_ngs_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = Outflow(pressure)

    def set_nonreflecting_outflow(self,
                                  boundary,
                                  pressure: float,
                                  type: str = "perfect",
                                  sigma: float = 0.25,
                                  reference_length: float = 1,
                                  tangential_convective_fluxes: bool = True,
                                  tangential_viscous_fluxes: bool = True,
                                  normal_viscous_fluxes: bool = False):

        boundaries = extract_from_ngs_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = NonReflectingOutflow(pressure, type, sigma, reference_length,
                                                  tangential_convective_fluxes, tangential_viscous_fluxes,
                                                  normal_viscous_fluxes)

    def set_inviscid_wall(self, boundary):

        boundaries = extract_from_ngs_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = InviscidWall()

    def set_symmetry(self, boundary):

        boundaries = extract_from_ngs_pattern(boundary, self.boundaries)

        for boundary in boundaries:
            self[boundary] = InviscidWall()

    def set_isothermal_wall(self, boundary, temperature: float):

        boundaries = extract_from_ngs_pattern(boundary, self.boundaries)
        for boundary in boundaries:
            self[boundary] = IsothermalWall(temperature)

    def set_adiabatic_wall(self, boundary):

        boundaries = extract_from_ngs_pattern(boundary, self.boundaries)
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
                 velocity: float,
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
                 velocity: float,
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


class NonReflectingOutflow(Condition):
    class TYPE(enum.Enum):
        PERFECT = "perfect"
        POINSOT = "poinsot"
        PIROZZOLI = "pirozzoli"

    def __init__(self, pressure,
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


class InviscidWall(Condition):

    def __init__(self) -> None: ...


class IsothermalWall(Condition):
    def __init__(self, temperature) -> None:
        self.temperature = CF(temperature)


class AdiabaticWall(Condition):

    def __init__(self) -> None:
        pass


class Initial(Condition):

    def __init__(self,
                 density: float,
                 velocity: float,
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
