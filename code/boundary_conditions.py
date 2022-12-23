from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from configuration import SolverConfiguration


class BoundaryConditions:

    def __init__(self, boundaries, solver_configuration: SolverConfiguration) -> None:
        self._boundaries = {boundary: None for boundary in set(boundaries)}
        self.solver_configuration = solver_configuration

    @property
    def boundaries(self):
        return self._boundaries

    def set_dirichlet(self,
                      boundary,
                      density: float,
                      velocity: tuple[float, ...],
                      temperature: float = None,
                      pressure: float = None,
                      energy: float = None):

        gamma = self.solver_configuration.heat_capacity_ratio
        boundaries = split_boundary(boundary)

        for boundary in boundaries:
            self.boundaries[boundary] = Dirichlet(
                density, velocity, temperature, pressure, energy, gamma)

    def set_farfield(self,
                     boundary,
                     density: float,
                     velocity: tuple[float, ...],
                     temperature: float = None,
                     pressure: float = None,
                     energy: float = None):

        gamma = self.solver_configuration.heat_capacity_ratio
        boundaries = split_boundary(boundary)

        for boundary in boundaries:
            self.boundaries[boundary] = FarField(density, velocity, temperature, pressure, energy, gamma)

    def set_outflow(self,
                    boundary,
                    pressure: float):

        boundaries = split_boundary(boundary)

        for boundary in boundaries:
            self.boundaries[boundary] = Outflow(pressure)

    def set_nonreflecting_outflow(self,
                                  boundary,
                                  pressure: float,
                                  type: str = "perfect",
                                  sigma: float = 0.25,
                                  reference_length: float = 1):

        boundaries = split_boundary(boundary)

        for boundary in boundaries:
            self.boundaries[boundary] = NonReflectingOutflow(pressure, type, sigma, reference_length)

    def set_inviscid_wall(self, boundary):

        boundaries = split_boundary(boundary)

        for boundary in boundaries:
            self.boundaries[boundary] = InviscidWall()

    def set_symmetry(self, boundary):

        boundaries = split_boundary(boundary)

        for boundary in boundaries:
            self.boundaries[boundary] = InviscidWall()

    def set_isothermal_wall(self, boundary, temperature: float):

        boundaries = split_boundary(boundary)
        for boundary in boundaries:
            self.boundaries[boundary] = IsothermalWall(temperature)

    def set_adiabatic_wall(self, boundary):

        boundaries = split_boundary(boundary)
        for boundary in boundaries:
            self.boundaries[boundary] = AdiabaticWall()


def split_boundary(boundary):
    if isinstance(boundary, str):
        boundary = boundary.split("|")
    return boundary


def convert_to_pressure(density, temperature=None, velocity=None, energy=None, gamma=1.4):
    if temperature is not None:
        return (gamma - 1) / gamma * temperature * density

    elif velocity is not None and energy is not None:
        return (gamma - 1) * (energy - density/2 * sum([u*u for u in velocity]))

    else:
        raise ValueError("Either temperature, or velocity and energy is needed!")


def convert_to_temperature(density, pressure=None, velocity=None, energy=None, gamma=1.4):
    if pressure is not None:
        return gamma/(gamma - 1) * pressure/density

    elif velocity is not None and energy is not None:
        return gamma/density * (energy - density/2 * sum([u*u for u in velocity]))

    else:
        raise ValueError("Either pressure, or velocity and energy is needed!")


def convert_to_energy(density, velocity, pressure=None, temperature=None, gamma=1.4):
    if pressure is not None:
        return pressure/(gamma - 1) + density/2 * sum([u*u for u in velocity])

    elif temperature is not None:
        return density*temperature/gamma + density/2 * sum([u*u for u in velocity])

    else:
        raise ValueError("Either temperature or pressure is needed!")


def convert_to_pressure_temperature_energy(density, velocity, temperature=None, pressure=None, energy=None, gamma=1.4):
    if temperature is not None:
        pressure = convert_to_pressure(density, temperature, velocity, energy, gamma)
        energy = convert_to_energy(density, velocity, pressure, temperature, gamma)
    elif pressure is not None:
        temperature = convert_to_temperature(density, pressure, velocity, energy, gamma)
        energy = convert_to_energy(density, velocity, pressure, temperature, gamma)
    elif energy is not None:
        temperature = convert_to_temperature(density, pressure, velocity, energy, gamma)
        pressure = convert_to_pressure(density, temperature, velocity, energy, gamma)
    else:
        raise ValueError("Either temperature, pressure or energy is needed!")

    return pressure, temperature, energy


class BC:

    def __str__(self) -> str:
        self.__class__.__name__


class Dirichlet(BC):

    def __init__(self,
                 density: float,
                 velocity: float,
                 temperature: Optional[float] = None,
                 pressure: Optional[float] = None,
                 energy: Optional[float] = None,
                 gamma: float = 1.4) -> None:

        self.density = density
        self.velocity = velocity

        pressure, temperature, energy = convert_to_pressure_temperature_energy(
            density, velocity, temperature, pressure, energy, gamma)

        self.temperature = temperature
        self.pressure = pressure
        self.energy = energy


class FarField(BC):

    def __init__(self,
                 density: float,
                 velocity: float,
                 temperature: Optional[float] = None,
                 pressure: Optional[float] = None,
                 energy: Optional[float] = None,
                 gamma: float = 1.4) -> None:

        self.density = density
        self.velocity = velocity

        pressure, temperature, energy = convert_to_pressure_temperature_energy(
            density, velocity, temperature, pressure, energy, gamma)

        self.temperature = temperature
        self.pressure = pressure
        self.energy = energy


class Outflow(BC):

    def __init__(self, pressure: float) -> None:
        self.pressure = pressure


class NonReflectingOutflow(BC):
    class TYPE(Enum):
        PERFECT = "perfect"
        POINSOT = "poinsot"
        PIROZZOLI = "pirozzoli"

    def __init__(self, pressure: float,
                 type: str = "perfect",
                 sigma: int = 0.25,
                 reference_length: float = 1) -> None:

        self.pressure = pressure
        self.type = self.TYPE(type)
        self.sigma = sigma
        self.reference_length = reference_length


class InviscidWall(BC):

    def __init__(self) -> None: ...


class IsothermalWall(BC):
    def __init__(self, temperature: float) -> None:
        self.temperature = temperature


class AdiabaticWall(BC):

    def __init__(self) -> None:
        pass
