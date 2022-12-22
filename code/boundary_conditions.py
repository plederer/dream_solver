from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


def split_boundary(boundary):
    if isinstance(boundary, str):
        boundary = boundary.split("|")
    return boundary


@dataclass
class BaseBC():

    density: Optional[float] = None
    velocity: Optional[tuple[float, ...]] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    energy: Optional[float] = None

    @classmethod
    def convert_to_PTE(cls, density, velocity, temperature=None, pressure=None, energy=None, gamma=1.4):
        if temperature is not None:
            pressure = cls.pressure(density, temperature, velocity, energy, gamma)
            energy = cls.energy(density, velocity, pressure, temperature, gamma)
        elif pressure is not None:
            temperature = cls.temperature(density, pressure, velocity, energy, gamma)
            energy = cls.energy(density, velocity, pressure, temperature, gamma)
        elif energy is not None:
            temperature = cls.temperature(density, pressure, velocity, energy, gamma)
            pressure = cls.pressure(density, temperature, velocity, energy, gamma)

        return pressure, temperature, energy

    @staticmethod
    def pressure(density, temperature=None, velocity=None, energy=None, gamma=1.4):
        if temperature is not None:
            return (gamma - 1) / gamma * temperature * density

        elif velocity is not None and energy is not None:
            return (gamma - 1) * (energy - density/2 * sum([u*u for u in velocity]))

        else:
            raise NotImplementedError()

    @staticmethod
    def temperature(density, pressure=None, velocity=None, energy=None, gamma=1.4):
        if pressure is not None:
            return gamma/(gamma - 1) * pressure/density

        elif velocity is not None and energy is not None:
            return gamma/density * (energy - density/2 * sum([u*u for u in velocity]))

        else:
            raise NotImplementedError()

    @staticmethod
    def energy(density, velocity, pressure=None, temperature=None, gamma=1.4):
        if pressure is not None:
            return pressure/(gamma - 1) + density/2 * sum([u*u for u in velocity])

        elif temperature is not None:
            return density*temperature/gamma + density/2 * sum([u*u for u in velocity])

        else:
            raise NotImplementedError()


class Dirichlet(BaseBC):
    ...


class FarField(BaseBC):
    ...


class Outflow(BaseBC):
    ...


@dataclass
class NROutflow(BaseBC):

    sigma: float = 0.25
