from __future__ import annotations
import numpy as np
from typing import NamedTuple, TYPE_CHECKING
from ngsolve import CF, InnerProduct

if TYPE_CHECKING:
    from .configuration import SolverConfiguration


class State(NamedTuple):
    velocity: tuple[float, ...] = None
    density: float = None
    pressure: float = None
    temperature: float = None
    energy: float = None

    @property
    def momentum(self) -> tuple[float, ...]:
        return tuple(self.density*u for u in self.velocity)

    @property
    def all_thermodynamic_none(self):
        return all([val is None for val in (self.pressure, self.temperature, self.energy)])


class Calculator:

    def determine_missing(self, state: State) -> State:

        density = state.density
        velocity = state.velocity
        pressure = state.pressure
        temperature = state.temperature
        energy = state.energy

        if velocity is None:
            raise ValueError("Velocity is needed!")

        if self._is_combination_available(density, pressure):
            temperature = self.temperature_dp(density, pressure)
            energy = self.inner_energy_p(pressure) + self.kinetic_energy_du(density, velocity)

        elif self._is_combination_available(density, temperature):
            pressure = self.pressure_dT(density, temperature)
            energy = density * (self.specific_inner_energy_T(temperature) + self.specific_kinetic_energy_u(velocity))

        elif self._is_combination_available(pressure, temperature):
            density = self.density_pT(pressure, temperature)
            energy = self.inner_energy_p(pressure) + self.kinetic_energy_du(density, velocity)

        elif self._is_combination_available(density, energy):
            kinetic_energy = self.kinetic_energy_du(density, velocity)
            inner_energy = self.inner_energy_EEk(energy, kinetic_energy)
            specific_inner_energy = self.specific_inner_energy_dEi(density, inner_energy)

            pressure = self.pressure_Ei(inner_energy)
            temperature = self.temperature_sEi(specific_inner_energy)

        else:
            raise NotImplementedError(f"Can not determine missing values!")

        return State(CF(velocity), density, pressure, temperature, energy)

    def velocity_dm(self, density, momentum):
        if isinstance(momentum, CF):
            velocity = momentum/density
        else:
            velocity = type(momentum)(np.array(momentum)/density)
        return velocity

    def momentum_du(self, density, velocity):
        if isinstance(velocity, CF):
            momentum = density * velocity
        else:
            momentum = type(velocity)(density * np.array(velocity))
        return momentum

    def kinetic_energy_du(self, density, velocity):
        return density * self.specific_kinetic_energy_u(velocity)

    def kinetic_energy_EEi(self, energy, inner_energy):
        return energy - inner_energy

    def kinetic_energy_dsEk(self, density, specific_kinetic_energy):
        return density * specific_kinetic_energy

    def inner_energy_EEk(self, energy, kinetic_energy):
        return energy - kinetic_energy

    def energy_EiEk(self, inner_energy, kinetic_energy):
        return inner_energy + kinetic_energy

    def specific_kinetic_energy_u(self, velocity):
        if isinstance(velocity, CF):
            sEk = InnerProduct(velocity, velocity)/2
        else:
            sEk = np.sum(np.square(np.array(velocity)))/2
        return sEk

    def specific_kinetic_energy_dEk(self, density, kinetic_energy):
        return kinetic_energy/density

    def specific_kinetic_energy_sEsEi(self, specific_energy, specific_inner_energy):
        return specific_energy - specific_inner_energy

    def specific_inner_energy_dEi(self, density, inner_energy):
        return inner_energy/density

    def specific_inner_energy_sEsEk(self, specific_energy, specific_kinetic_energy):
        return specific_energy - specific_kinetic_energy

    def specific_energy_dE(self, density, energy):
        return energy/density

    def specific_energy_sEisEk(self, specific_inner_energy, specific_kinetic_energy):
        return specific_inner_energy + specific_kinetic_energy

    def density_pT(self, pressure, temperature):
        raise NotImplementedError()

    def density_EiT(self, inner_energy, temperature):
        raise NotImplementedError()

    def pressure_dT(self, density, temperature):
        raise NotImplementedError()

    def pressure_Ei(self, inner_energy):
        raise NotImplementedError()

    def temperature_dp(self, density, pressure):
        raise NotImplementedError()

    def temperature_sEi(self, specific_inner_energy):
        raise NotImplementedError()

    def inner_energy_p(self, pressure):
        raise NotImplementedError()

    def inner_energy_dT(self, density, temperature):
        raise NotImplementedError()

    def specific_inner_energy_T(self, temperature):
        raise NotImplementedError()

    def specific_inner_energy_dp(self, density, pressure):
        raise NotImplementedError()

    def _is_combination_available(self, x, y) -> bool:
        combo = True
        if x is None or y is None:
            combo = False
        return combo


class IdealGasCalculator(Calculator):

    def __init__(self, gamma: float = 1.4) -> None:
        self.gamma = gamma

    def density_pT(self, pressure, temperature):
        return self.gamma/(self.gamma - 1) * pressure/temperature

    def density_EiT(self, inner_energy, temperature):
        return self.gamma * inner_energy / temperature

    def pressure_dT(self, density, temperature):
        return (self.gamma - 1)/self.gamma * temperature * density

    def pressure_Ei(self, inner_energy):
        return (self.gamma - 1) * inner_energy

    def temperature_dp(self, density, pressure):
        return self.gamma/(self.gamma - 1) * pressure/density

    def temperature_sEi(self, specific_inner_energy):
        return self.gamma * specific_inner_energy

    def inner_energy_p(self, pressure):
        return pressure/(self.gamma - 1)

    def inner_energy_dT(self, density, temperature):
        sEi = self.specific_inner_energy_T(temperature)
        return density * sEi

    def specific_inner_energy_T(self, temperature):
        return temperature/self.gamma

    def specific_inner_energy_dp(self, density, pressure):
        temperature = self.temperature_dp(density, pressure)
        return temperature/self.gamma


class DimensionlessFarfieldValues:

    @staticmethod
    def _as_parameter(value, boolean):
        if not boolean:
            value = value.Get()
        return value

    @classmethod
    def farfield(cls,
                 direction: tuple[float, ...],
                 cfg: SolverConfiguration,
                 normalize: bool = True,
                 as_parameter: bool = False) -> State:

        rho = cls.density(cfg)
        u = cls.velocity(direction, cfg, normalize, as_parameter)
        p = cls.pressure(cfg, as_parameter)
        T = cls.temperature(cfg, as_parameter)
        rho_E = cls.energy(cfg, as_parameter)
        return State(u, rho, p, T, rho_E)

    @classmethod
    def density(cls, cfg: SolverConfiguration):
        return 1

    @classmethod
    def velocity(cls,
                 vector: tuple[float, ...],
                 cfg: SolverConfiguration,
                 normalize: bool = True,
                 as_parameter: bool = False):
        M = cls._as_parameter(cfg.Mach_number, as_parameter)

        if np.allclose(vector, 0.0):
            ...
        elif normalize:
            vec = np.array(vector)
            vector = tuple(vec/np.sqrt(np.sum(np.square(vec))))

        if cfg.scaling is cfg.scaling.AERODYNAMIC:
            factor = 1
        elif cfg.scaling is cfg.scaling.ACOUSTIC:
            factor = M
        elif cfg.scaling is cfg.scaling.AEROACOUSTIC:
            factor = M/(1+M)

        return tuple(comp * factor for comp in vector)

    @classmethod
    def pressure(cls, cfg: SolverConfiguration, as_parameter: bool = False):
        gamma = cls._as_parameter(cfg.heat_capacity_ratio, as_parameter)
        M = cls._as_parameter(cfg.Mach_number, as_parameter)

        if cfg.scaling is cfg.scaling.AERODYNAMIC:
            factor = M**2
        elif cfg.scaling is cfg.scaling.ACOUSTIC:
            factor = 1
        elif cfg.scaling is cfg.scaling.AEROACOUSTIC:
            factor = (1 + M)**2

        return cls.density(cfg)/(gamma * factor)

    @classmethod
    def temperature(cls, cfg: SolverConfiguration, as_parameter: bool = False):
        gamma = cls._as_parameter(cfg.heat_capacity_ratio, as_parameter)
        M = cls._as_parameter(cfg.Mach_number, as_parameter)

        if cfg.scaling is cfg.scaling.AERODYNAMIC:
            factor = M**2
        elif cfg.scaling is cfg.scaling.ACOUSTIC:
            factor = 1
        elif cfg.scaling is cfg.scaling.AEROACOUSTIC:
            factor = (1 + M)**2

        return 1/((gamma - 1) * factor)

    @classmethod
    def energy(cls, cfg: SolverConfiguration, as_parameter: bool = False):
        gamma = cls._as_parameter(cfg.heat_capacity_ratio, as_parameter)
        M = cls._as_parameter(cfg.Mach_number, as_parameter)

        p = cls.pressure(cfg, as_parameter)

        if cfg.scaling is cfg.scaling.AERODYNAMIC:
            factor = 1
        elif cfg.scaling is cfg.scaling.ACOUSTIC:
            factor = M**2
        elif cfg.scaling is cfg.scaling.AEROACOUSTIC:
            factor = (M/(1 + M))**2

        return p/(gamma - 1) + cls.density(cfg)*factor/2

    @classmethod
    def speed_of_sound(cls, cfg: SolverConfiguration, as_parameter: bool = False):
        M = cls._as_parameter(cfg.Mach_number, as_parameter)

        if cfg.scaling is cfg.scaling.AERODYNAMIC:
            factor = 1/M
        elif cfg.scaling is cfg.scaling.ACOUSTIC:
            factor = 1
        elif cfg.scaling is cfg.scaling.AEROACOUSTIC:
            factor = 1/(1 + M)

        return factor
