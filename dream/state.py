from __future__ import annotations
from typing import NamedTuple
import numpy as np
from ngsolve import CF, InnerProduct


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

    def __repr__(self) -> str:
        rho = self.density
        u = self.velocity
        p = self.pressure
        T = self.temperature
        rho_E = self.energy
        return f"(\u03C1:{rho}, u:{u}, p:{p}, T:{T}, \u03C1E:{rho_E})"


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

        return State(velocity, density, pressure, temperature, energy)

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
