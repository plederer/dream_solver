class Calculator:

    def velocity_dm(self, density, momentum):
        return momentum/density

    def momentum_du(self, density, velocity):
        return density * velocity

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
        return (velocity * velocity)/2

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


class IdealGasCalculator(Calculator):

    def __init__(self, gamma: float = 1.4) -> None:
        self.gamma = gamma

    def determine_missing(self,
                          velocity,
                          density=None,
                          pressure=None,
                          temperature=None,
                          energy=None):

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
            raise NotImplementedError()

        return velocity, density, pressure, temperature, energy

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

    def _is_combination_available(self, x, y) -> bool:
        combo = False
        if x is not None and y is not None:
            combo = True
        return combo
