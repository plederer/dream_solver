from __future__ import annotations

from dream import bla
from dream.config import variable, State


class CompressibleState(State):

    density = variable(bla.as_scalar)
    velocity = variable(bla.as_vector)
    momentum = variable(bla.as_vector)
    pressure = variable(bla.as_scalar)
    temperature = variable(bla.as_scalar)
    energy = variable(bla.as_scalar)
    specific_energy = variable(bla.as_scalar)
    inner_energy = variable(bla.as_scalar)
    specific_inner_energy = variable(bla.as_scalar)
    kinetic_energy = variable(bla.as_scalar)
    specific_kinetic_energy = variable(bla.as_scalar)
    enthalpy = variable(bla.as_scalar)
    specific_enthalpy = variable(bla.as_scalar)
    speed_of_sound = variable(bla.as_scalar)

    convective_flux = variable(bla.as_matrix)
    diffusive_flux = variable(bla.as_matrix)

    viscosity = variable(bla.as_scalar)
    strain_rate_tensor = variable(bla.as_matrix)
    deviatoric_stress_tensor = variable(bla.as_matrix)
    heat_flux = variable(bla.as_vector)

    density_gradient = variable(bla.as_vector)
    velocity_gradient = variable(bla.as_matrix)
    momentum_gradient = variable(bla.as_matrix)
    pressure_gradient = variable(bla.as_vector)
    temperature_gradient = variable(bla.as_vector)
    energy_gradient = variable(bla.as_vector)
    specific_energy_gradient = variable(bla.as_vector)
    inner_energy_gradient = variable(bla.as_vector)
    specific_inner_energy_gradient = variable(bla.as_vector)
    kinetic_energy_gradient = variable(bla.as_vector)
    specific_kinetic_energy_gradient = variable(bla.as_vector)
    enthalpy_gradient = variable(bla.as_vector)
    specific_enthalpy_gradient = variable(bla.as_vector)


class ScalingState(CompressibleState):

    length = variable(bla.as_scalar)
    density = variable(bla.as_scalar)
    momentum = variable(bla.as_scalar)
    velocity = variable(bla.as_scalar)
    speed_of_sound = variable(bla.as_scalar)
    temperature = variable(bla.as_scalar)
    pressure = variable(bla.as_scalar)
    energy = variable(bla.as_scalar)
    inner_energy = variable(bla.as_scalar)
    kinetic_energy = variable(bla.as_scalar)
