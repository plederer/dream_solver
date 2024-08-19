from __future__ import annotations

from dream import bla
from dream.config import variable, State


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
