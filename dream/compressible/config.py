from __future__ import annotations

from dream import bla
from dream.config import variable, State, any
from dream.pde import Formulation
from dream.mesh import Condition


class CompressibleFormulation(Formulation, is_interface=True):
    ...


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


class FarField(Condition):

    name = "farfield"

    @any(default=None)
    def state(self, state) -> State:
        if state is not None:
            state = CompressibleState(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Farfield state not set!")

    @any(default=0)
    def theta(self, theta: float):
        return theta


class Outflow(Condition):

    name = "outflow"

    @any(default=None)
    def pressure(self, pressure) -> State:
        return pressure

    @pressure.getter_check
    def pressure(self) -> None:
        if self.data['pressure'] is None:
            raise ValueError("Static pressure not set!")


class CBC(Condition, is_interface=True):

    name = "cbc"

    @any(default=None)
    def state(self, state) -> State:
        if state is not None:
            state = CompressibleState(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Farfield state not set!")

    @any(default=0.28)
    def relaxation_parameters(self, param: float):
        parameters = ["acoustic_in", "entropy", "vorticity_0", "vorticity_1", "acoustic_out"]
        if bla.is_scalar(param):
            parameters = dict.fromkeys(parameters, param)
        elif isinstance(param, (list, tuple)):
            parameters = {key: value for key, value in zip(parameters, param)}
        elif isinstance(param, dict):
            parameters = {key: value for key, value in zip(parameters, param.values())}
        return param

    @any(default=False)
    def is_tangential_convective_fluxes(self, tangential_convective_fluxes: bool):
        return bool(tangential_convective_fluxes)

    @any(default=False)
    def is_viscous_fluxes(self, viscous_fluxes: bool):
        return bool(viscous_fluxes)


class GRCBC(CBC):

    name = "grcbc"


class NSCBC(CBC):

    name = "nscbc"

    @any(default=1)
    def length(self, length: float):
        return length


class InviscidWall(Condition):

    name = "inviscid_wall"


class Symmetry(Condition):

    name = "symmetry"


class IsothermalWall(Condition):

    name = "isothermal_wall"

    @any(default=None)
    def temperature(self, temperature) -> State:
        return temperature

    @temperature.getter_check
    def temperature(self) -> None:
        if self.data['temperature'] is None:
            raise ValueError("Temperature not set!")


class AdiabaticWall(Condition):

    name = "adiabatic_wall"


# ------- Domain Conditions ------- #


class PML(Condition):

    name = "pml"
