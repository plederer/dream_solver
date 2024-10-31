from __future__ import annotations
import ngsolve as ngs

from dream import bla
from dream.config import quantity, State, any
from dream.pde import FiniteElement
from dream.mesh import Condition


class CompressibleFiniteElement(FiniteElement, is_interface=True):

    @property
    def gfu(self) -> ngs.GridFunction:
        return self.cfg.pde.gfu

    @any(default=2)
    def order(self, order):
        return int(order)

    order: int

    def set_boundary_conditions(self) -> None:
        """ Boundary conditions for compressible flows are set weakly. Therefore we do nothing here."""
        pass


class CompressibleState(State):

    rho = quantity(bla.as_scalar, 'density')
    u = quantity(bla.as_vector, 'velocity')
    rho_u = quantity(bla.as_vector, 'momentum')
    p = quantity(bla.as_scalar, 'pressure')
    T = quantity(bla.as_scalar, 'temperature')
    rho_E = quantity(bla.as_scalar, 'energy')
    E = quantity(bla.as_scalar, 'specific_energy')
    rho_Ei = quantity(bla.as_scalar, 'inner_energy')
    Ei = quantity(bla.as_scalar, 'specific_inner_energy')
    rho_Ek = quantity(bla.as_scalar, 'kinetic_energy')
    Ek = quantity(bla.as_scalar, 'specific_kinetic_energy')
    rho_H = quantity(bla.as_scalar, 'enthalpy')
    H = quantity(bla.as_scalar, 'specific_enthalpy')
    c = quantity(bla.as_scalar, 'speed_of_sound')


class CompressibleStateGradient(State):

    grad_rho = quantity(bla.as_vector, 'density_gradient')
    grad_u = quantity(bla.as_matrix, 'velocity_gradient')
    grad_rho_u = quantity(bla.as_matrix, 'momentum_gradient')
    grad_p = quantity(bla.as_vector, 'pressure_gradient')
    grad_T = quantity(bla.as_vector, 'temperature_gradient')
    grad_rho_E = quantity(bla.as_vector, 'energy_gradient')
    grad_E = quantity(bla.as_vector, 'specific_energy_gradient')
    grad_rho_Ei = quantity(bla.as_vector, 'inner_energy_gradient')
    grad_Ei = quantity(bla.as_vector, 'specific_inner_energy_gradient')
    grad_rho_Ek = quantity(bla.as_vector, 'kinetic_energy_gradient')
    grad_Ek = quantity(bla.as_vector, 'specific_kinetic_energy_gradient')
    grad_rho_H = quantity(bla.as_vector, 'enthalpy_gradient')
    grad_H = quantity(bla.as_vector, 'specific_enthalpy_gradient')
    grad_c = quantity(bla.as_vector, 'speed_of_sound_gradient')

    strain = quantity(bla.as_matrix, 'strain_rate_tensor')


class ReferenceState(State):

    L = quantity(bla.as_scalar, "length")
    rho = quantity(bla.as_scalar, "density")
    rho_u = quantity(bla.as_scalar, "momentum")
    u = quantity(bla.as_scalar, "velocity")
    c = quantity(bla.as_scalar, "speed_of_sound")
    T = quantity(bla.as_scalar, "temperature")
    p = quantity(bla.as_scalar, "pressure")


class FarField(Condition):

    name = "farfield"

    @any(default=None)
    def state(self, state) -> CompressibleState:
        if state is not None:
            state = CompressibleState(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Farfield state not set!")

    state: CompressibleState


class Outflow(Condition):

    name = "outflow"

    @any(default=None)
    def state(self, state) -> CompressibleState:
        if state is not None:
            if bla.is_scalar(state):
                state = CompressibleState(p=state)
            else:
                state = CompressibleState(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Outflow state not set!")

    state: CompressibleState


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
        return parameters

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
    def state(self, state) -> CompressibleState:
        if state is not None:
            if bla.is_scalar(state):
                state = CompressibleState(T=state)
            else:
                state = CompressibleState(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Isothermal Wall state not set!")

    state: CompressibleState


class AdiabaticWall(Condition):

    name = "adiabatic_wall"


# ------- Domain Conditions ------- #


class PML(Condition):

    name = "pml"
