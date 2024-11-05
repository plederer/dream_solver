from __future__ import annotations
import ngsolve as ngs

from dream import bla
from dream.config import quantity, configuration, ngsdict
from dream.pde import FiniteElement
from dream.mesh import Condition


class CompressibleFiniteElement(FiniteElement, is_interface=True):

    @property
    def gfu(self) -> ngs.GridFunction:
        return self.cfg.pde.gfu

    def set_boundary_conditions(self) -> None:
        """ Boundary conditions for compressible flows are set weakly. Therefore we do nothing here."""
        pass


class flowstate(ngsdict):
    rho = quantity('density')
    u = quantity('velocity')
    rho_u = quantity('momentum')
    p = quantity('pressure')
    T = quantity('temperature')
    rho_E = quantity('energy')
    E = quantity('specific_energy')
    rho_Ei = quantity('inner_energy')
    Ei = quantity('specific_inner_energy')
    rho_Ek = quantity('kinetic_energy')
    Ek = quantity('specific_kinetic_energy')
    rho_H = quantity('enthalpy')
    H = quantity('specific_enthalpy')
    c = quantity('speed_of_sound')
    grad_rho = quantity('density_gradient')
    grad_u = quantity('velocity_gradient')
    grad_rho_u = quantity('momentum_gradient')
    grad_p = quantity('pressure_gradient')
    grad_T = quantity('temperature_gradient')
    grad_rho_E = quantity('energy_gradient')
    grad_E = quantity('specific_energy_gradient')
    grad_rho_Ei = quantity('inner_energy_gradient')
    grad_Ei = quantity('specific_inner_energy_gradient')
    grad_rho_Ek = quantity('kinetic_energy_gradient')
    grad_Ek = quantity('specific_kinetic_energy_gradient')
    grad_rho_H = quantity('enthalpy_gradient')
    grad_H = quantity('specific_enthalpy_gradient')
    grad_c = quantity('speed_of_sound_gradient')
    strain = quantity('strain_rate_tensor')


class referencestate(ngsdict):
    L = quantity("length")
    rho = quantity("density")
    rho_u = quantity("momentum")
    u = quantity("velocity")
    c = quantity("speed_of_sound")
    T = quantity("temperature")
    p = quantity("pressure")


class FarField(Condition):

    name = "farfield"

    @configuration(default=None)
    def state(self, state) -> flowstate:
        if state is not None:
            state = flowstate(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Farfield state not set!")

    @configuration(default=True)
    def identity_jacobian(self, use_identity_jacobian: bool):
        return bool(use_identity_jacobian)

    state: flowstate


class Outflow(Condition):

    name = "outflow"

    @configuration(default=None)
    def state(self, state) -> flowstate:
        if state is not None:
            if bla.is_scalar(state):
                state = flowstate(p=state)
            else:
                state = flowstate(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Outflow state not set!")

    state: flowstate


class CBC(Condition, is_interface=True):

    name = "cbc"

    @configuration(default=None)
    def state(self, state) -> flowstate:
        if state is not None:
            state = flowstate(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("CBC state not set!")

    @configuration(default="farfield")
    def target(self, target: str):

        target = str(target).lower()

        options = ["farfield", "outflow", "mass_inflow", "temperature_inflow"]
        if target not in options:
            raise ValueError(f"Invalid target '{target}'. Options are: {options}")

        return target

    @configuration(default=0.28)
    def relaxation_factor(self, factor: float):
        characteristic = ["acoustic_in", "entropy", "vorticity_0", "vorticity_1", "acoustic_out"]
        if bla.is_scalar(factor):
            characteristic = dict.fromkeys(characteristic, factor)
        elif isinstance(factor, (list, tuple)):
            characteristic = {key: value for key, value in zip(characteristic, factor)}
        elif isinstance(factor, dict):
            characteristic = {key: value for key, value in zip(characteristic, factor.values())}
        return characteristic

    @configuration(default=0.0)
    def tangential_relaxation(self, tangential_relaxation: bla.SCALAR):
        return bla.as_scalar(tangential_relaxation)

    @configuration(default=False)
    def is_viscous_fluxes(self, viscous_fluxes: bool):
        return bool(viscous_fluxes)

    def get_relaxation_matrix(self, **kwargs) -> dict[str, ngs.CF]:

        factors = self.relaxation_factor.copy()
        if self.mesh.dim == 2:
            factors.pop("vorticity_1")

        return factors

    state: flowstate
    target: str
    relaxation_factor: dict
    tangential_relaxation: ngs.CF
    is_viscous_fluxes: bool


class GRCBC(CBC):

    name = "grcbc"

    def get_relaxation_matrix(self, **kwargs) -> ngs.CF:
        dt = kwargs.get('dt', None)
        if dt is None:
            raise ValueError("Time step 'dt' not provided!")

        C = super().get_relaxation_matrix()
        return bla.diagonal(tuple(C.values()))/dt


class NSCBC(CBC):

    name = "nscbc"

    @configuration(default=1)
    def length(self, length: float):
        return length

    def get_relaxation_matrix(self, **kwargs) -> ngs.CF:
        c = kwargs.get('c', None)
        M = kwargs.get('M', None)
        if c is None or M is None:
            raise ValueError("Speed of sound 'c' and Mach number 'M' not provided!")

        sigmas = super().get_relaxation_matrix()
        eig = [c * (1 - M**2)] + self.mesh.dim * [c] + [c * (1 - M**2)]
        eig = [factor*eig/self.length for factor, eig in zip(sigmas.values(), eig)]
        return bla.diagonal(eig)

    length: float


class InviscidWall(Condition):

    name = "inviscid_wall"


class Symmetry(Condition):

    name = "symmetry"


class IsothermalWall(Condition):

    name = "isothermal_wall"

    @configuration(default=None)
    def state(self, state) -> flowstate:
        if state is not None:
            if bla.is_scalar(state):
                state = flowstate(T=state)
            else:
                state = flowstate(**state)
        return state

    @state.getter_check
    def state(self) -> None:
        if self.data['state'] is None:
            raise ValueError("Isothermal Wall state not set!")

    state: flowstate


class AdiabaticWall(Condition):

    name = "adiabatic_wall"


# ------- Domain Conditions ------- #


class PML(Condition):

    name = "pml"
