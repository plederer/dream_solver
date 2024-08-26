from __future__ import annotations

import logging
import ngsolve as ngs

from dream import bla
from dream.config import any, parameter, descriptor_configuration, equation
from dream.formulation import PDEConfiguration
from dream.compressible.state import CompressibleState, CompressibleStateGradient, ReferenceState
from dream.compressible.eos import IdealGas
from dream.compressible.viscosity import Inviscid, Constant, Sutherland
from dream.compressible.scaling import Aerodynamic, Aeroacoustic, Acoustic
from dream.compressible.riemann_solver import LaxFriedrich, Roe, HLL, HLLEM
from dream.compressible.conditions import CompressibleBC, CompressibleDC
from dream.compressible.formulations import Conservative, CompressibleFormulation

logger = logging.getLogger(__name__)


class CompressibleFlowConfiguration(PDEConfiguration):

    name = "compressible"

    bcs = CompressibleBC
    dcs = CompressibleDC

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.equations = CompressibleEquations(self)

    @descriptor_configuration(Conservative)
    def formulation(self, formulation) -> CompressibleFormulation:
        return formulation

    @parameter(default=0.3)
    def mach_number(self, mach_number: float):

        if mach_number < 0:
            raise ValueError("Invalid Mach number. Value has to be >= 0!")

        return mach_number

    @parameter(default=1)
    def reynolds_number(self, reynolds_number: float):
        """ Represents the ratio between inertial and viscous forces """
        if reynolds_number <= 0:
            raise ValueError("Invalid Reynold number. Value has to be > 0!")

        return reynolds_number

    @parameter(default=0.72)
    def prandtl_number(self, prandtl_number: float):
        if prandtl_number <= 0:
            raise ValueError("Invalid Prandtl_number. Value has to be > 0!")

        return prandtl_number

    @descriptor_configuration(default=IdealGas)
    def equation_of_state(self, equation_of_state):
        return equation_of_state

    @descriptor_configuration(default=Inviscid)
    def dynamic_viscosity(self, dynamic_viscosity):
        return dynamic_viscosity

    @descriptor_configuration(default=Aerodynamic)
    def scaling(self, scaling):
        return scaling

    @descriptor_configuration(default=LaxFriedrich)
    def riemann_solver(self, riemann_solver: LaxFriedrich):
        self.riemann_solver._cfg = self
        return riemann_solver

    @reynolds_number.getter_check
    def reynolds_number(self):
        if self.dynamic_viscosity.is_inviscid:
            raise ValueError("Inviscid solver configuration: Reynolds number not applicable")

    @prandtl_number.getter_check
    def prandtl_number(self):
        if self.dynamic_viscosity.is_inviscid:
            raise ValueError("Inviscid solver configuration: Prandtl number not applicable")

    @property
    def reference_reynolds_number(self):
        return self.reynolds_number/self.scaling.velocity_magnitude(self.mach_number)

    def get_farfield_state(self, direction: tuple[float, ...]) -> CompressibleState:
        return self.equations.get_farfield_state(direction)

    mach_number: ngs.Parameter
    reynolds_number: ngs.Parameter
    prandtl_number: ngs.Parameter
    equation_of_state: IdealGas
    dynamic_viscosity: Constant | Inviscid | Sutherland
    scaling: Aerodynamic | Acoustic | Aeroacoustic
    riemann_solver: LaxFriedrich | Roe | HLL | HLLEM


class CompressibleEquations:

    def __init__(self, cfg: CompressibleFlowConfiguration = None):
        if cfg is None:
            cfg = CompressibleFlowConfiguration()

        self.cfg = cfg

    def get_farfield_state(self, direction: tuple[float, ...] = None) -> CompressibleState:

        Ma = self.cfg.mach_number
        INF = CompressibleState()

        INF.rho = self.cfg.scaling.density()
        INF.c = self.cfg.scaling.speed_of_sound(Ma)
        INF.T = self.temperature(INF)
        INF.p = self.pressure(INF)

        if direction is not None:
            direction = bla.as_vector(direction)

            if not 1 <= direction.dim <= 3:
                raise ValueError(f"Invalid Dimension!")

            INF.u = self.cfg.scaling.velocity(direction, Ma)
            INF.rho_Ei = self.inner_energy(INF)
            INF.rho_Ek = self.kinetic_energy(INF)
            INF.rho_E = self.energy(INF)

        return INF

    def get_reference_state(self, direction: tuple[float, ...] = None) -> CompressibleState:

        INF = self.get_farfield_state(direction)
        INF_ = self.cfg.scaling.reference_values
        REF = ReferenceState()

        for key, value in INF_.items():
            value_ = INF.get(key, None)

            if value_ is not None:

                if bla.is_vector(value_):
                    value_ = bla.inner(value_, value_)

                REF[key] = value/value_

        return REF

    def get_primitive_convective_jacobian(self, U: CompressibleState, unit_vector: bla.VECTOR, type: str = None):
        lambdas = self.characteristic_velocities(U, unit_vector, type)
        LAMBDA = bla.diagonal(lambdas)
        return self.transform_characteristic_to_primitive(LAMBDA, U, unit_vector)

    def get_conservative_convective_jacobian(self, U: CompressibleState, unit_vector: bla.VECTOR, type: str = None):
        lambdas = self.characteristic_velocities(U, unit_vector, type)
        LAMBDA = bla.diagonal(lambdas)
        return self.transform_characteristic_to_conservative(LAMBDA, U, unit_vector)

    def get_characteristic_identity(
            self, U: CompressibleState, unit_vector: bla.VECTOR, type: str = None) -> bla.MATRIX:
        lambdas = self.characteristic_velocities(U, unit_vector, type)

        if type == "incoming":
            lambdas = (ngs.IfPos(-lam, 1, 0) for lam in lambdas)
        elif type == "outgoing":
            lambdas = (ngs.IfPos(lam, 1, 0) for lam in lambdas)
        else:
            raise ValueError(f"{str(type).capitalize()} invalid! Alternatives: {['incoming', 'outgoing']}")

        return bla.diagonal(lambdas)

    def transform_primitive_to_conservative(self, matrix: bla.MATRIX, U: CompressibleState):
        M = self.conservative_from_primitive(U)
        Minv = self.primitive_from_conservative(U)
        return M * matrix * Minv

    def transform_characteristic_to_primitive(self, matrix: bla.MATRIX, U: CompressibleState, unit_vector: bla.VECTOR):
        L = self.primitive_from_characteristic(U, unit_vector)
        Linv = self.characteristic_from_primitive(U, unit_vector)
        return L * matrix * Linv

    def transform_characteristic_to_conservative(
            self, matrix: bla.MATRIX, U: CompressibleState, unit_vector: bla.VECTOR):
        P = self.conservative_from_characteristic(U, unit_vector)
        Pinv = self.characteristic_from_conservative(U, unit_vector)
        return P * matrix * Pinv

    @equation
    def mach_number(self, U: CompressibleState):
        u = self.velocity(U)
        c = self.speed_of_sound(U)
        return ngs.sqrt(bla.inner(u, u))/c

    @equation
    def reynolds_number(self, U: CompressibleState):
        rho = self.density(U)
        u = self.velocity(U)
        mu = self.viscosity(U)
        return rho * u / mu

    @equation
    def convective_flux(self, U: CompressibleState) -> bla.MATRIX:
        """
        Conservative convective flux

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin form.Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_H = self.enthalpy(U)
        u = self.velocity(U)
        p = self.pressure(U)

        flux = (rho_u, bla.outer(rho_u, rho_u)/rho + p * ngs.Id(u.dim), rho_H * u)

        return bla.as_matrix(flux, dims=(u.dim + 2, u.dim))

    @equation
    def diffusive_flux(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.MATRIX:
        """
        Conservative diffusive flux

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin form.Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        u = self.velocity(U)
        tau = self.deviatoric_stress_tensor(U, dU)
        q = self.heat_flux(U, dU)

        Re = self.cfg.reference_reynolds_number
        Pr = self.cfg.prandtl_number

        continuity = tuple(0 for _ in range(u.dim))

        flux = (continuity, tau/Re, (tau*u - q/Pr)/Re)

        return bla.as_matrix(flux, dims=(u.dim + 2, u.dim))

    @equation
    def density(self, U: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.density(U)

    @equation
    def velocity(self, U: CompressibleState) -> bla.VECTOR:
        if U.is_set(U.rho, U.rho_u):
            logger.debug("Returning velocity from density and momentum.")

            if bla.is_zero(U.rho_u) and bla.is_zero(U.rho):
                return bla.as_vector((0.0 for _ in range(U.rho_u.dim)))

            return U.rho_u/U.rho

    @equation
    def momentum(self, U: CompressibleState) -> bla.VECTOR:
        if U.is_set(U.rho, U.u):
            logger.debug("Returning momentum from density and velocity.")
            return U.rho * U.u

    @equation
    def pressure(self, U: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.pressure(U)

    @equation
    def temperature(self, U: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.temperature(U)

    @equation
    def inner_energy(self, U: CompressibleState) -> bla.SCALAR:
        rho_Ei = self.cfg.equation_of_state.inner_energy(U)

        if rho_Ei is None:

            if U.is_set(U.rho_E, U.rho_Ek):
                logger.debug("Returning inner energy from energy and kinetic energy.")
                return U.rho_E - U.rho_Ek

        return rho_Ei

    @equation
    def specific_inner_energy(self, U: CompressibleState) -> bla.SCALAR:
        Ei = self.cfg.equation_of_state.specific_inner_energy(U)

        if Ei is None:

            if U.is_set(U.rho, U.rho_Ei):
                logger.debug("Returning specific inner energy from inner energy and density.")
                return U.rho_Ei/U.rho

            elif U.is_set(U.E, U.Ek):
                logger.debug(
                    "Returning specific inner energy from specific energy and specific kinetic energy.")
                return U.E - U.Ek

        return Ei

    @equation
    def kinetic_energy(self, U: CompressibleState) -> bla.SCALAR:
        if U.is_set(U.rho, U.u):
            logger.debug("Returning kinetic energy from density and velocity.")
            return 0.5 * U.rho * bla.inner(U.u, U.u)

        elif U.is_set(U.rho, U.rho_u):
            logger.debug("Returning kinetic energy from density and momentum.")
            return 0.5 * bla.inner(U.rho_u, U.rho_u)/U.rho

        elif U.is_set(U.rho_E, U.rho_Ei):
            logger.debug("Returning kinetic energy from energy and inner energy.")
            return U.rho_E - U.rho_Ei

        elif U.is_set(U.rho, U.Ek):
            logger.debug("Returning kinetic energy from density and specific kinetic energy.")
            return U.rho * U.Ek

    @equation
    def specific_kinetic_energy(self, U: CompressibleState) -> bla.SCALAR:
        if U.is_set(U.u):
            logger.debug("Returning specific kinetic energy from velocity.")
            return 0.5 * bla.inner(U.u, U.u)

        elif U.is_set(U.rho, U.rho_u):
            logger.debug("Returning specific kinetic energy from density and momentum.")
            return 0.5 * bla.inner(U.rho_u, U.rho_u)/U.rho**2

        elif U.is_set(U.rho, U.rho_Ek):
            logger.debug("Returning specific kinetic energy from density and kinetic energy.")
            return U.rho_Ek/U.rho

        elif U.is_set(U.E, U.Ei):
            logger.debug("Returning specific kinetic energy from specific energy and speicific inner energy.")
            return U.E - U.Ei

    @equation
    def energy(self, U: CompressibleState) -> bla.SCALAR:
        if U.is_set(U.rho, U.E):
            logger.debug("Returning energy from density and specific energy.")
            return U.rho * U.E

        elif U.is_set(U.rho_Ei, U.rho_Ek):
            logger.debug("Returning energy from inner energy and kinetic energy.")
            return U.rho_Ei + U.rho_Ek

    @equation
    def specific_energy(self, U: CompressibleState) -> bla.SCALAR:
        if U.is_set(U.rho, U.rho_E):
            logger.debug("Returning specific energy from density and energy.")
            return U.rho_E/U.rho

        elif U.is_set(U.Ei, U.Ek):
            logger.debug("Returning specific energy from specific inner energy and specific kinetic energy.")
            return U.Ei + U.Ek

    @equation
    def enthalpy(self, U: CompressibleState) -> bla.SCALAR:
        if U.is_set(U.rho_E, U.p):
            logger.debug("Returning enthalpy from energy and pressure.")
            return U.rho_E + U.p

        elif U.is_set(U.rho, U.H):
            logger.debug("Returning enthalpy from density and specific enthalpy.")
            return U.rho * U.H

    @equation
    def specific_enthalpy(self, U: CompressibleState) -> bla.SCALAR:
        if U.is_set(U.rho, U.rho_H):
            logger.debug("Returning specific enthalpy from density and enthalpy.")
            return U.rho_H/U.rho

        elif U.is_set(U.rho, U.rho_E, U.p):
            logger.debug("Returning specific enthalpy from specific energy, density and pressure.")
            return U.E + U.p/U.rho

    @equation
    def speed_of_sound(self, U: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.speed_of_sound(U)

    @equation
    def density_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        return self.cfg.equation_of_state.density_gradient(U, dU)

    @equation
    def velocity_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.MATRIX:
        if U.is_set(U.rho, U.rho_u, dU.rho, dU.rho_u):
            logger.debug("Returning velocity gradient from density and momentum.")
            return dU.rho_u/U.rho - bla.outer(U.rho_u, dU.rho)/U.rho**2

    @equation
    def momentum_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.MATRIX:
        if U.is_set(U.rho, U.u, dU.rho, dU.u):
            logger.debug("Returning momentum gradient from density and momentum.")
            return U.rho * dU.u + bla.outer(U.u, dU.rho)

    @equation
    def pressure_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        return self.cfg.equation_of_state.pressure_gradient(U, dU)

    @equation
    def temperature_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        return self.cfg.equation_of_state.temperature_gradient(U, dU)

    @equation
    def energy_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        if U.is_set(dU.rho_Ei, dU.rho_Ek):
            logger.debug("Returning energy gradient from inner energy and kinetic energy.")
            return dU.rho_Ei + dU.rho_Ek

    @equation
    def specific_energy_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        if U.is_set(dU.Ei, dU.Ek):
            logger.debug(
                "Returning specific energy gradient from specific inner energy and specific kinetic energy.")
            return dU.Ei + dU.Ek

    @equation
    def inner_energy_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        if U.is_set(dU.rho_E, dU.rho_Ek):
            logger.debug("Returning inner energy gradient from energy and kinetic energy.")
            return dU.rho_E - dU.rho_Ek

    @equation
    def specific_inner_energy_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        if U.is_set(dU.E, dU.Ek):
            logger.debug(
                "Returning specific inner energy gradient from specific energy and specific kinetic energy.")
            return dU.E - dU.Ek

    @equation
    def kinetic_energy_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        if U.is_set(dU.rho_E, dU.rho_Ei):
            logger.debug("Returning kinetic energy gradient from energy and inner energy.")
            return dU.rho_E - dU.rho_Ei

        elif U.is_set(U.rho, U.u, dU.rho, dU.u):
            logger.debug("Returning kinetic energy gradient from density and velocity.")
            return U.rho * (dU.u.trans * U.u) + 0.5 * dU.rho * bla.inner(U.u, U.u)

        elif U.is_set(U.rho, U.rho_u, dU.rho, dU.rho_u):
            logger.debug("Returning kinetic energy gradient from density and momentum.")
            return (dU.rho_u.trans * U.rho_u)/U.rho - 0.5 * dU.rho * bla.inner(U.rho_u, U.rho_u)/U.rho**2

    @equation
    def specific_kinetic_energy_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        if U.is_set(dU.E, dU.Ei):
            logger.debug(
                "Returning specific kinetic energy gradient from specific energy and specific inner energy.")
            return dU.E - dU.Ei

        elif U.is_set(U.u, dU.u):
            logger.debug("Returning specific kinetic energy gradient from velocity.")
            return dU.u.trans * U.u

        elif U.is_set(U.rho, U.rho_u, dU.rho, dU.rho_u):
            logger.debug("Returning specific kinetic energy gradient from density and momentum.")
            return (dU.rho_u.trans * U.rho_u)/U.rho**2 - dU.rho * bla.inner(U.rho_u, U.rho_u)/U.rho**3

    @equation
    def enthalpy_gradient(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:
        if U.is_set(dU.rho_E, dU.p):
            logger.debug("Returning enthalpy gradient from energy and pressure.")
            return dU.rho_E + dU.p

    @equation
    def strain_rate_tensor(self, dU: CompressibleStateGradient) -> bla.MATRIX:
        if dU.is_set(dU.u):
            logger.debug("Returning strain rate tensor from velocity.")
            return 0.5 * (dU.u + dU.u.trans) - 1/3 * bla.trace(dU.u, id=True)

    @equation
    def deviatoric_stress_tensor(self, U: CompressibleState, dU: CompressibleStateGradient):

        mu = self.viscosity(U)
        eps = self.strain_rate_tensor(dU)

        if U.is_set(mu, eps):
            logger.debug("Returning deviatoric stress tensor from strain rate tensor and viscosity.")
            return 2 * mu * eps

    @equation
    def viscosity(self, U: CompressibleState) -> bla.SCALAR:
        return self.cfg.dynamic_viscosity.viscosity(U, self)

    @equation
    def heat_flux(self, U: CompressibleState, dU: CompressibleStateGradient) -> bla.VECTOR:

        k = self.viscosity(U)

        if U.is_set(k, dU.T):
            logger.debug("Returning heat flux from temperature gradient.")
            return -k * dU.T

    @equation
    def characteristic_velocities(self, U: CompressibleState, unit_vector: bla.VECTOR, type: str = None) -> bla.VECTOR:
        return self.cfg.equation_of_state.characteristic_velocities(U, unit_vector, type)

    @equation
    def characteristic_variables(
            self, U: CompressibleState, dU: CompressibleStateGradient, unit_vector: bla.VECTOR) -> bla.VECTOR:
        return self.cfg.equation_of_state.characteristic_variables(U, dU, unit_vector)

    @equation
    def characteristic_amplitudes(self, U: CompressibleState, dU: CompressibleStateGradient, unit_vector: bla.VECTOR,
                                  type: str = None) -> bla.VECTOR:
        return self.cfg.equation_of_state.characteristic_amplitudes(U, dU, unit_vector, type)

    @equation
    def primitive_from_conservative(self, U: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.primitive_from_conservative(U)

    @equation
    def primitive_from_characteristic(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.cfg.equation_of_state.primitive_from_characteristic(U, unit_vector)

    @equation
    def primitive_convective_jacobian_x(self, U: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.primitive_convective_jacobian_x(U)

    @equation
    def primitive_convective_jacobian_y(self, U: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.primitive_convective_jacobian_y(U)

    @equation
    def conservative_from_primitive(self, U: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.conservative_from_primitive(U)

    @equation
    def conservative_from_characteristic(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.cfg.equation_of_state.conservative_from_characteristic(U, unit_vector)

    @equation
    def conservative_convective_jacobian_x(self, U: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.conservative_convective_jacobian_x(U)

    @equation
    def conservative_convective_jacobian_y(self, U: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.conservative_convective_jacobian_y(U)

    @equation
    def characteristic_from_primitive(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.cfg.equation_of_state.characteristic_from_primitive(U, unit_vector)

    @equation
    def characteristic_from_conservative(self, U: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.cfg.equation_of_state.characteristic_from_conservative(U, unit_vector)
