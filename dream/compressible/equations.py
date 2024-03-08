from __future__ import annotations

import logging
import ngsolve as ngs
from dream import bla

from .state import CompressibleState, ScalingState


logger = logging.getLogger(__name__)


class CompressibleEquations:

    label: str = "compressible"

    def __init__(self, cfg: CompressibleFlowConfiguration = None):
        if cfg is None:
            cfg = CompressibleFlowConfiguration()

        self.cfg = cfg

    @property
    def Reynolds_number_reference(self):
        return self.cfg.Reynolds_number/self.cfg.scaling.velocity_magnitude(self.cfg.Mach_number)

    def get_farfield_state(self, direction: tuple[float, ...] = None) -> CompressibleState:

        Ma = self.cfg.Mach_number
        INF = CompressibleState()

        INF.density = self.cfg.scaling.density()
        INF.speed_of_sound = self.cfg.scaling.speed_of_sound(Ma)
        INF.temperature = self.temperature(INF)
        INF.pressure = self.pressure(INF)

        if direction is not None:
            direction = bla.as_vector(direction)

            if not 1 <= direction.dim <= 3:
                raise ValueError(f"Invalid Dimension!")

            INF.velocity = self.cfg.scaling.velocity(direction, Ma)
            INF.inner_energy = self.inner_energy(INF)
            INF.kinetic_energy = self.kinetic_energy(INF)
            INF.energy = self.energy(INF)

        return INF

    def get_reference_state(self, direction: tuple[float, ...] = None) -> CompressibleState:

        INF = self.get_farfield_state(direction)
        INF_ = self.cfg.scaling.dimensional_infinity_values
        REF = ScalingState()

        for key, value in INF_.items():
            value_ = getattr(INF, key, None)

            if value_ is not None:

                if bla.is_vector(value_):
                    value_ = bla.inner(value_, value_)

                REF[key] = value/value_

        return REF

    def get_primitive_convective_jacobian(self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None):
        lambdas = self.characteristic_velocities(state, unit_vector, type_)
        LAMBDA = bla.diagonal(lambdas)
        return self.transform_characteristic_to_primitive(LAMBDA, state, unit_vector)

    def get_conservative_convective_jacobian(
            self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None):
        lambdas = self.characteristic_velocities(state, unit_vector, type_)
        LAMBDA = bla.diagonal(lambdas)
        return self.transform_characteristic_to_conservative(LAMBDA, state, unit_vector)

    def get_characteristic_identity(
            self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None) -> bla.MATRIX:
        lambdas = self.characteristic_velocities(state, unit_vector, type_)

        if type_ == "incoming":
            lambdas = (ngs.IfPos(-lam, 1, 0) for lam in lambdas)
        elif type_ == "outgoing":
            lambdas = (ngs.IfPos(lam, 1, 0) for lam in lambdas)
        else:
            raise ValueError(f"{str(type).capitalize()} invalid! Alternatives: {['incoming', 'outgoing']}")

        return bla.diagonal(lambdas)

    def transform_primitive_to_conservative(self, matrix: bla.MATRIX, state: CompressibleState):
        M = self.conservative_from_primitive(state)
        Minv = self.primitive_from_conservative(state)
        return M * matrix * Minv

    def transform_characteristic_to_primitive(
            self, matrix: bla.MATRIX, state: CompressibleState, unit_vector: bla.VECTOR):
        L = self.primitive_from_characteristic(state, unit_vector)
        Linv = self.characteristic_from_primitive(state, unit_vector)
        return L * matrix * Linv

    def transform_characteristic_to_conservative(
            self, matrix: bla.MATRIX, state: CompressibleState, unit_vector: bla.VECTOR):
        P = self.conservative_from_characteristic(state, unit_vector)
        Pinv = self.characteristic_from_conservative(state, unit_vector)
        return P * matrix * Pinv

    @form.equation
    def Mach_number(self, state: CompressibleState):
        u = self.velocity(state)
        c = self.speed_of_sound(state)
        return ngs.sqrt(bla.inner(u, u))/c

    @form.equation
    def Reynolds_number(self, state: CompressibleState):
        rho = self.density(state)
        u = self.velocity(state)
        mu = self.viscosity(state)
        return rho * u / mu

    @form.equation
    def convective_flux(self, state: CompressibleState) -> bla.MATRIX:
        """
        Conservative convective flux

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin form.Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        rho = self.density(state)
        rho_u = self.momentum(state)
        rho_H = self.enthalpy(state)
        u = self.velocity(state)
        p = self.pressure(state)

        flux = (rho_u, bla.outer(rho_u, rho_u)/rho + p * ngs.Id(u.dim), rho_H * u)

        return bla.as_matrix(flux, dims=(u.dim + 2, u.dim))

    @form.equation
    def diffusive_flux(self, state: CompressibleState) -> bla.MATRIX:
        """
        Conservative diffusive flux

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin form.Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        u = self.velocity(state)
        tau = self.deviatoric_stress_tensor(state)
        q = self.heat_flux(state)

        Re = self.Reynolds_number_reference
        Pr = self.cfg.Prandtl_number
        mu = self.viscosity(state)

        continuity = tuple(0 for _ in range(u.dim))

        flux = (continuity, mu/Re * tau, mu/Re * (tau*u - q/Pr))

        return bla.as_matrix(flux, dims=(u.dim + 2, u.dim))

    @form.equation
    def density(self, state: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.density(state)

    @form.equation
    def velocity(self, state: CompressibleState) -> bla.VECTOR:
        rho = state.density
        rho_u = state.momentum

        if state.is_set(rho, rho_u):
            logger.debug("Returning velocity from density and momentum.")

            if bla.is_zero(rho_u) and bla.is_zero(rho):
                return bla.as_vector((0.0 for _ in range(rho_u.dim)))

            return rho_u/rho

    @form.equation
    def momentum(self, state: CompressibleState) -> bla.VECTOR:
        rho = state.density
        u = state.velocity

        if state.is_set(rho, u):
            logger.debug("Returning momentum from density and velocity.")
            return rho * u

    @form.equation
    def pressure(self, state: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.pressure(state)

    @form.equation
    def temperature(self, state: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.temperature(state)

    @form.equation
    def inner_energy(self, state: CompressibleState) -> bla.SCALAR:
        rho_Ei = self.cfg.equation_of_state.inner_energy(state)

        if rho_Ei is None:
            rho_E = state.energy
            rho_Ek = state.kinetic_energy

            if state.is_set(rho_E, rho_Ek):
                logger.debug("Returning bla.inner energy from energy and kinetic energy.")
                return rho_E - rho_Ek

        return rho_Ei

    @form.equation
    def specific_inner_energy(self, state: CompressibleState) -> bla.SCALAR:
        Ei = self.cfg.equation_of_state.specific_inner_energy(state)

        if Ei is None:

            rho = state.density
            rho_Ei = state.inner_energy

            Ek = state.specific_kinetic_energy
            E = state.specific_energy

            if state.is_set(rho, rho_Ei):
                logger.debug("Returning specific bla.inner energy from bla.inner energy and density.")
                return rho_Ei/rho

            elif state.is_set(E, Ek):
                logger.debug(
                    "Returning specific bla.inner energy from specific energy and specific kinetic energy.")
                return E - Ek

        return Ei

    @form.equation
    def kinetic_energy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        rho_u = state.momentum
        u = state.velocity

        rho_E = state.energy
        rho_Ei = state.inner_energy

        Ek = state.specific_kinetic_energy

        if state.is_set(rho, u):
            logger.debug("Returning kinetic energy from density and velocity.")
            return 0.5 * rho * bla.inner(u, u)

        elif state.is_set(rho, rho_u):
            logger.debug("Returning kinetic energy from density and momentum.")
            return 0.5 * bla.inner(rho_u, rho_u)/rho

        elif state.is_set(rho_E, rho_Ei):
            logger.debug("Returning kinetic energy from energy and bla.inner energy.")
            return rho_E - rho_Ei

        elif state.is_set(rho, Ek):
            logger.debug("Returning kinetic energy from density and specific kinetic energy.")
            return rho * Ek

    @form.equation
    def specific_kinetic_energy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        rho_u = state.momentum
        u = state.velocity

        E = state.specific_energy
        Ei = state.specific_inner_energy
        rho_Ek = state.kinetic_energy

        if state.is_set(u):
            logger.debug("Returning specific kinetic energy from velocity.")
            return 0.5 * bla.inner(u, u)

        elif state.is_set(rho, rho_u):
            logger.debug("Returning specific kinetic energy from density and momentum.")
            return 0.5 * bla.inner(rho_u, rho_u)/rho**2

        elif state.is_set(rho, rho_Ek):
            logger.debug("Returning specific kinetic energy from density and kinetic energy.")
            return rho_Ek/rho

        elif state.is_set(E, Ei):
            logger.debug("Returning specific kinetic energy from specific energy and speicific bla.inner energy.")
            return E - Ei

    @form.equation
    def energy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        E = state.specific_energy

        rho_Ei = state.inner_energy
        rho_Ek = state.kinetic_energy

        if state.is_set(rho, E):
            logger.debug("Returning energy from density and specific energy.")
            return rho * E

        elif state.is_set(rho_Ei, rho_Ek):
            logger.debug("Returning energy from bla.inner energy and kinetic energy.")
            return rho_Ei + rho_Ek

    @form.equation
    def specific_energy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        rho_E = state.energy

        Ei = state.specific_inner_energy
        Ek = state.specific_kinetic_energy

        if state.is_set(rho, rho_E):
            logger.debug("Returning specific energy from density and energy.")
            return rho_E/rho

        elif state.is_set(Ei, Ek):
            logger.debug("Returning specific energy from specific bla.inner energy and specific kinetic energy.")
            return Ei + Ek

    @form.equation
    def enthalpy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        H = state.specific_enthalpy

        rho_E = state.energy
        p = state.pressure

        if state.is_set(rho_E, p):
            logger.debug("Returning enthalpy from energy and pressure.")
            return rho_E + p

        elif state.is_set(rho, H):
            logger.debug("Returning enthalpy from density and specific enthalpy.")
            return rho * H

    @form.equation
    def specific_enthalpy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        rho_H = state.enthalpy

        rho_E = state.energy
        E = state.specific_energy
        p = state.pressure

        if state.is_set(rho, rho_H):
            logger.debug("Returning specific enthalpy from density and enthalpy.")
            return rho_H/rho

        elif state.is_set(rho, rho_E, p):
            logger.debug("Returning specific enthalpy from specific energy, density and pressure.")
            return E + p/rho

    @form.equation
    def speed_of_sound(self, state: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.speed_of_sound(state)

    @form.equation
    def density_gradient(self, state: CompressibleState) -> bla.VECTOR:
        return self.cfg.equation_of_state.density_gradient(state)

    @form.equation
    def velocity_gradient(self, state: CompressibleState) -> bla.MATRIX:
        rho = state.density
        rho_u = state.momentum

        grad_rho = state.density_gradient
        grad_rho_u = state.momentum_gradient

        if state.is_set(rho, rho_u, grad_rho, grad_rho_u):
            logger.debug("Returning velocity gradient from density and momentum.")
            return grad_rho_u/rho - bla.outer(rho_u, grad_rho)/rho**2

    @form.equation
    def momentum_gradient(self, state: CompressibleState) -> bla.MATRIX:
        rho = state.density
        u = state.velocity

        grad_rho = state.density_gradient
        grad_u = state.velocity_gradient

        if state.is_set(rho, u, grad_rho, grad_u):
            logger.debug("Returning momentum gradient from density and momentum.")
            return rho * grad_u + bla.outer(u, grad_rho)

    @form.equation
    def pressure_gradient(self, state: CompressibleState) -> bla.VECTOR:
        return self.cfg.equation_of_state.pressure_gradient(state)

    @form.equation
    def temperature_gradient(self, state: CompressibleState) -> bla.VECTOR:
        return self.cfg.equation_of_state.temperature_gradient(state)

    @form.equation
    def energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_rho_Ei = state.energy_gradient
        grad_rho_Ek = state.kinetic_energy_gradient

        if state.is_set(grad_rho_Ei, grad_rho_Ek):
            logger.debug("Returning energy gradient from bla.inner energy and kinetic energy.")
            return grad_rho_Ei + grad_rho_Ek

    @form.equation
    def specific_energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_Ei = state.specific_energy_gradient
        grad_Ek = state.specific_kinetic_energy_gradient

        if state.is_set(grad_Ei, grad_Ek):
            logger.debug(
                "Returning specific energy gradient from specific bla.inner energy and specific kinetic energy.")
            return grad_Ei + grad_Ek

    @form.equation
    def inner_energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_rho_E = state.energy_gradient
        grad_rho_Ek = state.kinetic_energy_gradient

        if state.is_set(grad_rho_E, grad_rho_Ek):
            logger.debug("Returning bla.inner energy gradient from energy and kinetic energy.")
            return grad_rho_E - grad_rho_Ek

    @form.equation
    def specific_inner_energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_E = state.specific_energy_gradient
        grad_Ek = state.specific_kinetic_energy_gradient

        if state.is_set(grad_E, grad_Ek):
            logger.debug(
                "Returning specific bla.inner energy gradient from specific energy and specific kinetic energy.")
            return grad_E - grad_Ek

    @form.equation
    def kinetic_energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_rho_E = state.energy_gradient
        grad_rho_Ei = state.inner_energy_gradient

        rho = state.density
        grad_rho = state.density_gradient
        u = state.velocity
        grad_u = state.velocity_gradient

        rho_u = state.momentum
        grad_rho_u = state.momentum_gradient

        if state.is_set(grad_rho_E, grad_rho_Ei):
            logger.debug("Returning kinetic energy gradient from energy and bla.inner energy.")
            return grad_rho_E - grad_rho_Ei

        elif state.is_set(rho, u, grad_rho, grad_u):
            logger.debug("Returning kinetic energy gradient from density and velocity.")
            return rho * (grad_u.trans * u) + 0.5 * grad_rho * bla.inner(u, u)

        elif state.is_set(rho, rho_u, grad_rho, grad_rho_u):
            logger.debug("Returning kinetic energy gradient from density and momentum.")
            return (grad_rho_u.trans * rho_u)/rho - 0.5 * grad_rho * bla.inner(rho_u, rho_u)/rho**2

    @form.equation
    def specific_kinetic_energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_E = state.specific_energy_gradient
        grad_Ei = state.specific_inner_energy_gradient

        rho = state.density
        grad_rho = state.density_gradient
        u = state.velocity
        grad_u = state.velocity_gradient

        rho_u = state.momentum
        grad_rho_u = state.momentum_gradient

        if state.is_set(grad_E, grad_Ei):
            logger.debug(
                "Returning specific kinetic energy gradient from specific energy and specific bla.inner energy.")
            return grad_E - grad_Ei

        elif state.is_set(u, grad_u):
            logger.debug("Returning specific kinetic energy gradient from velocity.")
            return grad_u.trans * u

        elif state.is_set(rho, rho_u, grad_rho, grad_rho_u):
            logger.debug("Returning specific kinetic energy gradient from density and momentum.")
            return (grad_rho_u.trans * rho_u)/rho**2 - grad_rho * bla.inner(rho_u, rho_u)/rho**3

    @form.equation
    def enthalpy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_rho_E = state.energy_gradient
        grad_p = state.pressure_gradient

        if state.is_set(grad_rho_E, grad_p):
            logger.debug("Returning enthalpy gradient from energy and pressure.")
            return grad_rho_E + grad_p

    @form.equation
    def strain_rate_tensor(self, state: CompressibleState) -> bla.MATRIX:
        grad_u = self.velocity_gradient(state)

        if state.is_set(grad_u):
            logger.debug("Returning strain rate tensor from velocity.")
            return 0.5 * (grad_u + grad_u.trans) - 1/3 * bla.trace(grad_u, id=True)

    @form.equation
    def deviatoric_stress_tensor(self, state: CompressibleState):

        mu = state.viscosity
        EPS = state.strain_rate_tensor

        if state.is_set(mu, EPS):
            logger.debug("Returning deviatoric stress tensor from strain rate tensor and viscosity.")
            return 2 * mu * EPS

    @form.equation
    def viscosity(self, state: CompressibleState) -> bla.SCALAR:
        return self.cfg.dynamic_viscosity.viscosity(state, self)

    @form.equation
    def heat_flux(self, state: CompressibleState) -> bla.VECTOR:

        gradient_T = state.temperature_gradient
        k = state.viscosity

        if state.is_set(k, gradient_T):
            return -k * gradient_T

    @form.equation
    def characteristic_velocities(
            self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None) -> bla.VECTOR:
        return self.cfg.equation_of_state.characteristic_velocities(state, unit_vector, type_)

    @form.equation
    def characteristic_variables(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.VECTOR:
        return self.cfg.equation_of_state.characteristic_variables(state, unit_vector)

    @form.equation
    def characteristic_amplitudes(
            self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None) -> bla.VECTOR:
        return self.cfg.equation_of_state.characteristic_amplitudes(state, unit_vector, type_)

    @form.equation
    def primitive_from_conservative(self, state: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.primitive_from_conservative(state)

    @form.equation
    def primitive_from_characteristic(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.cfg.equation_of_state.primitive_from_characteristic(state, unit_vector)

    @form.equation
    def primitive_convective_jacobian_x(self, state: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.primitive_convective_jacobian_x(state)

    @form.equation
    def primitive_convective_jacobian_y(self, state: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.primitive_convective_jacobian_y(state)

    @form.equation
    def conservative_from_primitive(self, state: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.conservative_from_primitive(state)

    @form.equation
    def conservative_from_characteristic(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.cfg.equation_of_state.conservative_from_characteristic(state, unit_vector)

    @form.equation
    def conservative_convective_jacobian_x(self, state: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.conservative_convective_jacobian_x(state)

    @form.equation
    def conservative_convective_jacobian_y(self, state: CompressibleState) -> bla.MATRIX:
        return self.cfg.equation_of_state.conservative_convective_jacobian_y(state)

    @form.equation
    def characteristic_from_primitive(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.cfg.equation_of_state.characteristic_from_primitive(state, unit_vector)

    @form.equation
    def characteristic_from_conservative(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.cfg.equation_of_state.characteristic_from_conservative(state, unit_vector)
