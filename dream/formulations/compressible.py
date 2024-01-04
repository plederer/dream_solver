from __future__ import annotations
import typing
import ngsolve as ngs

from dream import bla
from dream import mesh as dmesh
from dream.config import State, Variable_
from dream.time_schemes import TransientGridfunction

from . import formulation as form

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


# ------- CompressibleState -------- #

class CompressibleState(State):

    density = Variable_(bla.as_scalar)
    velocity = Variable_(bla.as_vector)
    momentum = Variable_(bla.as_vector)
    pressure = Variable_(bla.as_scalar)
    temperature = Variable_(bla.as_scalar)
    energy = Variable_(bla.as_scalar)
    specific_energy = Variable_(bla.as_scalar)
    inner_energy = Variable_(bla.as_scalar)
    specific_inner_energy = Variable_(bla.as_scalar)
    kinetic_energy = Variable_(bla.as_scalar)
    specific_kinetic_energy = Variable_(bla.as_scalar)
    enthalpy = Variable_(bla.as_scalar)
    specific_enthalpy = Variable_(bla.as_scalar)
    speed_of_sound = Variable_(bla.as_scalar)

    viscosity = Variable_(bla.as_scalar)
    strain_rate_tensor = Variable_(bla.as_matrix)
    deviatoric_stress_tensor = Variable_(bla.as_matrix)
    heat_flux = Variable_(bla.as_vector)

    density_gradient = Variable_(bla.as_vector)
    velocity_gradient = Variable_(bla.as_matrix)
    momentum_gradient = Variable_(bla.as_matrix)
    pressure_gradient = Variable_(bla.as_vector)
    temperature_gradient = Variable_(bla.as_vector)
    energy_gradient = Variable_(bla.as_vector)
    specific_energy_gradient = Variable_(bla.as_vector)
    inner_energy_gradient = Variable_(bla.as_vector)
    specific_inner_energy_gradient = Variable_(bla.as_vector)
    kinetic_energy_gradient = Variable_(bla.as_vector)
    specific_kinetic_energy_gradient = Variable_(bla.as_vector)
    enthalpy_gradient = Variable_(bla.as_vector)
    specific_enthalpy_gradient = Variable_(bla.as_vector)


class ScalingState(CompressibleState):

    length = Variable_(bla.as_scalar)
    density = Variable_(bla.as_scalar)
    momentum = Variable_(bla.as_scalar)
    velocity = Variable_(bla.as_scalar)
    speed_of_sound = Variable_(bla.as_scalar)
    temperature = Variable_(bla.as_scalar)
    pressure = Variable_(bla.as_scalar)
    energy = Variable_(bla.as_scalar)
    inner_energy = Variable_(bla.as_scalar)
    kinetic_energy = Variable_(bla.as_scalar)

# ------- Dynamic Configuration ------- #


class MixedMethod:

    types: dict[str, MixedMethod] = {}

    def __init_subclass__(cls, label: str) -> None:
        cls.types[label] = cls
        cls.types[str(None)] = cls

    def get_mixed_space(self, cfg: SolverConfiguration, dmesh: dmesh.DreamMesh) -> None:
        if cfg.flow.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Viscous configuration requires mixed method!")
        return None


class StrainHeat(MixedMethod, label="strain_heat"):

    def get_mixed_space(self, cfg: SolverConfiguration, dmesh: dmesh.DreamMesh) -> StrainHeatSpace:
        if cfg.flow.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        return StrainHeatSpace(cfg, dmesh)


class Gradient(MixedMethod, label="gradient"):

    def get_mixed_space(self, cfg: SolverConfiguration, dmesh: dmesh.DreamMesh) -> GradientSpace:

        if cfg.flow.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid configuration does not require mixed method!")

        return GradientSpace(cfg, dmesh)


# ------- Dynamic Equations ------- #


class EquationOfState(form.DynamicEquations):

    def density(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def pressure(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def temperature(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def inner_energy(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def specific_inner_energy(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def speed_of_sound(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def density_gradient(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def pressure_gradient(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def temperature_gradient(self, state: CompressibleState) -> ngs.CF:
        raise NotImplementedError()

    def characteristic_velocities(
            self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None) -> bla.VECTOR:
        raise NotImplementedError()

    def characteristic_variables(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.VECTOR:
        raise NotImplementedError()

    def characteristic_amplitudes(
            self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None) -> bla.VECTOR:
        raise NotImplementedError()

    def primitive_from_conservative(self, state: CompressibleState) -> bla.MATRIX:
        raise NotImplementedError()

    def primitive_from_characteristic(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        raise NotImplementedError()

    def primitive_convective_jacobian_x(self, state: CompressibleState) -> bla.MATRIX:
        raise NotImplementedError()

    def primitive_convective_jacobian_y(self, state: CompressibleState) -> bla.MATRIX:
        raise NotImplementedError()

    def conservative_from_primitive(self, state: CompressibleState) -> bla.MATRIX:
        raise NotImplementedError()

    def conservative_from_characteristic(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        raise NotImplementedError()

    def conservative_convective_jacobian_x(self, state: CompressibleState) -> bla.MATRIX:
        raise NotImplementedError()

    def conservative_convective_jacobian_y(self, state: CompressibleState) -> bla.MATRIX:
        raise NotImplementedError()

    def characteristic_from_primitive(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        raise NotImplementedError()

    def characteristic_from_conservative(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        raise NotImplementedError()

    def format(self):
        formatter = self.formatter.new()
        formatter.entry('Equation of CompressibleState', str(self))
        return formatter.output


class IdealGas(EquationOfState, labels=["ideal", "perfect"]):

    def __init__(self, heat_capacity_ratio: float = 1.4) -> None:
        self._heat_capacity_ratio = ngs.Parameter(heat_capacity_ratio)

    @property
    def heat_capacity_ratio(self):
        return self._heat_capacity_ratio

    @heat_capacity_ratio.setter
    def heat_capacity_ratio(self, value):
        if isinstance(value, ngs.Parameter):
            value = value.Get()
        self._heat_capacity_ratio.Set(value)

    def density(self, state: CompressibleState) -> bla.SCALAR:
        r"""Returns the density from a given state

        .. math::
            \rho = \frac{\gamma}{\gamma - 1} \frac{p}{T}
            \rho = \gamma \frac{\rho E_i}{T}
            \rho = \gamma \frac{p}{c^2}
        """

        gamma = self.heat_capacity_ratio

        p = state.pressure
        T = state.temperature
        c = state.speed_of_sound
        rho_Ei = state.inner_energy

        if state.is_set(state.pressure, state.temperature):
            self.logger.debug("Returning density from pressure and temperature.")
            return gamma/(gamma - 1) * p/T

        elif state.is_set(p, c):
            self.logger.debug("Returning density from pressure and speed of sound.")
            return gamma * p/c**2

        elif state.is_set(rho_Ei, T):
            self.logger.debug("Returning density from inner energy and temperature.")
            return gamma * rho_Ei/T

    def pressure(self, state: CompressibleState) -> bla.SCALAR:
        r"""Returns the density from a given state

        .. math::
            p = \frac{\gamma - 1}{\gamma} \rho T
            p = (\gamma - 1) \rho E_i
            p = \rho \frac{c^2}{\gamma}
        """

        gamma = self.heat_capacity_ratio

        rho = state.density
        T = state.temperature
        c = state.speed_of_sound
        rho_Ei = state.inner_energy

        if state.is_set(rho, T):
            self.logger.debug("Returning pressure from density and temperature.")
            return (gamma - 1)/gamma * rho * T

        elif state.is_set(rho_Ei):
            self.logger.debug("Returning pressure from inner energy.")
            return (gamma - 1) * rho_Ei

        elif state.is_set(rho, c):
            self.logger.debug("Returning pressure from density and speed of sound.")
            return rho * c**2/gamma

    def temperature(self, state: CompressibleState) -> bla.SCALAR:
        r"""Returns the temperature from a given state

        .. math::
            T = \frac{\gamma}{\gamma - 1} \frac{p}{\rho}
            T = \gamma E_i
            T = \frac{c^2}{\gamma - 1}
        """

        gamma = self.heat_capacity_ratio

        rho = state.density
        p = state.pressure
        Ei = state.specific_inner_energy
        c = state.speed_of_sound

        if state.is_set(p, rho):
            self.logger.debug("Returning temperature from density and pressure.")
            return gamma/(gamma - 1) * p/rho

        elif state.is_set(Ei):
            self.logger.debug("Returning temperature from specific inner energy.")
            return gamma * Ei

        elif state.is_set(c):
            self.logger.debug("Returning temperature from speed of sound.")
            return c**2/(gamma - 1)

    def inner_energy(self, state: CompressibleState) -> bla.SCALAR:
        r"""Returns the inner energy from a given state

        .. math::
            \rho E_i = \frac{p}{\gamma - 1}
            \rho E_i = \rho \frac{T}{\gamma}
        """

        gamma = self.heat_capacity_ratio

        p = state.pressure
        rho = state.density
        T = state.temperature

        if state.is_set(p):
            self.logger.debug("Returning inner energy from pressure.")
            return p/(gamma - 1)

        elif state.is_set(rho, T):
            self.logger.debug("Returning inner energy from density and temperature.")
            return rho * T/gamma

    def specific_inner_energy(self, state: CompressibleState) -> bla.SCALAR:
        r"""Returns the specific inner energy from a given state

        .. math::
            E_i = \frac{T}{\gamma}
            E_i = \frac{p}{\rho (\gamma - 1)}
        """

        gamma = self.heat_capacity_ratio

        T = state.temperature
        rho = state.density
        p = state.pressure

        if state.is_set(T):
            self.logger.debug("Returning specific inner energy from temperature.")
            return T/gamma

        elif state.is_set(rho, p):
            self.logger.debug("Returning specific inner energy from density and pressure.")
            return p/(gamma - 1)/rho

    def speed_of_sound(self, state: CompressibleState) -> bla.SCALAR:
        r"""Returns the speed of sound from a given state

        .. math::
            c = \sqrt(\gamma \frac{p}{\rho})
            c = \sqrt((\gamma - 1) T)
        """

        gamma = self.heat_capacity_ratio

        rho = state.density
        p = state.pressure
        T = state.temperature
        Ei = state.specific_inner_energy

        if state.is_set(rho, p):
            self.logger.debug("Returning speed of sound from pressure and density.")
            return ngs.sqrt(gamma * p/rho)

        elif state.is_set(T):
            self.logger.debug("Returning speed of sound from temperature.")
            return ngs.sqrt((gamma - 1) * T)

        elif state.is_set(Ei):
            self.logger.debug("Returning speed of sound from specific inner energy.")
            return ngs.sqrt((gamma - 1) * Ei/gamma)

    def density_gradient(self, state: CompressibleState) -> bla.VECTOR:
        r"""Returns the density gradient from a given state

        .. math::
            \nabla \rho = \frac{\gamma}{\gamma - 1} \left[ \frac{\nabla p}{T} - p \frac{\nabla T}{T^2} \right]
            \nabla \rho = \gamma \left[ \frac{ \nabla (\rho E_i)}{T} - \rho E_i \frac{\nabla T}{T^2} \right]
        """

        gamma = self.heat_capacity_ratio

        p = state.pressure
        T = state.temperature
        rho_Ei = state.inner_energy

        grad_p = state.pressure_gradient
        grad_T = state.temperature_gradient
        grad_rho_Ei = state.inner_energy_gradient

        if state.is_set(p, T, grad_p, grad_T):
            self.logger.debug("Returning density gradient from pressure and temperature.")
            return gamma/(gamma - 1) * (grad_p/T - p * grad_T/T**2)

        elif state.is_set(T, rho_Ei, grad_T, grad_rho_Ei):
            self.logger.debug("Returning density gradient from temperature and inner energy.")
            return gamma * (grad_rho_Ei/T - rho_Ei * grad_T/T**2)

    def pressure_gradient(self, state: CompressibleState) -> bla.VECTOR:
        r"""Returns the pressure gradient from a given state

        .. math::
            \nabla p = \frac{\gamma - 1}{\gamma} \left[ (\nabla \rho) T + (\nabla T) \rho \right]
            \nabla p = (\gamma - 1) \nabla \rho E_i
        """

        gamma = self.heat_capacity_ratio

        rho = state.density
        T = state.temperature

        grad_rho = state.density_gradient
        grad_T = state.temperature_gradient
        grad_rho_Ei = state.inner_energy_gradient

        if state.is_set(rho, T, grad_rho, grad_T):
            self.logger.debug("Returning pressure gradient from density and temperature.")
            return (gamma - 1)/gamma * (grad_rho * T + rho * grad_T)

        elif state.is_set(grad_rho_Ei):
            self.logger.debug("Returning pressure gradient from inner energy gradient.")
            return (gamma - 1) * grad_rho_Ei

    def temperature_gradient(self, state: CompressibleState) -> bla.VECTOR:
        r"""Returns the temperature gradient from a given state

        .. math::
            \nabla T = \frac{\gamma}{\gamma - 1} \left[ \frac{\nabla p}{\rho} - p \frac{\nabla \rho}{\rho^2} \right]
            \nabla T = \gamma \nabla E_i
        """

        gamma = self.heat_capacity_ratio

        rho = state.density
        p = state.pressure

        grad_rho = state.density_gradient
        grad_p = state.pressure_gradient
        grad_Ei = state.specific_inner_energy_gradient

        if state.is_set(rho, p, grad_p, grad_rho):
            self.logger.debug("Returning temperature gradient from density and pressure.")
            return gamma/(gamma - 1) * (grad_p/rho - p * grad_rho/rho**2)

        elif state.is_set(grad_Ei):
            self.logger.debug("Returning temperature gradient from specific inner energy gradient.")
            return gamma * grad_Ei

    def characteristic_velocities(self,
                                  state: CompressibleState,
                                  unit_vector: bla.VECTOR,
                                  type_: str = None) -> bla.VECTOR:

        u = state.velocity
        c = state.speed_of_sound
        unit_vector = bla.as_vector(unit_vector)

        if state.is_set(u, c):

            un = bla.inner(u, unit_vector)

            lam_m_c = un - c
            lam = un
            lam_p_c = un + c

            if type_ is None:
                pass

            elif type_ == "absolute":
                lam_m_c = bla.abs(lam_m_c)
                lam = bla.abs(lam)
                lam_p_c = bla.abs(lam_p_c)

            elif type_ == "incoming":
                lam_m_c = bla.min(lam_m_c, 0)
                lam = bla.min(lam, 0)
                lam_p_c = bla.min(lam_p_c, 0)

            elif type_ == "outgoing":
                lam_m_c = bla.max(lam_m_c, 0)
                lam = bla.max(lam, 0)
                lam_p_c = bla.max(lam_p_c, 0)

            else:
                raise ValueError(
                    f"{str(type).capitalize()} invalid! Alternatives: {[None, 'absolute', 'incoming', 'outgoing']}")

            return bla.as_vector([lam_m_c] + u.dim * [lam] + [lam_p_c])

    def characteristic_variables(self,
                                 state: CompressibleState,
                                 unit_vector: bla.VECTOR) -> bla.VECTOR:

        unit_vector = bla.as_vector(unit_vector)

        rho = state.density
        c = state.speed_of_sound

        grad_rho = state.density_gradient
        grad_p = state.pressure_gradient
        grad_u = state.velocity_gradient

        if state.is_set(rho, c, grad_rho, grad_p, grad_u):

            grad_rho_n = bla.inner(grad_rho, unit_vector)
            grad_p_n = bla.inner(grad_p, unit_vector)
            grad_u_n = grad_u * unit_vector

            if unit_vector.dim == 2:

                char = (
                    grad_p_n - bla.inner(grad_u_n, unit_vector) * c * rho,
                    grad_rho_n * c**2 - grad_p_n,
                    grad_u_n[0] * unit_vector[1] - grad_u_n[1] * unit_vector[0],
                    grad_p_n + bla.inner(grad_u_n, unit_vector) * c * rho
                )

            else:
                raise NotImplementedError("Characteristic Variables not implemented for 3d!")

            return bla.as_vector(char)

    def characteristic_amplitudes(self,
                                  state: CompressibleState,
                                  unit_vector: bla.VECTOR,
                                  type_: str = None) -> bla.VECTOR:
        """ The charachteristic amplitudes are defined as

            .. math::
                \mathcal{L} = \Lambda * L_inverse * dV/dn,

        where Lambda is the eigenvalue matrix, L_inverse is the mapping from
        primitive variables to charachteristic variables and dV/dn is the
        derivative normal to the boundary.
        """
        velocities = self.characteristic_velocities(state, unit_vector, type_)
        variables = self.characteristic_variables(state, unit_vector)

        if state.is_set(velocities, variables):
            return bla.as_vector([vel * var for vel, var in zip(velocities, variables)])

    def primitive_from_conservative(self, state: CompressibleState) -> bla.MATRIX:
        """
        The M inverse matrix transforms conservative variables to primitive variables

        Equation E16.2.11, page 149

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        gamma = self.heat_capacity_ratio

        rho = state.density
        u = state.velocity

        if state.is_set(rho, u):

            if u.dim == 2:

                ux, uy = u

                Minv = (1, 0, 0, 0,
                        -ux/rho, 1/rho, 0, 0,
                        -uy/rho, 0, 1/rho, 0,
                        (gamma - 1)/2 * bla.inner(u, u), -(gamma - 1) * ux, -(gamma - 1) * uy, gamma - 1)

            else:
                raise NotImplementedError()

            dim = u.dim + 2

            return bla.as_matrix(Minv, dims=(dim, dim))

    def primitive_from_characteristic(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        """
        The L matrix transforms characteristic variables to primitive variables

        Equation E16.5.2, page 183

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        unit_vector = bla.as_vector(unit_vector)

        rho = state.density
        c = state.speed_of_sound

        if state.is_set(rho, c):
            if unit_vector.dim == 2:
                d0, d1 = unit_vector

                L = (0.5/c**2, 1/c**2, 0, 0.5/c**2,
                     -d0/(2*c*rho), 0, d1, d0/(2*c*rho),
                     -d1/(2*c*rho), 0, -d0, d1/(2*c*rho),
                     0.5, 0, 0, 0.5)
            else:
                return NotImplementedError()

            dim = unit_vector.dim + 2

            return bla.as_matrix(L, dims=(dim, dim))

    def primitive_convective_jacobian_x(self, state: CompressibleState) -> bla.MATRIX:

        u = state.velocity
        rho = state.density
        c = state.speed_of_sound

        if state.is_set(u, rho, c):
            if u.dim == 2:
                ux, _ = u

                A = (ux, rho, 0, 0,
                     0, ux, 0, 1/rho,
                     0, 0, ux, 0,
                     0, rho*c**2, 0, ux)

            else:
                raise NotImplementedError()

            dim = u.dim + 2

            return bla.as_matrix(A, dims=(dim, dim))

    def primitive_convective_jacobian_y(self, state: CompressibleState) -> bla.MATRIX:

        u = state.velocity
        rho = state.density
        c = state.speed_of_sound

        if state.is_set(u, rho, c):
            if u.dim == 2:
                _, uy = u

                B = (uy, 0, rho, 0,
                     0, uy, 0, 0,
                     0, 0, uy, 1/rho,
                     0, 0, rho*c**2, uy)

            else:
                raise NotImplementedError()

            dim = u.dim + 2

            return bla.as_matrix(B, dims=(dim, dim))

    def conservative_from_primitive(self, state: CompressibleState) -> bla.MATRIX:
        """
        The M matrix transforms primitive variables to conservative variables

        Equation E16.2.10, page 149

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        gamma = self.heat_capacity_ratio

        rho = state.density
        u = state.velocity

        if state.is_set(rho, u):

            if u.dim == 2:

                ux, uy = u

                M = (1, 0, 0, 0,
                     ux, rho, 0, 0,
                     uy, 0, rho, 0,
                     0.5*bla.inner(u, u), rho*ux, rho*uy, 1/(gamma - 1))
            else:
                raise NotImplementedError()

            dim = u.dim + 2

            return bla.as_matrix(M, dims=(dim, dim))

    def conservative_from_characteristic(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        """
        The P matrix transforms characteristic variables to conservative variables

        Equation E16.5.3, page 183

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        unit_vector = bla.as_vector(unit_vector)

        gamma = self.heat_capacity_ratio
        rho = state.density
        c = state.speed_of_sound
        u = state.velocity

        if state.is_set(rho, c, u):
            if unit_vector.dim == 2:
                d0, d1 = unit_vector
                ux, uy = u

                P = (
                    (1 / (2 * c ** 2),
                     1 / c ** 2, 0, 1 / (2 * c ** 2),
                     -d0 / (2 * c) + ux / (2 * c ** 2),
                     ux / c ** 2, d1 * rho, d0 / (2 * c) + ux / (2 * c ** 2),
                     -d1 / (2 * c) + uy / (2 * c ** 2),
                     uy / c ** 2, -d0 * rho, d1 / (2 * c) + uy / (2 * c ** 2),
                     0.5 / (gamma - 1) - d0 * ux / (2 * c) - d1 * uy / (2 * c) + bla.inner(u, u) / (4 * c ** 2),
                     bla.inner(u, u) / (2 * c ** 2),
                     -d0 * rho * uy + d1 * rho * ux, 0.5 / (gamma - 1) + d0 * ux / (2 * c) + d1 * uy / (2 * c) +
                     bla.inner(u, u) / (4 * c ** 2)))

            else:
                raise NotImplementedError()

            dim = unit_vector.dim + 2

            return bla.as_matrix(P, dims=(dim, dim))

    def conservative_convective_jacobian_x(self, state: CompressibleState) -> bla.MATRIX:
        r""" First Jacobian of the convective fluxes 

        .. math::
            A = \partial f_c / \partial U

        Equation E16.5.3, page 144

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4

        """

        gamma = self.heat_capacity_ratio

        u = state.velocity
        E = state.specific_energy

        if state.is_set(u, E):
            if u.dim == 2:
                ux, uy = u

                A = (0, 1, 0, 0,
                     (gamma - 3)/2 * ux**2 + (gamma - 1)/2 * uy**2, (3 - gamma) * ux, -(gamma - 1) * uy, gamma - 1,
                     -ux*uy, uy, ux, 0,
                     -gamma*ux*E + (gamma - 1)*ux*bla.inner(u, u), gamma*E - (gamma - 1)/2 * (uy**2 + 3*ux**2), -(gamma - 1)*ux*uy, gamma*ux)

            else:
                raise NotImplementedError()

            dim = u.dim + 2

            return bla.as_matrix(A, dims=(dim, dim))

    def conservative_convective_jacobian_y(self, state: CompressibleState) -> bla.MATRIX:
        r""" Second Jacobian of the convective fluxes 

        .. math::
            B = \partial g_c / \partial U

        Equation E16.5.3, page 144

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """

        gamma = self.heat_capacity_ratio

        u = state.velocity
        E = state.specific_energy

        if state.is_set(u, E):
            if u.dim == 2:
                ux, uy = u

                B = (
                    0, 0, 1, 0, -ux * uy, uy, ux, 0, (gamma - 3) / 2 * uy ** 2 + (gamma - 1) / 2 * ux ** 2, -(gamma - 1) * ux,
                    (3 - gamma) * uy, gamma - 1, -gamma * uy * E + (gamma - 1) * uy * bla.inner(u, u), -(gamma - 1) * ux * uy, gamma * E -
                    (gamma - 1) / 2 * (ux ** 2 + 3 * uy ** 2),
                    gamma * uy)

            else:
                raise NotImplementedError()

            dim = u.dim + 2

            return bla.as_matrix(B, dims=(dim, dim))

    def characteristic_from_primitive(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        """
        The L inverse matrix transforms primitive variables to charactersitic variables

        Equation E16.5.1, page 182

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        unit_vector = bla.as_vector(unit_vector)

        rho = state.density
        c = state.speed_of_sound

        if state.is_set(rho, c):
            if unit_vector.dim == 2:
                d0, d1 = unit_vector

                Linv = (0, -rho*c*d0, -rho*c*d1, 1,
                        c**2, 0, 0, -1,
                        0, d1, -d0, 0,
                        0, rho*c*d0, rho*c*d1, 1)

            else:
                return NotImplementedError()

            dim = unit_vector.dim + 2
            return bla.as_matrix(Linv, dims=(dim, dim))

    def characteristic_from_conservative(self, state: CompressibleState, unit_vector: bla.VECTOR) -> bla.MATRIX:
        """
        The P inverse matrix transforms conservative variables to characteristic variables

        Equation E16.5.4, page 183

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        unit_vector = bla.as_vector(unit_vector)

        gamma = self.heat_capacity_ratio
        rho = state.density
        c = state.speed_of_sound
        u = state.velocity

        if state.is_set(rho, c, u):
            if unit_vector.dim == 2:
                d0, d1 = unit_vector
                ux, uy = u

                Pinv = (
                    c*d0*ux + c*d1*uy + (gamma - 1)*bla.inner(u, u)/2, -c*d0 + ux*(1 - gamma), -c*d1 + uy*(1 - gamma), gamma - 1,
                    c**2 - (gamma - 1)*bla.inner(u, u)/2, -ux*(1 - gamma), -uy*(1 - gamma), 1 - gamma,
                    d0*uy/rho - d1*ux/rho, d1/rho, -d0/rho, 0,
                    -c*d0*ux - c*d1*uy + (gamma - 1)*bla.inner(u, u)/2, c*d0 + ux*(1 - gamma), c*d1 + uy*(1 - gamma), gamma - 1)
            else:
                raise NotImplementedError()

            dim = unit_vector.dim + 2

            return bla.as_matrix(Pinv, dims=(dim, dim))

    def format(self):
        formatter = self.formatter.new()
        formatter.output += super().format()
        formatter.entry('Heat Capacity Ratio', self.heat_capacity_ratio)
        return formatter.output


class DynamicViscosity(form.DynamicEquations):

    @property
    def is_inviscid(self):
        return isinstance(self, Inviscid)

    def viscosity(self, state: CompressibleState, *args):
        raise NotImplementedError()

    def format(self):
        formatter = self.formatter.new()
        formatter.entry('Dynamic Viscosity', str(self))
        return formatter.output


class Inviscid(DynamicViscosity, labels=['inviscid']):

    def viscosity(self, state: CompressibleState, *args):
        raise TypeError("Inviscid Setting! Dynamic Viscosity not defined!")


class Constant(DynamicViscosity, labels=['constant']):

    def viscosity(self, state: CompressibleState, *args):
        return 1


class Sutherland(DynamicViscosity, labels=['sutherland']):

    def __init__(self,
                 measurement_temperature: float = 110.4,
                 measurement_viscosity: float = 1.716e-5) -> None:

        self._measurement_temperature = measurement_temperature
        self._measurement_viscosity = measurement_viscosity

    @property
    def measurement_temperature(self):
        return self._measurement_temperature

    @measurement_temperature.setter
    def measurement_temperature(self, value):
        self._measurement_temperature = value

    @property
    def measurement_viscosity(self):
        return self._measurement_viscosity

    @measurement_viscosity.setter
    def measurement_viscosity(self, value):
        self._measurement_viscosity = value

    def viscosity(self, state: CompressibleState, equations: CompressibleEquations):

        T = state.temperature

        if state.is_set(T):

            REF = equations.reference_state()
            INF = equations.farfield_state()

            Tinf = INF.temperature
            T0 = self.measurement_temperature/REF.temperature

            return (T/Tinf)**(3/2) * (Tinf + T0)/(T + T0)

    def format(self):
        formatter = self.formatter.new()
        formatter.output += super().format()
        formatter.entry('Law Reference Temperature', self.measurement_temperature)
        formatter.entry('Law Reference Viscosity', self.measurement_viscosity)

        return formatter.output


class Scaling(form.DynamicEquations):

    @property
    def INF(self) -> ScalingState:
        return self._INF

    def __init__(self, farfield_state: ScalingState = None) -> None:

        if farfield_state is None:
            farfield_state = ScalingState()
            farfield_state.length = 1
            farfield_state.density = 1.293
            farfield_state.velocity = 102.9
            farfield_state.speed_of_sound = 343
            farfield_state.temperature = 293.15
            farfield_state.pressure = 101325

        self._INF = farfield_state

    def density(self) -> float:
        return 1.0

    def velocity_magnitude(self, Mach_number: float):
        raise NotImplementedError()

    def speed_of_sound(self, Mach_number: float):
        raise NotImplementedError()

    def velocity(self, direction: tuple[float, ...], Mach_number: float):
        mag = self.velocity_magnitude(Mach_number)
        return mag * bla.unit_vector(direction)

    def format(self):
        formatter = self.formatter.new()
        formatter.entry('Scaling', str(self))
        return formatter.output


class Aerodynamic(Scaling, labels=["aerodynamic"]):

    def _check_Mach_number(self, Mach_number: float):
        Ma = Mach_number
        if isinstance(Ma, ngs.Parameter):
            Ma = Ma.Get()

        if Ma <= 0.0:
            raise ValueError("Aerodynamic scaling requires Mach number > 0")

    def velocity_magnitude(self, Mach_number: float):
        return 1.0

    def speed_of_sound(self, Mach_number: float):
        self._check_Mach_number(Mach_number)
        return 1/Mach_number


class Acoustic(Scaling,  labels=["acoustic"]):

    def velocity_magnitude(self, Mach_number: float):
        return Mach_number

    def speed_of_sound(self, Mach_number: float):
        return 1.0


class Aeroacoustic(Scaling, labels=["aeroacoustic"]):

    def velocity_magnitude(self, Mach_number: float):
        Ma = Mach_number
        return Ma/(1 + Ma)

    def speed_of_sound(self, Mach_number: float):
        Ma = Mach_number
        return 1/(1 + Ma)


class RiemannSolver(form.DynamicEquations):

    def __init__(self, cfg: CompressibleFlowConfig = None) -> None:
        if cfg is None:
            cfg = CompressibleFlowConfig()

        self.cfg = cfg

    @property
    def eq(self) -> CompressibleEquations:
        return self.cfg.equations

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR):
        NotImplementedError()


class LaxFriedrich(RiemannSolver, labels=["lf", "lax_friedrich"]):

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR):
        unit_vector = bla.as_vector(unit_vector)

        u = self.eq.velocity(state)
        c = self.eq.speed_of_sound(state)

        lambda_max = bla.abs(bla.inner(u, unit_vector)) + c
        return lambda_max * ngs.Id(unit_vector.dim + 2)


class Roe(RiemannSolver, labels=["roe"]):

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR):
        unit_vector = bla.as_vector(unit_vector)

        lambdas = self.eq.characteristic_velocities(state, unit_vector, type_="absolute")
        return self.eq.transform_characteristic_to_conservative(bla.diagonal(lambdas), state, unit_vector)


class HLL(RiemannSolver, labels=["hll"]):

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR):
        unit_vector = bla.as_vector(unit_vector)

        u = self.eq.velocity(state)
        c = self.eq.speed_of_sound(state)

        un = bla.inner(u, unit_vector)
        s_plus = bla.max(un + c)

        return s_plus * ngs.Id(unit_vector.dim + 2)


class HLLEM(RiemannSolver, labels=["hllem"]):

    def __init__(self, theta_0: float = 1e-8, cfg: CompressibleFlowConfig = None) -> None:
        self.theta_0 = theta_0
        super().__init__(cfg)

    @property
    def theta_0(self):
        """ Defines a threshold value used to stabilize contact waves, when the eigenvalue tends to zero.

        This can occur if the flow is parallel to the element or domain boundary!
        """
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value):
        self._theta_0 = float(value)

    def convective_stabilisation_matrix(self, state: CompressibleState, unit_vector: bla.VECTOR):
        unit_vector = bla.as_vector(unit_vector)

        u = self.eq.velocity(state)
        c = self.eq.speed_of_sound(state)

        un = bla.inner(u, unit_vector)
        un_abs = bla.abs(un)
        s_plus = bla.max(un + c)

        theta = bla.max(un_abs/(un_abs + c), self.theta_0)
        THETA = bla.diagonal([1] + unit_vector.dim * [theta] + [1])
        THETA = self.eq.transform_characteristic_to_conservative(THETA, state, unit_vector)

        return s_plus * THETA


# ------- Equations ------- #


class CompressibleEquations(form.FlowEquations):

    def __init__(self, cfg: CompressibleFlowConfig = None) -> None:

        if cfg is None:
            cfg = CompressibleFlowConfig()

        self.cfg = cfg

    def farfield_state(self, direction: tuple[float, ...] = None) -> CompressibleState:

        Ma = self.cfg.Mach_number
        INF = CompressibleState()

        INF.density = self.cfg.scaling.density()
        INF.speed_of_sound = self.cfg.scaling.speed_of_sound(Ma)
        INF.temperature = self.temperature(INF)
        INF.pressure = self.pressure(INF)

        if direction is not None:
            direction = bla.as_vector(direction)

            if not 1 <= direction.dim <= 3:
                raise ValueError(f"Direction")

            INF.velocity = self.cfg.scaling.velocity(direction, Ma)
            INF.inner_energy = self.inner_energy(INF)
            INF.kinetic_energy = self.kinetic_energy(INF)
            INF.energy = self.energy(INF)

        return INF

    def reference_state(self, direction: tuple[float, ...] = None) -> CompressibleState:
        INF = self.farfield_state(direction)
        INF_ = self.cfg.scaling.INF
        REF = ScalingState()

        for key, value in INF_.items():
            value_ = getattr(INF, key, None)

            if value_ is not None:

                if bla.is_vector(value_):
                    value_ = bla.inner(value_, value_)

                setattr(REF, key, value/value_)

        return REF

    def reference_Reynolds_number(self, Reynolds_number):
        return Reynolds_number/self.cfg.scaling.velocity_magnitude(self.cfg.Mach_number)

    def local_Mach_number(self, state: CompressibleState):
        u = self.velocity(state)
        c = self.speed_of_sound(state)

        return ngs.sqrt(bla.inner(u, u))/c

    def local_Reynolds_number(self, state: CompressibleState):
        rho = self.density(state)
        u = self.velocity(state)
        mu = self.viscosity(state)
        return rho * u / mu

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

        Re = self.reference_Reynolds_number(self.cfg.Reynolds_number)
        Pr = self.cfg.Prandtl_number
        mu = self.viscosity(state)

        continuity = tuple(0 for _ in range(u.dim))

        flux = (continuity, mu/Re * tau, mu/Re * (tau*u - q/Pr))

        return bla.as_matrix(flux, dims=(u.dim + 2, u.dim))

    def primitive_convective_jacobian(self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None):
        lambdas = self.characteristic_velocities(state, unit_vector, type_)
        LAMBDA = bla.diagonal(lambdas)
        return self.transform_characteristic_to_primitive(LAMBDA, state, unit_vector)

    def conservative_convective_jacobian(self, state: CompressibleState, unit_vector: bla.VECTOR, type_: str = None):
        lambdas = self.characteristic_velocities(state, unit_vector, type_)
        LAMBDA = bla.diagonal(lambdas)
        return self.transform_characteristic_to_conservative(LAMBDA, state, unit_vector)

    def characteristic_identity(
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
    def density(self, state: CompressibleState) -> bla.SCALAR:
        return self.cfg.equation_of_state.density(state)

    @form.equation
    def velocity(self, state: CompressibleState) -> bla.VECTOR:
        rho = state.density
        rho_u = state.momentum

        if state.is_set(rho, rho_u):
            self.logger.debug("Returning velocity from density and momentum.")

            if bla.is_zero(rho_u) and bla.is_zero(rho):
                return bla.as_vector((0.0 for _ in range(rho_u.dim)))

            return rho_u/rho

    @form.equation
    def momentum(self, state: CompressibleState) -> bla.VECTOR:
        rho = state.density
        u = state.velocity

        if state.is_set(rho, u):
            self.logger.debug("Returning momentum from density and velocity.")
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
                self.logger.debug("Returning bla.inner energy from energy and kinetic energy.")
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
                self.logger.debug("Returning specific bla.inner energy from bla.inner energy and density.")
                return rho_Ei/rho

            elif state.is_set(E, Ek):
                self.logger.debug(
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
            self.logger.debug("Returning kinetic energy from density and velocity.")
            return 0.5 * rho * bla.inner(u, u)

        elif state.is_set(rho, rho_u):
            self.logger.debug("Returning kinetic energy from density and momentum.")
            return 0.5 * bla.inner(rho_u, rho_u)/rho

        elif state.is_set(rho_E, rho_Ei):
            self.logger.debug("Returning kinetic energy from energy and bla.inner energy.")
            return rho_E - rho_Ei

        elif state.is_set(rho, Ek):
            self.logger.debug("Returning kinetic energy from density and specific kinetic energy.")
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
            self.logger.debug("Returning specific kinetic energy from velocity.")
            return 0.5 * bla.inner(u, u)

        elif state.is_set(rho, rho_u):
            self.logger.debug("Returning specific kinetic energy from density and momentum.")
            return 0.5 * bla.inner(rho_u, rho_u)/rho**2

        elif state.is_set(rho, rho_Ek):
            self.logger.debug("Returning specific kinetic energy from density and kinetic energy.")
            return rho_Ek/rho

        elif state.is_set(E, Ei):
            self.logger.debug("Returning specific kinetic energy from specific energy and speicific bla.inner energy.")
            return E - Ei

    @form.equation
    def energy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        E = state.specific_energy

        rho_Ei = state.inner_energy
        rho_Ek = state.kinetic_energy

        if state.is_set(rho, E):
            self.logger.debug("Returning energy from density and specific energy.")
            return rho * E

        elif state.is_set(rho_Ei, rho_Ek):
            self.logger.debug("Returning energy from bla.inner energy and kinetic energy.")
            return rho_Ei + rho_Ek

    @form.equation
    def specific_energy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        rho_E = state.energy

        Ei = state.specific_inner_energy
        Ek = state.specific_kinetic_energy

        if state.is_set(rho, rho_E):
            self.logger.debug("Returning specific energy from density and energy.")
            return rho_E/rho

        elif state.is_set(Ei, Ek):
            self.logger.debug("Returning specific energy from specific bla.inner energy and specific kinetic energy.")
            return Ei + Ek

    @form.equation
    def enthalpy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        H = state.specific_enthalpy

        rho_E = state.energy
        p = state.pressure

        if state.is_set(rho_E, p):
            self.logger.debug("Returning enthalpy from energy and pressure.")
            return rho_E + p

        elif state.is_set(rho, H):
            self.logger.debug("Returning enthalpy from density and specific enthalpy.")
            return rho * H

    @form.equation
    def specific_enthalpy(self, state: CompressibleState) -> bla.SCALAR:
        rho = state.density
        rho_H = state.enthalpy

        rho_E = state.energy
        E = state.specific_energy
        p = state.pressure

        if state.is_set(rho, rho_H):
            self.logger.debug("Returning specific enthalpy from density and enthalpy.")
            return rho_H/rho

        elif state.is_set(rho, rho_E, p):
            self.logger.debug("Returning specific enthalpy from specific energy, density and pressure.")
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
            self.logger.debug("Returning velocity gradient from density and momentum.")
            return grad_rho_u/rho - bla.outer(rho_u, grad_rho)/rho**2

    @form.equation
    def momentum_gradient(self, state: CompressibleState) -> bla.MATRIX:
        rho = state.density
        u = state.velocity

        grad_rho = state.density_gradient
        grad_u = state.velocity_gradient

        if state.is_set(rho, u, grad_rho, grad_u):
            self.logger.debug("Returning momentum gradient from density and momentum.")
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
            self.logger.debug("Returning energy gradient from bla.inner energy and kinetic energy.")
            return grad_rho_Ei + grad_rho_Ek

    @form.equation
    def specific_energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_Ei = state.specific_energy_gradient
        grad_Ek = state.specific_kinetic_energy_gradient

        if state.is_set(grad_Ei, grad_Ek):
            self.logger.debug(
                "Returning specific energy gradient from specific bla.inner energy and specific kinetic energy.")
            return grad_Ei + grad_Ek

    @form.equation
    def inner_energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_rho_E = state.energy_gradient
        grad_rho_Ek = state.kinetic_energy_gradient

        if state.is_set(grad_rho_E, grad_rho_Ek):
            self.logger.debug("Returning bla.inner energy gradient from energy and kinetic energy.")
            return grad_rho_E - grad_rho_Ek

    @form.equation
    def specific_inner_energy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_E = state.specific_energy_gradient
        grad_Ek = state.specific_kinetic_energy_gradient

        if state.is_set(grad_E, grad_Ek):
            self.logger.debug(
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
            self.logger.debug("Returning kinetic energy gradient from energy and bla.inner energy.")
            return grad_rho_E - grad_rho_Ei

        elif state.is_set(rho, u, grad_rho, grad_u):
            self.logger.debug("Returning kinetic energy gradient from density and velocity.")
            return rho * (grad_u.trans * u) + 0.5 * grad_rho * bla.inner(u, u)

        elif state.is_set(rho, rho_u, grad_rho, grad_rho_u):
            self.logger.debug("Returning kinetic energy gradient from density and momentum.")
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
            self.logger.debug(
                "Returning specific kinetic energy gradient from specific energy and specific bla.inner energy.")
            return grad_E - grad_Ei

        elif state.is_set(u, grad_u):
            self.logger.debug("Returning specific kinetic energy gradient from velocity.")
            return grad_u.trans * u

        elif state.is_set(rho, rho_u, grad_rho, grad_rho_u):
            self.logger.debug("Returning specific kinetic energy gradient from density and momentum.")
            return (grad_rho_u.trans * rho_u)/rho**2 - grad_rho * bla.inner(rho_u, rho_u)/rho**3

    @form.equation
    def enthalpy_gradient(self, state: CompressibleState) -> bla.VECTOR:
        grad_rho_E = state.energy_gradient
        grad_p = state.pressure_gradient

        if state.is_set(grad_rho_E, grad_p):
            self.logger.debug("Returning enthalpy gradient from energy and pressure.")
            return grad_rho_E + grad_p

    @form.equation
    def strain_rate_tensor(self, state: CompressibleState) -> bla.MATRIX:
        grad_u = self.velocity_gradient(state)

        if state.is_set(grad_u):
            self.logger.debug("Returning strain rate tensor from velocity.")
            return 0.5 * (grad_u + grad_u.trans) - 1/3 * bla.trace(grad_u, id=True)

    @form.equation
    def deviatoric_stress_tensor(self, state: CompressibleState):

        mu = state.viscosity
        EPS = state.strain_rate_tensor

        if state.is_set(mu, EPS):
            self.logger.debug("Returning deviatoric stress tensor from strain rate tensor and viscosity.")
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


# ------- Boundary Conditions ------- #


class FarField(dmesh.Boundary):
    def __init__(self, state: CompressibleState, theta_0: float = 0):
        super().__init__(state)
        self.theta_0 = theta_0


class Outflow(dmesh.Boundary):

    def __init__(self, pressure: CompressibleState | float):
        if not isinstance(pressure, CompressibleState):
            pressure = CompressibleState(pressure=pressure)
        super().__init__(pressure)


class NSCBC(dmesh.Boundary):

    def __init__(self,
                 state: CompressibleState,
                 sigma: float = 0.25,
                 reference_length: float = 1,
                 tangential_convective_fluxes: bool = True) -> None:

        super().__init__(state)
        self.sigma = sigma
        self.reference_length = reference_length
        self.tang_conv_flux = tangential_convective_fluxes


class InviscidWall(dmesh.Boundary):
    ...


class Symmetry(dmesh.Boundary):
    ...


class IsothermalWall(dmesh.Boundary):

    def __init__(self, temperature: float | CompressibleState) -> None:
        if not isinstance(temperature, CompressibleState):
            temperature = CompressibleState(temperature=temperature)
        super().__init__(temperature)


class AdiabaticWall(dmesh.Boundary):
    ...


class CompressibleBC(form.FormulationBC):
    farfield = FarField
    outflow = Outflow
    nscbc = NSCBC
    inviscid_wall = InviscidWall
    symmetry = Symmetry
    isothermal_wall = IsothermalWall
    adiabatic_wall = AdiabaticWall
    periodic = dmesh.Periodic


# ------- Domain Conditions ------- #


class PML(dmesh.Domain):
    ...


class CompressibleDC(form.FormulationDC):
    initial = dmesh.Initial
    force = dmesh.Force
    perturbation = dmesh.Perturbation
    sponge_layer = dmesh.SpongeLayer
    psponge_layer = dmesh.PSpongeLayer
    grid_deformation = dmesh.GridDeformation
    pml = PML


# ------- Formulations ------- #

class CompressibleFormulation(form.Formulation):
    ...

# --- Conservative --- #


class Primal(form.Space):

    def get_state_from_variable(self, gfu: ngs.GridFunction = None) -> CompressibleState:
        if gfu is None:
            gfu = self.gfu

        state = CompressibleState()
        state.density = gfu[0]
        state.momentum = gfu[slice(1, self.dmesh.dim + 1)]
        state.energy = gfu[self.dmesh.dim + 1]

        return state

    def get_variable_from_state(self, state: CompressibleState) -> ngs.CF:
        eq = self.cfg.flow.equations

        density = eq.density(state)
        momentum = eq.momentum(state)
        energy = eq.energy(state)
        return ngs.CF((density, momentum, energy))

    def set_variables(self, state: CompressibleState):
        state.velocity = self.equations.velocity(state)
        state.inner_energy = self.equations.inner_energy(state)
        state.kinetic_energy = self.equations.kinetic_energy(state)
        state.specific_inner_energy = self.equations.specific_inner_energy(state)
        state.specific_kinetic_energy = self.equations.specific_kinetic_energy(state)
        state.pressure = self.equations.pressure(state)
        state.temperature = self.equations.temperature(state)
        state.speed_of_sound = self.equations.speed_of_sound(state)
        state.enthalpy = self.equations.enthalpy(state)
        state.specific_enthalpy = self.equations.specific_enthalpy(state)

        return state


class PrimalElement(Primal):

    def get_space(self) -> ngs.L2:
        dim = self.dmesh.dim + 2
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        V = ngs.L2(mesh, order=order)
        V = self.dmesh._reduce_psponge_layers_order_elementwise(V)

        return V**dim

    def set_configuration_flags(self):

        self.has_time_derivative = False
        if not self.cfg.simulation.is_stationary:
            self.has_time_derivative = True

    def get_transient_gridfunction(self) -> TransientGridfunction:
        if not self.cfg.simulation.is_stationary:
            return self.cfg.simulation.scheme.get_transient_gridfunction(self.gfu)


class PrimalFacet(Primal):

    @property
    def mask(self) -> ngs.GridFunction:
        mask = getattr(self, "_mask", None)

        if mask is None:

            dmesh = self.dmesh

            fes = ngs.FacetFESpace(dmesh.ngsmesh, order=0)
            mask = ngs.GridFunction(fes, name="mask")
            mask.vec[:] = 0
            mask.vec[~fes.GetDofs(self.dmesh.bcs.get_domain_boundaries(True))] = 1

        return mask

    def get_space(self) -> ngs.FacetFESpace:
        dim = self.dmesh.dim + 2
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        VHAT = ngs.FacetFESpace(mesh, order=order)
        VHAT = self.dmesh._reduce_psponge_layers_order_facetwise(VHAT)

        if self.dmesh.is_periodic:
            VHAT = ngs.Periodic(VHAT)

        return VHAT**dim

    def get_transient_gridfunction(self) -> TransientGridfunction:
        if not self.cfg.simulation.is_stationary and self.dmesh.bcs.get(NSCBC):
            return self.cfg.simulation.scheme.get_transient_gridfunction(self.gfu)

    def add_mass_bilinearform(self, blf: ngs.BilinearForm, dx=ngs.dx, **dx_kwargs):
        return super().add_mass_bilinearform(blf, dx=ngs.dx, element_boundary=True, **dx_kwargs)

    def add_mass_linearform(self, state: CompressibleState, lf: ngs.LinearForm, dx=ngs.dx, **dx_kwargs):
        return super().add_mass_linearform(state, lf, dx=ngs.dx, element_boundary=True, **dx_kwargs)


class StrainHeatSpace(form.Space):

    def get_space(self) -> ngs.FESpace:
        dim = 4 * self.dmesh.dim - 3
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.L2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return Q**dim


class GradientSpace(form.Space):

    def get_space(self) -> ngs.FESpace:
        dim = 4 * self.dmesh.dim - 3
        order = self.cfg.fem.order
        mesh = self.dmesh.ngsmesh

        Q = ngs.L2(mesh, order=order)
        Q = self.dmesh._reduce_psponge_layers_order_elementwise(Q)

        return Q**dim


class Conservative(CompressibleFormulation, label="conservative"):

    @property
    def U(self) -> PrimalElement:
        return self.spaces["U"]

    @property
    def Uhat(self) -> PrimalFacet:
        return self.spaces["Uhat"]

    @property
    def Q(self) -> StrainHeat | GradientSpace:
        return self.spaces["Q"]

    def get_space(self) -> form.Spaces:
        mixed_method = self.cfg.flow.mixed_method

        spaces = form.Spaces()
        spaces["U"] = PrimalElement(self.cfg, self.dmesh)
        spaces["Uhat"] = PrimalFacet(self.cfg, self.dmesh)
        spaces["Q"] = mixed_method.get_mixed_space(self.cfg, self.dmesh)

        return spaces

    def assemble_system(self, blf: ngs.BilinearForm, lf: ngs.ngs.LinearForm):

        if self.U.dt:
            self.add_time_derivative(blf, lf)

        self.add_convection(blf, lf)

    def add_time_derivative(self, blf: ngs.BilinearForm, lf: ngs.ngs.LinearForm):
        scheme = self.cfg.simulation.scheme
        U = self.U

        dt = U.dt.swap_level(U.trial)

        blf += bla.inner(scheme.scheme(dt), U.test) * ngs.dx

    def add_convection(self, blf: ngs.BilinearForm, lf: ngs.ngs.LinearForm):

        bonus_vol = self.cfg.fem.bonus_int_order.VOL
        bonus_bnd = self.cfg.fem.bonus_int_order.BND

        U = self.spaces.U
        Uhat = self.spaces.Uhat

        dx = ngs.dx(bonus_intorder=self.cfg.fem.bonus_int_order.VOL)
        dxe = ngs.dx(element_boundary=True, bonus_intorder=self.cfg.fem.bonus_int_order.BND)

        U_ = U.get_state_from_variable(U.trial)
        flux = self.cfg.flow.equations.convective_flux(U_)

        fes_mask = ngs.FacetFESpace(self.mesh, order=0)
        mask = ngs.GridFunction

        blf -= bla.inner(flux, ngs.grad(U.test)) * dx

        U, V = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        # Subtract boundary regions
        mask_fes = FacetFESpace(self.mesh, order=0)
        mask = GridFunction(mask_fes)
        mask.vec[:] = 0
        mask.vec[~mask_fes.GetDofs(self.dmesh.boundary(self.dmesh.bcs.pattern))] = 1

        var_form = -InnerProduct(self.convective_flux(U), grad(V)) * dx(bonus_intorder=bonus_vol)
        var_form += InnerProduct(self.convective_numerical_flux(U, Uhat, self.normal),
                                 V) * dx(element_boundary=True, bonus_intorder=bonus_bnd)
        var_form -= mask * InnerProduct(self.convective_numerical_flux(U, Uhat, self.normal),
                                        Vhat) * dx(element_boundary=True, bonus_intorder=bonus_bnd)

        blf += var_form.Compile(compile_flag)

    def convective_numerical_flux(self, U_: CompressibleState, Uhat_: CompressibleState):
        ...


# ------- Configuration ------- #


class CompressibleFlowConfig(form.FlowConfig, label="compressible"):

    formulation = CompressibleFormulation
    bcs = CompressibleBC
    dcs = CompressibleDC

    def __init__(self) -> None:
        super().__init__(CompressibleEquations(self))
        self._Mach_number = ngs.Parameter(0.3)
        self._Reynolds_number = ngs.Parameter(1)
        self._Prandtl_number = ngs.Parameter(0.72)
        self.equation_of_state = 'ideal'
        self.dynamic_viscosity = 'inviscid'
        self.scaling = "aerodynamic"
        self.riemann_solver = "lax_friedrich"
        self.mixed_method = None

    def INF(self, direction: tuple[float, ...]) -> CompressibleState:
        return self.equations.farfield_state(direction)

    @property
    def equations(self) -> CompressibleEquations:
        return self._equations

    @property
    def Mach_number(self) -> ngs.Parameter:
        return self._Mach_number

    @Mach_number.setter
    def Mach_number(self, Mach_number: float):
        if isinstance(Mach_number, ngs.Parameter):
            Mach_number = Mach_number.Get()

        if Mach_number < 0:
            raise ValueError("Invalid Mach number. Value has to be >= 0!")
        else:
            self._Mach_number.Set(Mach_number)

    @property
    def Reynolds_number(self) -> ngs.Parameter:
        """ Represents the ratio between inertial and viscous forces """
        if self.dynamic_viscosity.is_inviscid:
            raise Exception("Inviscid solver configuration: Reynolds number not applicable")
        return self._Reynolds_number

    @Reynolds_number.setter
    def Reynolds_number(self, Reynolds_number: float):
        if isinstance(Reynolds_number, ngs.Parameter):
            Reynolds_number = Reynolds_number.Get()

        if Reynolds_number <= 0:
            raise ValueError("Invalid Reynold number. Value has to be > 0!")
        else:
            self._Reynolds_number.Set(Reynolds_number)

    @property
    def Prandtl_number(self) -> ngs.Parameter:
        if self.dynamic_viscosity.is_inviscid:
            raise Exception("Inviscid solver configuration: Prandtl number not applicable")
        return self._Prandtl_number

    @Prandtl_number.setter
    def Prandtl_number(self, Prandtl_number: float):
        if isinstance(Prandtl_number, ngs.Parameter):
            Prandtl_number = Prandtl_number.Get()

        if Prandtl_number <= 0:
            raise ValueError("Invalid Prandtl_number. Value has to be > 0!")
        else:
            self._Prandtl_number.Set(Prandtl_number)

    @property
    def equation_of_state(self) -> IdealGas:
        return self._equation_of_state

    @equation_of_state.setter
    def equation_of_state(self, equation_of_state: str):
        self._equation_of_state = self._get_type(equation_of_state, EquationOfState)

    @property
    def dynamic_viscosity(self) -> Constant | Inviscid | Sutherland:
        return self._dynamic_viscosity

    @dynamic_viscosity.setter
    def dynamic_viscosity(self, dynamic_viscosity: str):
        self._dynamic_viscosity = self._get_type(dynamic_viscosity, DynamicViscosity)

    @property
    def scaling(self) -> Aerodynamic | Aeroacoustic | Acoustic:
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: str):
        self._scaling = self._get_type(scaling, Scaling)

    @property
    def riemann_solver(self) -> LaxFriedrich | Roe | HLL | HLLEM:
        return self._riemann_solver

    @riemann_solver.setter
    def riemann_solver(self, riemann_solver: str):
        self._riemann_solver = self._get_type(riemann_solver, RiemannSolver, cfg=self)

    @property
    def mixed_method(self) -> MixedMethod | StrainHeat | Gradient:
        return self._mixed_method

    @mixed_method.setter
    def mixed_method(self, mixed_method: str | None):
        if mixed_method is None:
            mixed_method = str(None)
        self._mixed_method = self._get_type(mixed_method, MixedMethod)

    def format(self):
        formatter = self.formatter.new()
        formatter.subheader('Compressible Flow Configuration').newline()
        formatter.entry("Mach Number", self.Mach_number)
        if not self.dynamic_viscosity.is_inviscid:
            formatter.entry("Reynolds Number", self.Reynolds_number)
            formatter.entry("Prandtl Number", self.Prandtl_number)
        formatter.add_config(self.equation_of_state)
        formatter.add_config(self.dynamic_viscosity)
        formatter.add_config(self.scaling)
        return formatter.output
