from __future__ import annotations

import logging
import ngsolve as ngs

from dream import bla
from dream.config import MultipleConfiguration, parameter_configuration
from dream.compressible.state import CompressibleState

logger = logging.getLogger(__name__)


class EquationOfState(MultipleConfiguration, is_interface=True):

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


class IdealGas(EquationOfState):

    aliases = ('ideal', 'perfect', )

    @parameter_configuration(default=1.4)
    def heat_capacity_ratio(self, heat_capacity_ratio: float):
        return heat_capacity_ratio

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
            logger.debug("Returning density from pressure and temperature.")
            return gamma/(gamma - 1) * p/T

        elif state.is_set(p, c):
            logger.debug("Returning density from pressure and speed of sound.")
            return gamma * p/c**2

        elif state.is_set(rho_Ei, T):
            logger.debug("Returning density from inner energy and temperature.")
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
            logger.debug("Returning pressure from density and temperature.")
            return (gamma - 1)/gamma * rho * T

        elif state.is_set(rho_Ei):
            logger.debug("Returning pressure from inner energy.")
            return (gamma - 1) * rho_Ei

        elif state.is_set(rho, c):
            logger.debug("Returning pressure from density and speed of sound.")
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
            logger.debug("Returning temperature from density and pressure.")
            return gamma/(gamma - 1) * p/rho

        elif state.is_set(Ei):
            logger.debug("Returning temperature from specific inner energy.")
            return gamma * Ei

        elif state.is_set(c):
            logger.debug("Returning temperature from speed of sound.")
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
            logger.debug("Returning inner energy from pressure.")
            return p/(gamma - 1)

        elif state.is_set(rho, T):
            logger.debug("Returning inner energy from density and temperature.")
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
            logger.debug("Returning specific inner energy from temperature.")
            return T/gamma

        elif state.is_set(rho, p):
            logger.debug("Returning specific inner energy from density and pressure.")
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
            logger.debug("Returning speed of sound from pressure and density.")
            return ngs.sqrt(gamma * p/rho)

        elif state.is_set(T):
            logger.debug("Returning speed of sound from temperature.")
            return ngs.sqrt((gamma - 1) * T)

        elif state.is_set(Ei):
            logger.debug("Returning speed of sound from specific inner energy.")
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
            logger.debug("Returning density gradient from pressure and temperature.")
            return gamma/(gamma - 1) * (grad_p/T - p * grad_T/T**2)

        elif state.is_set(T, rho_Ei, grad_T, grad_rho_Ei):
            logger.debug("Returning density gradient from temperature and inner energy.")
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
            logger.debug("Returning pressure gradient from density and temperature.")
            return (gamma - 1)/gamma * (grad_rho * T + rho * grad_T)

        elif state.is_set(grad_rho_Ei):
            logger.debug("Returning pressure gradient from inner energy gradient.")
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
            logger.debug("Returning temperature gradient from density and pressure.")
            return gamma/(gamma - 1) * (grad_p/rho - p * grad_rho/rho**2)

        elif state.is_set(grad_Ei):
            logger.debug("Returning temperature gradient from specific inner energy gradient.")
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

    heat_capacity_ratio: ngs.Parameter
