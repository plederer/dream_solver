from __future__ import annotations

import logging
import ngsolve as ngs

from dream import bla
from dream.config import InterfaceConfiguration, parameter
from dream.compressible.config import flowstate

logger = logging.getLogger(__name__)


class EquationOfState(InterfaceConfiguration, is_interface=True):

    def density(self, U: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def pressure(self, U: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def temperature(self, U: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def inner_energy(self, U: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def specific_inner_energy(self, U: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def speed_of_sound(self, U: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def density_gradient(self, U: flowstate, dU: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def pressure_gradient(self, U: flowstate, dU: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def temperature_gradient(self, U: flowstate, dU: flowstate) -> ngs.CF:
        raise NotImplementedError()

    def characteristic_velocities(
            self, U: flowstate, unit_vector: bla.VECTOR, type_: str = None) -> bla.VECTOR:
        raise NotImplementedError()

    def characteristic_variables(
            self, U: flowstate, dU: flowstate, unit_vector: bla.VECTOR) -> bla.VECTOR:
        raise NotImplementedError()

    def characteristic_amplitudes(
            self, U: flowstate, dU: flowstate, unit_vector: bla.VECTOR, type_: str = None) -> bla.VECTOR:
        raise NotImplementedError()

    def primitive_from_conservative(self, U: flowstate) -> bla.MATRIX:
        raise NotImplementedError()

    def primitive_from_characteristic(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        raise NotImplementedError()

    def primitive_convective_jacobian_x(self, U: flowstate) -> bla.MATRIX:
        raise NotImplementedError()

    def primitive_convective_jacobian_y(self, U: flowstate) -> bla.MATRIX:
        raise NotImplementedError()

    def conservative_from_primitive(self, U: flowstate) -> bla.MATRIX:
        raise NotImplementedError()

    def conservative_from_characteristic(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        raise NotImplementedError()

    def conservative_convective_jacobian_x(self, U: flowstate) -> bla.MATRIX:
        raise NotImplementedError()

    def conservative_convective_jacobian_y(self, U: flowstate) -> bla.MATRIX:
        raise NotImplementedError()

    def characteristic_from_primitive(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        raise NotImplementedError()

    def characteristic_from_conservative(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
        raise NotImplementedError()

    def isentropic_density(self, U: flowstate, Uref: flowstate) -> ngs.CF:
        raise NotImplementedError()


class IdealGas(EquationOfState):

    name = 'ideal'
    aliases = ('perfect', )

    @parameter(default=1.4)
    def heat_capacity_ratio(self, heat_capacity_ratio: float):
        return heat_capacity_ratio

    def density(self, U: flowstate) -> bla.SCALAR:
        r"""Returns the density from a given state

        .. math::
            \rho = \frac{\gamma}{\gamma - 1} \frac{p}{T}
            \rho = \gamma \frac{\rho E_i}{T}
            \rho = \gamma \frac{p}{c^2}
        """

        gamma = self.heat_capacity_ratio
        if U.rho is not None:
            return U.rho

        elif all((U.p, U.T)):
            logger.debug("Returning density from pressure and temperature.")
            return gamma/(gamma - 1) * U.p/U.T

        elif all((U.p, U.c)):
            logger.debug("Returning density from pressure and speed of sound.")
            return gamma * U.p/U.c**2

        elif all((U.rho_Ei, U.T)):
            logger.debug("Returning density from inner energy and temperature.")
            return gamma * U.rho_Ei/U.T

        return None

    def pressure(self, U: flowstate) -> bla.SCALAR:
        r"""Returns the density from a given state

        .. math::
            p = \frac{\gamma - 1}{\gamma} \rho T
            p = (\gamma - 1) \rho E_i
            p = \rho \frac{c^2}{\gamma}
        """

        gamma = self.heat_capacity_ratio
        if U.p is not None:
            return U.p

        elif all((U.rho, U.T)):
            logger.debug("Returning pressure from density and temperature.")
            return (gamma - 1)/gamma * U.rho * U.T

        elif U.rho_Ei is not None:
            logger.debug("Returning pressure from inner energy.")
            return (gamma - 1) * U.rho_Ei

        elif all((U.rho, U.c)):
            logger.debug("Returning pressure from density and speed of sound.")
            return U.rho * U.c**2/gamma

        return None

    def temperature(self, U: flowstate) -> bla.SCALAR:
        r"""Returns the temperature from a given state

        .. math::
            T = \frac{\gamma}{\gamma - 1} \frac{p}{\rho}
            T = \gamma E_i
            T = \frac{c^2}{\gamma - 1}
        """

        gamma = self.heat_capacity_ratio
        if U.T is not None:
            return U.T

        elif all((U.p, U.rho)):
            logger.debug("Returning temperature from density and pressure.")
            return gamma/(gamma - 1) * U.p/U.rho

        elif U.Ei is not None:
            logger.debug("Returning temperature from specific inner energy.")
            return gamma * U.Ei

        elif U.c is not None:
            logger.debug("Returning temperature from speed of sound.")
            return U.c**2/(gamma - 1)

        return None

    def inner_energy(self, U: flowstate) -> bla.SCALAR:
        r"""Returns the inner energy from a given state

        .. math::
            \rho E_i = \frac{p}{\gamma - 1}
            \rho E_i = \rho \frac{T}{\gamma}
        """

        gamma = self.heat_capacity_ratio
        if U.rho_Ei is not None:
            return U.rho_Ei

        elif U.p is not None:
            logger.debug("Returning inner energy from pressure.")
            return U.p/(gamma - 1)

        elif all((U.rho, U.T)):
            logger.debug("Returning inner energy from density and temperature.")
            return U.rho * U.T/gamma

        return None

    def specific_inner_energy(self, U: flowstate) -> bla.SCALAR:
        r"""Returns the specific inner energy from a given state

        .. math::
            E_i = \frac{T}{\gamma}
            E_i = \frac{p}{\rho (\gamma - 1)}
        """

        gamma = self.heat_capacity_ratio
        if U.Ei is not None:
            return U.Ei

        elif U.T is not None:
            logger.debug("Returning specific inner energy from temperature.")
            return U.T/gamma

        elif all((U.rho, U.p)):
            logger.debug("Returning specific inner energy from density and pressure.")
            return U.p/(gamma - 1)/U.rho

        return None

    def speed_of_sound(self, U: flowstate) -> bla.SCALAR:
        r"""Returns the speed of sound from a given state

        .. math::
            c = \sqrt(\gamma \frac{p}{\rho})
            c = \sqrt((\gamma - 1) T)
        """

        gamma = self.heat_capacity_ratio

        if U.c is not None:
            return U.c

        elif all((U.rho, U.p)):
            logger.debug("Returning speed of sound from pressure and density.")
            return ngs.sqrt(gamma * U.p/U.rho)

        elif U.T is not None:
            logger.debug("Returning speed of sound from temperature.")
            return ngs.sqrt((gamma - 1) * U.T)

        elif U.Ei is not None:
            logger.debug("Returning speed of sound from specific inner energy.")
            return ngs.sqrt((gamma - 1) * U.Ei/gamma)

        return None

    def density_gradient(self, U: flowstate, dU: flowstate) -> bla.VECTOR:
        r"""Returns the density gradient from a given state

        .. math::
            \nabla \rho = \frac{\gamma}{\gamma - 1} \left[ \frac{\nabla p}{T} - p \frac{\nabla T}{T^2} \right]
            \nabla \rho = \gamma \left[ \frac{ \nabla (\rho E_i)}{T} - \rho E_i \frac{\nabla T}{T^2} \right]
        """

        gamma = self.heat_capacity_ratio
        if dU.grad_rho is not None:
            return dU.grad_rho

        elif all((U.p, U.T, dU.grad_p, dU.grad_T)):
            logger.debug("Returning density gradient from pressure and temperature.")
            return gamma/(gamma - 1) * (dU.grad_p/U.T - U.p * dU.grad_T/U.T**2)

        elif all((U.T, U.rho_Ei, dU.grad_T, dU.grad_rho_Ei)):
            logger.debug("Returning density gradient from temperature and inner energy.")
            return gamma * (dU.grad_rho_Ei/U.T - U.rho_Ei * dU.grad_T/U.T**2)

    def pressure_gradient(self, U: flowstate, dU: flowstate) -> bla.VECTOR:
        r"""Returns the pressure gradient from a given state

        .. math::
            \nabla p = \frac{\gamma - 1}{\gamma} \left[ (\nabla \rho) T + (\nabla T) \rho \right]
            \nabla p = (\gamma - 1) \nabla \rho E_i
        """

        gamma = self.heat_capacity_ratio
        if dU.grad_p is not None:
            return dU.grad_p

        elif all((U.rho, U.T, dU.grad_rho, dU.grad_T)):
            logger.debug("Returning pressure gradient from density and temperature.")
            return (gamma - 1)/gamma * (dU.grad_rho * U.T + U.rho * dU.grad_T)

        elif dU.grad_rho_Ei is not None:
            logger.debug("Returning pressure gradient from inner energy gradient.")
            return (gamma - 1) * dU.grad_rho_Ei

    def temperature_gradient(self, U: flowstate,  dU: flowstate) -> bla.VECTOR:
        r"""Returns the temperature gradient from a given state

        .. math::
            \nabla T = \frac{\gamma}{\gamma - 1} \left[ \frac{\nabla p}{\rho} - p \frac{\nabla \rho}{\rho^2} \right]
            \nabla T = \gamma \nabla E_i
        """

        gamma = self.heat_capacity_ratio
        if dU.grad_T is not None:
            return dU.grad_T

        elif all((U.rho, U.p, dU.grad_p, dU.grad_rho)):
            logger.debug("Returning temperature gradient from density and pressure.")
            return gamma/(gamma - 1) * (dU.grad_p/U.rho - U.p * dU.grad_rho/U.rho**2)

        elif dU.grad_Ei is not None:
            logger.debug("Returning temperature gradient from specific inner energy gradient.")
            return gamma * dU.grad_Ei

    def characteristic_velocities(self, U: flowstate, unit_vector: bla.VECTOR, type: str = None) -> bla.VECTOR:

        unit_vector = bla.as_vector(unit_vector)

        if all((U.u, U.c)):

            un = bla.inner(U.u, unit_vector)

            lam_m_c = un - U.c
            lam = un
            lam_p_c = un + U.c

            if type is None:
                pass

            elif type == "absolute":
                lam_m_c = bla.abs(lam_m_c)
                lam = bla.abs(lam)
                lam_p_c = bla.abs(lam_p_c)

            elif type == "incoming":
                lam_m_c = bla.min(lam_m_c, 0)
                lam = bla.min(lam, 0)
                lam_p_c = bla.min(lam_p_c, 0)

            elif type == "outgoing":
                lam_m_c = bla.max(lam_m_c, 0)
                lam = bla.max(lam, 0)
                lam_p_c = bla.max(lam_p_c, 0)

            else:
                raise ValueError(
                    f"{str(type).capitalize()} invalid! Alternatives: {[None, 'absolute', 'incoming', 'outgoing']}")

            return bla.as_vector([lam_m_c] + U.u.dim * [lam] + [lam_p_c])

    def characteristic_variables(
            self, U: flowstate, dU: flowstate, unit_vector: bla.VECTOR) -> bla.VECTOR:

        unit_vector = bla.as_vector(unit_vector)

        if all((U.rho, U.c, dU.grad_rho, dU.grad_p, dU.grad_u)):

            grad_rho_n = bla.inner(dU.grad_rho, unit_vector)
            grad_p_n = bla.inner(dU.grad_p, unit_vector)
            grad_u_n = dU.grad_u * unit_vector

            if unit_vector.dim == 2:

                char = (
                    grad_p_n - bla.inner(grad_u_n, unit_vector) * U.c * U.rho,
                    grad_rho_n * U.c**2 - grad_p_n,
                    grad_u_n[0] * unit_vector[1] - grad_u_n[1] * unit_vector[0],
                    grad_p_n + bla.inner(grad_u_n, unit_vector) * U.c * U.rho
                )

            else:
                raise NotImplementedError("Characteristic Variables not implemented for 3d!")

            return bla.as_vector(char)

    def characteristic_amplitudes(self, U: flowstate, dU: flowstate, unit_vector: bla.VECTOR,
                                  type_: str = None) -> bla.VECTOR:
        """ The charachteristic amplitudes are defined as

            .. math::
                \mathcal{L} = \Lambda * L_inverse * dV/dn,

        where Lambda is the eigenvalue matrix, L_inverse is the mapping from
        primitive variables to charachteristic variables and dV/dn is the
        derivative normal to the boundary.
        """
        velocities = self.characteristic_velocities(U, unit_vector, type_)
        variables = self.characteristic_variables(U, dU, unit_vector)

        if all((velocities, variables)):
            return bla.as_vector([vel * var for vel, var in zip(velocities, variables)])

    def primitive_from_conservative(self, U: flowstate) -> bla.MATRIX:
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

        if all((U.rho, U.u)):

            if U.u.dim == 2:

                ux, uy = U.u

                Minv = (1, 0, 0, 0,
                        -ux/U.rho, 1/U.rho, 0, 0,
                        -uy/U.rho, 0, 1/U.rho, 0,
                        (gamma - 1)/2 * bla.inner(U.u, U.u), -(gamma - 1) * ux, -(gamma - 1) * uy, gamma - 1)

            else:
                raise NotImplementedError()

            dim = U.u.dim + 2

            return bla.as_matrix(Minv, dims=(dim, dim))

    def primitive_from_characteristic(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
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

        if all((U.rho, U.c)):
            if unit_vector.dim == 2:
                d0, d1 = unit_vector

                L = (0.5/U.c**2, 1/U.c**2, 0, 0.5/U.c**2,
                     -d0/(2*U.c*U.rho), 0, d1, d0/(2*U.c*U.rho),
                     -d1/(2*U.c*U.rho), 0, -d0, d1/(2*U.c*U.rho),
                     0.5, 0, 0, 0.5)
            else:
                return NotImplementedError()

            dim = unit_vector.dim + 2

            return bla.as_matrix(L, dims=(dim, dim))

    def primitive_convective_jacobian_x(self, U: flowstate) -> bla.MATRIX:

        if all((U.u, U.rho, U.c)):
            dim = U.u.dim + 2

            if dim == 4:
                ux, _ = U.u

                A = (ux, U.rho, 0, 0,
                     0, ux, 0, 1/U.rho,
                     0, 0, ux, 0,
                     0, U.rho*U.c**2, 0, ux)

            else:
                raise NotImplementedError()

            return bla.as_matrix(A, dims=(dim, dim))

    def primitive_convective_jacobian_y(self, U: flowstate) -> bla.MATRIX:

        if all((U.u, U.rho, U.c)):
            dim = U.u.dim + 2

            if dim == 4:
                _, uy = U.u

                B = (uy, 0, U.rho, 0,
                     0, uy, 0, 0,
                     0, 0, uy, 1/U.rho,
                     0, 0, U.rho*U.c**2, uy)

            else:
                raise NotImplementedError()

            return bla.as_matrix(B, dims=(dim, dim))

    def conservative_from_primitive(self, U: flowstate) -> bla.MATRIX:
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

        if all((U.rho, U.u)):
            dim = U.u.dim + 2

            if dim == 4:

                ux, uy = U.u

                M = (1, 0, 0, 0,
                     ux, U.rho, 0, 0,
                     uy, 0, U.rho, 0,
                     0.5*bla.inner(U.u, U.u), U.rho*ux, U.rho*uy, 1/(gamma - 1))
            else:
                raise NotImplementedError()

            return bla.as_matrix(M, dims=(dim, dim))

    def conservative_from_characteristic(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
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

        if all((U.rho, U.c, U.u)):
            dim = unit_vector.dim + 2

            if dim == 4:
                d0, d1 = unit_vector
                ux, uy = U.u

                P = (
                    (1 / (2 * U.c ** 2),
                     1 / U.c ** 2, 0, 1 / (2 * U.c ** 2),
                     -d0 / (2 * U.c) + ux / (2 * U.c ** 2),
                     ux / U.c ** 2, d1 * U.rho, d0 / (2 * U.c) + ux / (2 * U.c ** 2),
                     -d1 / (2 * U.c) + uy / (2 * U.c ** 2),
                     uy / U.c ** 2, -d0 * U.rho, d1 / (2 * U.c) + uy / (2 * U.c ** 2),
                     0.5 / (gamma - 1) - d0 * ux / (2 * U.c) - d1 * uy / (2 * U.c) + bla.inner(U.u, U.u) / (4 * U.c ** 2),
                     bla.inner(U.u, U.u) / (2 * U.c ** 2),
                     -d0 * U.rho * uy + d1 * U.rho * ux, 0.5 / (gamma - 1) + d0 * ux / (2 * U.c) + d1 * uy / (2 * U.c) +
                     bla.inner(U.u, U.u) / (4 * U.c ** 2)))

            else:
                raise NotImplementedError()

            return bla.as_matrix(P, dims=(dim, dim))

    def conservative_convective_jacobian_x(self, U: flowstate) -> bla.MATRIX:
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

        if all((U.u, U.E)):
            dim = U.u.dim + 2

            if dim == 4:
                ux, uy = U.u

                A = (0, 1, 0, 0,
                     (gamma - 3)/2 * ux**2 + (gamma - 1)/2 * uy**2, (3 - gamma) * ux, -(gamma - 1) * uy, gamma - 1,
                     -ux*uy, uy, ux, 0,
                     -gamma*ux*U.E + (gamma - 1)*ux*bla.inner(U.u, U.u), gamma*U.E - (gamma - 1)/2 * (uy**2 + 3*ux**2), -(gamma - 1)*ux*uy, gamma*ux)

            else:
                raise NotImplementedError()

            return bla.as_matrix(A, dims=(dim, dim))

    def conservative_convective_jacobian_y(self, U: flowstate) -> bla.MATRIX:
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

        if all((U.u, U.E)):
            dim = U.u.dim + 2

            if dim == 4:
                ux, uy = U.u

                B = (
                    0, 0, 1, 0, -ux * uy, uy, ux, 0, (gamma - 3) / 2 * uy ** 2 + (gamma - 1) / 2 * ux ** 2, -(gamma - 1) * ux,
                    (3 - gamma) * uy, gamma - 1, -gamma * uy * U.E + (gamma - 1) * uy * bla.inner(U.u, U.u), -(gamma - 1) * ux * uy, gamma * U.E -
                    (gamma - 1) / 2 * (ux ** 2 + 3 * uy ** 2),
                    gamma * uy)

            else:
                raise NotImplementedError()

            return bla.as_matrix(B, dims=(dim, dim))

    def characteristic_from_primitive(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
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

        if all((U.rho, U.c)):
            dim = unit_vector.dim + 2

            if dim == 4:
                d0, d1 = unit_vector

                Linv = (0, -U.rho*U.c*d0, -U.rho*U.c*d1, 1,
                        U.c**2, 0, 0, -1,
                        0, d1, -d0, 0,
                        0, U.rho*U.c*d0, U.rho*U.c*d1, 1)

            else:
                return NotImplementedError()

            return bla.as_matrix(Linv, dims=(dim, dim))

    def characteristic_from_conservative(self, U: flowstate, unit_vector: bla.VECTOR) -> bla.MATRIX:
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

        if all((U.rho, U.c, U.u)):
            dim = unit_vector.dim + 2

            if dim == 4:
                d0, d1 = unit_vector
                ux, uy = U.u

                Pinv = (
                    U.c*d0*ux + U.c*d1*uy + (gamma - 1)*bla.inner(U.u, U.u)/2, -U.c*d0 + ux*(1 - gamma), -U.c*d1 + uy*(1 - gamma), gamma - 1,
                    U.c**2 - (gamma - 1)*bla.inner(U.u, U.u)/2, -ux*(1 - gamma), -uy*(1 - gamma), 1 - gamma,
                    d0*uy/U.rho - d1*ux/U.rho, d1/U.rho, -d0/U.rho, 0,
                    -U.c*d0*ux - U.c*d1*uy + (gamma - 1)*bla.inner(U.u, U.u)/2, U.c*d0 + ux*(1 - gamma), U.c*d1 + uy*(1 - gamma), gamma - 1)
            else:
                raise NotImplementedError()

            return bla.as_matrix(Pinv, dims=(dim, dim))

    def isentropic_density(self, U: flowstate, Uref: flowstate) -> ngs.CF:
        r"""Returns the isentropic density from a given state

        .. math::
            \rho = \rho_{ref} (\frac{T}{T_{ref}})^{\frac{1}{\gamma - 1}}
            \rho = \rho_{ref} (\frac{p}{p_{ref}})^{\frac{1}{\gamma}}
        """
        if U.p is not None and all((Uref.rho, Uref.p)):
            logger.debug("Returning isentropic density from pressure.")
            return Uref.rho * (U.p/Uref.p)**(1/self.heat_capacity_ratio)

        elif U.T is not None and all((Uref.rho, Uref.T)):
            logger.debug("Returning isentropic density from temperature.")
            return Uref.rho * (U.T/Uref.T)**(1/(self.heat_capacity_ratio - 1))

        return None

    heat_capacity_ratio: ngs.Parameter
