""" Compressible flow solver configuration """
from __future__ import annotations

import logging
import ngsolve as ngs

from dream import bla
from dream.config import dream_configuration, equation
from dream.mesh import (BoundaryConditions, DomainConditions)
from dream.solver import SolverConfiguration
from dream.time import TransientRoutine

from .eos import IdealGas, EquationOfState
from .viscosity import Inviscid, Constant, Sutherland, DynamicViscosity
from .scaling import Aerodynamic, Aeroacoustic, Acoustic, Scaling
from .riemann_solver import LaxFriedrich, Roe, HLL, HLLEM, Upwind, RiemannSolver
from .config import flowfields, BCS, DCS, CompressibleFiniteElementMethod
from .conservative import ConservativeFiniteElementMethod

logger = logging.getLogger(__name__)


class CompressibleFlowSolver(SolverConfiguration):

    name = "compressible"

    def __init__(self, mesh: ngs.Mesh, **default) -> None:
        bcs = BoundaryConditions(mesh, BCS)
        dcs = DomainConditions(mesh, DCS)

        self._mach_number = ngs.Parameter(0.3)
        self._reynolds_number = ngs.Parameter(150)
        self._prandtl_number = ngs.Parameter(0.72)

        DEFAULT = {
            "time": TransientRoutine(mesh, self),
            "equation_of_state": IdealGas(mesh, self),
            "dynamic_viscosity": Inviscid(mesh, self),
            "scaling": Aerodynamic(mesh, self),
            "riemann_solver": LaxFriedrich(mesh, self),
            "mach_number": 0.3,
            "reynolds_number": 150,
            "prandtl_number": 0.72,
        }
        DEFAULT.update(default)
        super().__init__(mesh=mesh, bcs=bcs, dcs=dcs, **DEFAULT)

    @dream_configuration
    def fem(self) -> ConservativeFiniteElementMethod:
        r""" Sets the finite element for the compressible flow solver. 

            :getter: Returns the finite element
            :setter: Sets the finite element, defaults to ConservativeFiniteElement
        """
        return self._fem

    @fem.setter
    def fem(self, fem):
        OPTIONS = [ConservativeFiniteElementMethod]
        self._fem = self._get_configuration_option(fem, OPTIONS, CompressibleFiniteElementMethod)

    @dream_configuration
    def mach_number(self) -> ngs.Parameter:
        r""" Sets the ratio of the farfield flow velocity to the farfield speed of sound.

            .. math::
                \Ma_\infty = \frac{|\bm{u}_\infty|}{c_\infty}

            :getter: Returns the Mach number
            :setter: Sets the Mach number, defaults to 0.3
        """
        return self._mach_number

    @mach_number.setter
    def mach_number(self, mach_number: float):
        if mach_number < 0:
            raise ValueError("Invalid Mach number. Value has to be >= 0!")

        self._mach_number.Set(mach_number)

    @dream_configuration
    def reynolds_number(self) -> ngs.Parameter:
        r""" Sets the ratio of inertial to viscous forces 

            .. math::
                \Re_\infty = \frac{\rho_\infty |\bm{u}_\infty| L}{\mu_\infty}

            :getter: Returns the Reynolds number
            :setter: Sets the Reynolds number, defaults to 150
        """
        return self._reynolds_number

    @reynolds_number.setter
    def reynolds_number(self, reynolds_number: float):
        if reynolds_number <= 0:
            raise ValueError("Invalid Reynolds number. Value has to be > 0!")
        self._reynolds_number.Set(reynolds_number)

    @dream_configuration
    def prandtl_number(self) -> ngs.Parameter:
        r""" Sets the ratio of momentum diffusivity to thermal diffusivity 

            .. math::
                \Pr_\infty = \frac{c_p \mu_\infty}{k_\infty}

            :getter: Returns the Prandtl number
            :setter: Sets the Prandtl number, defaults to 0.72
        """
        return self._prandtl_number

    @prandtl_number.setter
    def prandtl_number(self, prandtl_number: float):
        if prandtl_number <= 0:
            raise ValueError("Invalid Prandtl number. Value has to be > 0!")
        self._prandtl_number.Set(prandtl_number)

    @dream_configuration
    def equation_of_state(self) -> IdealGas:
        r""" Sets the equation of state for the compressible flow solver. 

            :getter: Returns the equation of state
            :setter: Sets the equation of state, defaults to IdealGas
        """
        return self._equation_of_state

    @equation_of_state.setter
    def equation_of_state(self, equation_of_state: IdealGas):
        OPTIONS = [IdealGas]
        self._equation_of_state = self._get_configuration_option(equation_of_state, OPTIONS, EquationOfState)

    @dream_configuration
    def dynamic_viscosity(self) -> Inviscid | Constant | Sutherland:
        r""" Sets the dynamic viscosity for the compressible flow solver.

            :getter: Returns the dynamic viscosity
            :setter: Sets the dynamic viscosity, defaults to Inviscid
        """
        return self._dynamic_viscosity

    @dynamic_viscosity.setter
    def dynamic_viscosity(self, dynamic_viscosity: str | Inviscid | Constant | Sutherland):
        OPTIONS = [Inviscid, Constant, Sutherland]
        self._dynamic_viscosity = self._get_configuration_option(dynamic_viscosity, OPTIONS, DynamicViscosity)

    @dream_configuration
    def scaling(self) -> Aerodynamic | Acoustic | Aeroacoustic:
        r""" Sets the dimensional scaling for the compressible flow solver.

            :getter: Returns the scaling
            :setter: Sets the scaling, defaults to Aerodynamic
        """
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: str | Aerodynamic | Acoustic | Aeroacoustic):
        OPTIONS = [Aerodynamic, Acoustic, Aeroacoustic]
        self._scaling = self._get_configuration_option(scaling, OPTIONS, Scaling)

    @dream_configuration
    def riemann_solver(self) -> LaxFriedrich | Roe | HLL | HLLEM | Upwind:
        r""" Sets the Riemann solver for the compressible flow solver.

            :getter: Returns the Riemann solver
            :setter: Sets the Riemann solver, defaults to LaxFriedrich
        """
        return self._riemann_solver

    @riemann_solver.setter
    def riemann_solver(self, riemann_solver: str | LaxFriedrich | Roe | HLL | HLLEM | Upwind):
        OPTIONS = [LaxFriedrich, Roe, HLL, HLLEM, Upwind]
        self._riemann_solver = self._get_configuration_option(riemann_solver, OPTIONS, RiemannSolver)

    def get_farfield_state(self, direction: tuple[float, ...]) -> flowfields:
        r""" Returns the dimensionless farfield fields depending on the scaling in use and the flow direction. 

            Aerodynamic Scaling
                .. math:: 
                    \begin{align*}
                        \rho &= 1, &
                        | \bm{u} |  &= 1, &
                        c &= \frac{1}{\Ma_\infty}, &
                        T &= \frac{1}{(\gamma - 1)\Ma_\infty^2}, &
                        p &= \frac{1}{\gamma \Ma_\infty^2}.
                    \end{align*}

            Acoustic Scaling
                .. math::
                    \begin{align*}
                        \rho &= 1, &
                        | \bm{u} |  &= \Ma_\infty, &
                        c &= 1, &
                        T &= \frac{1}{(\gamma - 1)}, &
                        p &= \frac{1}{\gamma}.
                    \end{align*}

            Aeroacoustic Scaling
                .. math::
                    \begin{align*}
                        \rho &= 1, &
                        | \bm{u} |  &= \frac{\Ma_\infty}{1 + \Ma_\infty}, &
                        c &= \frac{1}{1 + \Ma_\infty}, &
                        T &= \frac{1}{(\gamma - 1) ( 1+ \Ma_\infty^2)}, &
                        p &= \frac{1}{\gamma (1+ \Ma_\infty^2)}.
                    \end{align*}

            :param direction: A container containing the flow direction
            :type direction: tuple[float, ...]
        """

        Ma = self.mach_number
        INF = flowfields()

        INF.rho = self.scaling.density()
        INF.c = self.scaling.speed_of_sound(Ma)
        INF.T = self.temperature(INF)
        INF.p = self.pressure(INF)

        direction = bla.as_vector(direction)

        if not direction.dim == self.mesh.dim:
            raise ValueError(f"Direction dimension {direction.dim} does not match mesh dimension {self.mesh.dim}")

        INF.u = self.scaling.velocity(direction, Ma)
        INF.rho_Ei = self.inner_energy(INF)
        INF.rho_Ek = self.kinetic_energy(INF)
        INF.rho_E = self.energy(INF)

        return INF

    def get_dimensionful_state(self, U: flowfields) -> flowfields:
        r""" Returns the dimensionful fields from given fields depending on the scaling. 

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
        """

        REF = self.scaling.dimensionful_values
        DIM = flowfields()
        for key, value in U.items():
            if key in REF:
                DIM[key] = value * REF[key]

        return DIM

    def get_convective_flux(self, U: flowfields) -> bla.MATRIX:
        r""" Returns the conservative convective flux from given fields.

            .. math::
                \bm{F} = \begin{pmatrix} \rho \bm{u} \\ \rho \bm{u} \otimes \bm{u} + p \bm{I} \\ \rho H \bm{u} \end{pmatrix}

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
        """
        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_H = self.enthalpy(U)
        u = self.velocity(U)
        p = self.pressure(U)

        flux = (rho_u, bla.outer(rho_u, rho_u)/rho + p * ngs.Id(u.dim), rho_H * u)

        return bla.as_matrix(flux, dims=(u.dim + 2, u.dim))

    def get_diffusive_flux(self, U: flowfields, dU: flowfields) -> bla.MATRIX:
        r""" Returns the conservative diffusive flux from given states.

            .. math::
                \bm{G} = \begin{pmatrix} \bm{0} \\ \bm{\tau} \\ \left( \bm{\tau} \bm{u} - \bm{q}\right) \end{pmatrix}

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
            :param dU: A dictionary containing the gradients of the flow quantities
            :type dU: flowfields
        """
        u = self.velocity(U)
        tau = self.deviatoric_stress_tensor(U, dU)
        q = self.heat_flux(U, dU)

        continuity = tuple(0 for _ in range(u.dim))

        return bla.as_matrix((continuity, tau, tau*u - q), dims=(u.dim + 2, u.dim))

    def get_local_mach_number(self, U: flowfields):
        r""" Returns the local Mach number from given fields.

            .. math::
                \Ma = \frac{| \bm{u} |}{c}

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
        """
        u = self.velocity(U)
        c = self.speed_of_sound(U)
        return ngs.sqrt(bla.inner(u, u))/c

    def get_local_reynolds_number(self, U: flowfields):
        r""" Returns the local Reynolds number from given fields.

            .. math::
                \Re = \frac{\rho | \bm{u} |}{\mu}

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
        """
        rho = self.density(U)
        u = self.velocity(U)
        mu = self.viscosity(U)
        return rho * ngs.sqrt(bla.inner(u, u)) / mu

    def get_primitive_convective_jacobian(self, U: flowfields, unit_vector: bla.VECTOR, type: str = None):
        lambdas = self.characteristic_velocities(U, unit_vector, type)
        LAMBDA = bla.diagonal(lambdas)
        return self.transform_characteristic_to_primitive(LAMBDA, U, unit_vector)

    def get_primitive_convective_identity(self, U: flowfields, unit_vector: bla.VECTOR, type: str = None):
        LAMBDA = self.get_characteristic_identity(U, unit_vector, type)
        return self.transform_characteristic_to_primitive(LAMBDA, U, unit_vector)

    def get_conservative_convective_jacobian(self, U: flowfields, unit_vector: bla.VECTOR, type: str = None):
        lambdas = self.characteristic_velocities(U, unit_vector, type)
        LAMBDA = bla.diagonal(lambdas)
        return self.transform_characteristic_to_conservative(LAMBDA, U, unit_vector)

    def get_conservative_convective_identity(self, U: flowfields, unit_vector: bla.VECTOR, type: str = None):
        LAMBDA = self.get_characteristic_identity(U, unit_vector, type)
        return self.transform_characteristic_to_conservative(LAMBDA, U, unit_vector)

    def get_characteristic_identity(
            self, U: flowfields, unit_vector: bla.VECTOR, type: str = None) -> bla.MATRIX:
        lambdas = self.characteristic_velocities(U, unit_vector, type)

        if type == "incoming":
            lambdas = (ngs.IfPos(-lam, 1, 0) for lam in lambdas)
        elif type == "outgoing":
            lambdas = (ngs.IfPos(lam, 1, 0) for lam in lambdas)
        else:
            raise ValueError(f"{str(type).capitalize()} invalid! Alternatives: {['incoming', 'outgoing']}")

        return bla.diagonal(lambdas)

    def transform_primitive_to_conservative(self, matrix: bla.MATRIX, U: flowfields):
        M = self.conservative_from_primitive(U)
        Minv = self.primitive_from_conservative(U)
        return M * matrix * Minv

    def transform_characteristic_to_primitive(self, matrix: bla.MATRIX, U: flowfields, unit_vector: bla.VECTOR):
        L = self.primitive_from_characteristic(U, unit_vector)
        Linv = self.characteristic_from_primitive(U, unit_vector)
        return L * matrix * Linv

    def transform_characteristic_to_conservative(self, matrix: bla.MATRIX, U: flowfields, unit_vector: bla.VECTOR):
        P = self.conservative_from_characteristic(U, unit_vector)
        Pinv = self.characteristic_from_conservative(U, unit_vector)
        return P * matrix * Pinv

    @equation
    def density(self, U: flowfields) -> bla.SCALAR:
        return self.equation_of_state.density(U)

    @equation
    def velocity(self, U: flowfields) -> bla.VECTOR:
        if U.u is not None:
            return U.u

        elif all((U.rho, U.rho_u)):
            logger.debug("Returning velocity from density and momentum.")

            if bla.is_zero(U.rho_u) and bla.is_zero(U.rho):
                return bla.as_vector((0.0 for _ in range(U.rho_u.dim)))

            return U.rho_u/U.rho

    @equation
    def momentum(self, U: flowfields) -> bla.VECTOR:
        if U.rho_u is not None:
            return U.rho_u

        elif all((U.rho, U.u)):
            logger.debug("Returning momentum from density and velocity.")
            return U.rho * U.u

    @equation
    def pressure(self, U: flowfields) -> bla.SCALAR:
        return self.equation_of_state.pressure(U)

    @equation
    def temperature(self, U: flowfields) -> bla.SCALAR:
        return self.equation_of_state.temperature(U)

    @equation
    def inner_energy(self, U: flowfields) -> bla.SCALAR:
        rho_Ei = self.equation_of_state.inner_energy(U)

        if rho_Ei is None:

            if all((U.rho_E, U.rho_Ek)):
                logger.debug("Returning inner energy from energy and kinetic energy.")
                return U.rho_E - U.rho_Ek

        return rho_Ei

    @equation
    def specific_inner_energy(self, U: flowfields) -> bla.SCALAR:
        Ei = self.equation_of_state.specific_inner_energy(U)

        if Ei is None:

            if all((U.rho, U.rho_Ei)):
                logger.debug("Returning specific inner energy from inner energy and density.")
                return U.rho_Ei/U.rho

            elif all((U.E, U.Ek)):
                logger.debug(
                    "Returning specific inner energy from specific energy and specific kinetic energy.")
                return U.E - U.Ek

        return Ei

    @equation
    def kinetic_energy(self, U: flowfields) -> bla.SCALAR:
        if U.rho_Ek is not None:
            return U.rho_Ek

        elif all((U.rho, U.u)):
            logger.debug("Returning kinetic energy from density and velocity.")
            return 0.5 * U.rho * bla.inner(U.u, U.u)

        elif all((U.rho, U.rho_u)):
            logger.debug("Returning kinetic energy from density and momentum.")
            return 0.5 * bla.inner(U.rho_u, U.rho_u)/U.rho

        elif all((U.rho_E, U.rho_Ei)):
            logger.debug("Returning kinetic energy from energy and inner energy.")
            return U.rho_E - U.rho_Ei

        elif all((U.rho, U.Ek)):
            logger.debug("Returning kinetic energy from density and specific kinetic energy.")
            return U.rho * U.Ek

        return None

    @equation
    def specific_kinetic_energy(self, U: flowfields) -> bla.SCALAR:
        if U.Ek is not None:
            return U.Ek

        elif U.u is not None:
            logger.debug("Returning specific kinetic energy from velocity.")
            return 0.5 * bla.inner(U.u, U.u)

        elif all((U.rho, U.rho_u)):
            logger.debug("Returning specific kinetic energy from density and momentum.")
            return 0.5 * bla.inner(U.rho_u, U.rho_u)/U.rho**2

        elif all((U.rho, U.rho_Ek)):
            logger.debug("Returning specific kinetic energy from density and kinetic energy.")
            return U.rho_Ek/U.rho

        elif all((U.E, U.Ei)):
            logger.debug("Returning specific kinetic energy from specific energy and speicific inner energy.")
            return U.E - U.Ei

        return None

    @equation
    def energy(self, U: flowfields) -> bla.SCALAR:
        if U.rho_E is not None:
            return U.rho_E

        elif all((U.rho, U.E)):
            logger.debug("Returning energy from density and specific energy.")
            return U.rho * U.E

        elif all((U.rho_Ei, U.rho_Ek)):
            logger.debug("Returning energy from inner energy and kinetic energy.")
            return U.rho_Ei + U.rho_Ek

        else:
            logger.debug("Returning energy from calculated inner energy and kinetic energy")
            return self.inner_energy(U) + self.kinetic_energy(U)

    @equation
    def specific_energy(self, U: flowfields) -> bla.SCALAR:
        if U.E is not None:
            return U.E

        if all((U.rho, U.rho_E)):
            logger.debug("Returning specific energy from density and energy.")
            return U.rho_E/U.rho

        elif all((U.Ei, U.Ek)):
            logger.debug("Returning specific energy from specific inner energy and specific kinetic energy.")
            return U.Ei + U.Ek

        return None

    @equation
    def enthalpy(self, U: flowfields) -> bla.SCALAR:
        if U.rho_H is not None:
            return U.rho_H

        elif all((U.rho_E, U.p)):
            logger.debug("Returning enthalpy from energy and pressure.")
            return U.rho_E + U.p

        elif all((U.rho, U.H)):
            logger.debug("Returning enthalpy from density and specific enthalpy.")
            return U.rho * U.H

        return None

    @equation
    def specific_enthalpy(self, U: flowfields) -> bla.SCALAR:
        if U.H is not None:
            return U.H

        elif all((U.rho, U.rho_H)):
            logger.debug("Returning specific enthalpy from density and enthalpy.")
            return U.rho_H/U.rho

        elif all((U.rho, U.rho_E, U.p)):
            logger.debug("Returning specific enthalpy from specific energy, density and pressure.")
            return U.E + U.p/U.rho

        return None

    @equation
    def speed_of_sound(self, U: flowfields) -> bla.SCALAR:
        return self.equation_of_state.speed_of_sound(U)

    @equation
    def density_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        return self.equation_of_state.density_gradient(U, dU)

    @equation
    def velocity_gradient(self, U: flowfields, dU: flowfields) -> bla.MATRIX:
        if dU.grad_u is not None:
            return dU.grad_u
        elif all((U.rho, U.rho_u, dU.grad_rho, dU.grad_rho_u)):
            logger.debug("Returning velocity gradient from density and momentum.")
            return dU.grad_rho_u/U.rho - bla.outer(U.rho_u, dU.grad_rho)/U.rho**2

    @equation
    def momentum_gradient(self, U: flowfields, dU: flowfields) -> bla.MATRIX:
        if dU.grad_rho_u is not None:
            return dU.grad_rho_u
        elif all((U.rho, U.u, dU.grad_rho, dU.grad_u)):
            logger.debug("Returning momentum gradient from density and momentum.")
            return U.rho * dU.grad_u + bla.outer(U.u, dU.grad_rho)

    @equation
    def pressure_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        return self.equation_of_state.pressure_gradient(U, dU)

    @equation
    def temperature_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        return self.equation_of_state.temperature_gradient(U, dU)

    @equation
    def energy_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        if dU.grad_rho_E is not None:
            return dU.grad_rho_E
        elif all((dU.grad_rho_Ei, dU.grad_rho_Ek)):
            logger.debug("Returning energy gradient from inner energy and kinetic energy.")
            return dU.grad_rho_Ei + dU.grad_rho_Ek

    @equation
    def specific_energy_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        if dU.grad_E is not None:
            return dU.grad_E
        elif all((dU.grad_Ei, dU.grad_Ek)):
            logger.debug(
                "Returning specific energy gradient from specific inner energy and specific kinetic energy.")
            return dU.grad_Ei + dU.grad_Ek

    @equation
    def inner_energy_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        if dU.grad_rho_Ei is not None:
            return dU.grad_rho_Ei
        elif all((dU.grad_rho_E, dU.grad_rho_Ek)):
            logger.debug("Returning inner energy gradient from energy and kinetic energy.")
            return dU.grad_rho_E - dU.grad_rho_Ek

    @equation
    def specific_inner_energy_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        if dU.grad_Ei is not None:
            return dU.grad_Ei
        elif all((dU.grad_E, dU.grad_Ek)):
            logger.debug(
                "Returning specific inner energy gradient from specific energy and specific kinetic energy.")
            return dU.grad_E - dU.grad_Ek

    @equation
    def kinetic_energy_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        if dU.grad_rho_Ek is not None:
            return dU.grad_rho_Ek
        elif all((dU.grad_rho_E, dU.grad_rho_Ei)):
            logger.debug("Returning kinetic energy gradient from energy and inner energy.")
            return dU.grad_rho_E - dU.grad_rho_Ei

        elif all((U.rho, U.u, dU.grad_rho, dU.grad_u)):
            logger.debug("Returning kinetic energy gradient from density and velocity.")
            return U.rho * (dU.grad_u.trans * U.u) + 0.5 * dU.grad_rho * bla.inner(U.u, U.u)

        elif all((U.rho, U.rho_u, dU.grad_rho, dU.grad_rho_u)):
            logger.debug("Returning kinetic energy gradient from density and momentum.")
            return (dU.grad_rho_u.trans * U.rho_u)/U.rho - 0.5 * dU.grad_rho * bla.inner(U.rho_u, U.rho_u)/U.rho**2

    @equation
    def specific_kinetic_energy_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        if dU.grad_Ek is not None:
            return dU.grad_Ek
        elif all((dU.grad_E, dU.grad_Ei)):
            logger.debug(
                "Returning specific kinetic energy gradient from specific energy and specific inner energy.")
            return dU.grad_E - dU.grad_Ei

        elif all((U.u, dU.grad_u)):
            logger.debug("Returning specific kinetic energy gradient from velocity.")
            return dU.grad_u.trans * U.u

        elif all((U.rho, U.rho_u, dU.grad_rho, dU.grad_rho_u)):
            logger.debug("Returning specific kinetic energy gradient from density and momentum.")
            return (dU.grad_rho_u.trans * U.rho_u)/U.rho**2 - dU.grad_rho * bla.inner(U.rho_u, U.rho_u)/U.rho**3

    @equation
    def enthalpy_gradient(self, U: flowfields, dU: flowfields) -> bla.VECTOR:
        if dU.grad_rho_H is not None:
            return dU.grad_rho_H
        elif all((dU.grad_rho_E, dU.grad_p)):
            logger.debug("Returning enthalpy gradient from energy and pressure.")
            return dU.grad_rho_E + dU.grad_p

    @equation
    def strain_rate_tensor(self, dU: flowfields) -> bla.MATRIX:
        if dU.eps is not None:
            return dU.eps
        elif dU.grad_u is not None:
            logger.debug("Returning strain rate tensor from velocity.")
            return 0.5 * (dU.grad_u + dU.grad_u.trans) - 1/3 * bla.trace(dU.grad_u) * ngs.Id(self.mesh.dim)

    @equation
    def deviatoric_stress_tensor(self, U: flowfields, dU: flowfields):
        r""" Returns the deviatoric stress tensor from the given states. 

            .. math::
                \bm{\tau} = \frac{2\mu}{\Re_r} \bm{\varepsilon}

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
            :param dU: A dictionary containing the gradients of the flow quantities
            :type dU: flowfields
        """

        mu = self.viscosity(U)
        Re = self.scaling.reference_reynolds_number
        eps = self.strain_rate_tensor(dU)

        if all((mu, eps)):
            logger.debug("Returning deviatoric stress tensor from strain rate tensor and viscosity.")
            return 2*mu/Re * eps

    @equation
    def viscosity(self, U: flowfields) -> bla.SCALAR:
        return self.dynamic_viscosity.viscosity(U)

    @equation
    def heat_flux(self, U: flowfields, dU: flowfields) -> bla.VECTOR:

        k = self.viscosity(U)
        Re = self.scaling.reference_reynolds_number
        Pr = self.prandtl_number

        if all((k, dU.grad_T)):
            logger.debug("Returning heat flux from temperature gradient.")
            return -k/(Re * Pr) * dU.grad_T

    @equation
    def characteristic_velocities(self, U: flowfields, unit_vector: bla.VECTOR, type: str = None) -> bla.VECTOR:
        return self.equation_of_state.characteristic_velocities(U, unit_vector, type)

    @equation
    def characteristic_variables(
            self, U: flowfields, dU: flowfields, unit_vector: bla.VECTOR) -> bla.VECTOR:
        return self.equation_of_state.characteristic_variables(U, dU, unit_vector)

    @equation
    def characteristic_amplitudes(self, U: flowfields, dU: flowfields, unit_vector: bla.VECTOR,
                                  type: str = None) -> bla.VECTOR:
        return self.equation_of_state.characteristic_amplitudes(U, dU, unit_vector, type)

    @equation
    def primitive_from_conservative(self, U: flowfields) -> bla.MATRIX:
        return self.equation_of_state.primitive_from_conservative(U)

    @equation
    def primitive_from_characteristic(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.equation_of_state.primitive_from_characteristic(U, unit_vector)

    @equation
    def primitive_convective_jacobian_x(self, U: flowfields) -> bla.MATRIX:
        return self.equation_of_state.primitive_convective_jacobian_x(U)

    @equation
    def primitive_convective_jacobian_y(self, U: flowfields) -> bla.MATRIX:
        return self.equation_of_state.primitive_convective_jacobian_y(U)

    @equation
    def conservative_from_primitive(self, U: flowfields) -> bla.MATRIX:
        return self.equation_of_state.conservative_from_primitive(U)

    @equation
    def conservative_from_characteristic(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.equation_of_state.conservative_from_characteristic(U, unit_vector)

    @equation
    def conservative_convective_jacobian_x(self, U: flowfields) -> bla.MATRIX:
        return self.equation_of_state.conservative_convective_jacobian_x(U)

    @equation
    def conservative_convective_jacobian_y(self, U: flowfields) -> bla.MATRIX:
        return self.equation_of_state.conservative_convective_jacobian_y(U)

    @equation
    def characteristic_from_primitive(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.equation_of_state.characteristic_from_primitive(U, unit_vector)

    @equation
    def characteristic_from_conservative(self, U: flowfields, unit_vector: bla.VECTOR) -> bla.MATRIX:
        return self.equation_of_state.characteristic_from_conservative(U, unit_vector)

    @equation
    def isentropic_density(self, U: flowfields, Uref: flowfields) -> bla.SCALAR:
        return self.equation_of_state.isentropic_density(U, Uref)

    @equation
    def pressure_coefficient(self, U: flowfields, Uref: flowfields):
        return (self.pressure(U) - self.pressure(Uref))/(self.kinetic_energy(Uref))

    @equation
    def stress_tensor(self, U: flowfields, dU: flowfields = None) -> bla.MATRIX:

        sigma = -self.pressure(U) * ngs.Id(self.mesh.dim)
        if not self.dynamic_viscosity.is_inviscid:
            sigma += self.deviatoric_stress_tensor(U, dU)

        return sigma

    @equation
    def drag_coefficient(
            self, U: flowfields, dU: flowfields, Uref: flowfields, drag_direction: tuple[float, ...]) -> bla.SCALAR:
        stress = self.stress_tensor(U, dU) * self.mesh.normal
        return bla.inner(stress, bla.unit_vector(drag_direction))/(self.kinetic_energy(Uref))

    @equation
    def lift_coefficient(
            self, U: flowfields, dU: flowfields, Uref: flowfields, lift_direction: tuple[float, ...]) -> bla.SCALAR:
        stress = self.stress_tensor(U, dU) * self.mesh.normal
        return bla.inner(stress, bla.unit_vector(lift_direction))/(self.kinetic_energy(Uref))
