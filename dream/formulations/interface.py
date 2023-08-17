from __future__ import annotations
import abc
import enum
import dataclasses
from typing import Optional, TYPE_CHECKING, NamedTuple

from ngsolve import *

from ..time_schemes import time_scheme_factory, TimeLevelsGridfunction
from ..region import DreamMesh
from ..state import IdealGasCalculator
from ..crs import Inviscid, Constant, Sutherland

import logging
logger = logging.getLogger("DreAm.Formulations")

if TYPE_CHECKING:
    from configuration import SolverConfiguration
    from ngsolve.comp import ProxyFunction


class CompressibleFormulations(enum.Enum):
    PRIMITIVE = "primitive"
    CONSERVATIVE = "conservative"


class MixedMethods(enum.Enum):
    NONE = None
    GRADIENT = "gradient"
    STRAIN_HEAT = "strain_heat"


class RiemannSolver(enum.Enum):
    LAX_FRIEDRICH = 'lax_friedrich'
    ROE = 'roe'
    HLL = 'hll'
    HLLEM = 'hllem'


class Scaling(enum.Enum):
    ACOUSTIC = 'acoustic'
    AERODYNAMIC = 'aerodynamic'
    AEROACOUSTIC = 'aeroacoustic'


@dataclasses.dataclass
class TensorIndices:
    XX: Optional[int] = None
    XY: Optional[int] = None
    XZ: Optional[int] = None
    YX: Optional[int] = None
    YY: Optional[int] = None
    YZ: Optional[int] = None
    ZX: Optional[int] = None
    ZY: Optional[int] = None
    ZZ: Optional[int] = None

    def __post_init__(self):
        coordinates = vars(self).copy()
        for attr, value in coordinates.items():
            if value is None:
                delattr(self, attr)

    def __len__(self):
        return len(vars(self))

    def __iter__(self):
        for value in vars(self).values():
            yield value


class FiniteElementSpace(NamedTuple):

    space: Optional[ProductSpace] = None
    TnT: Optional[Settings] = None
    components: Optional[Settings] = None

    class Settings:

        def __init__(self,
                     PRIMAL,
                     PRIMAL_FACET,
                     MIXED=None,
                     PML=None,
                     NSCBC=None,
                     **kwargs) -> None:

            self.PRIMAL = PRIMAL
            self.PRIMAL_FACET = PRIMAL_FACET
            self.MIXED = MIXED
            self.PML = PML
            self.NSCBC = NSCBC
            self.__dict__.update(**kwargs)

        def __repr__(self) -> str:
            return "(" + ", ".join([f"{key}: {val}" for key, val in vars(self).items()]) + ")"

    @classmethod
    def from_settings(cls, settings: Settings):

        fes = settings.PRIMAL * settings.PRIMAL_FACET

        settings = vars(settings)
        spaces = {type: space for type, space in settings.items() if space is not None}

        non_default_spaces = tuple(spaces.values())[2:]
        for space in non_default_spaces:
            fes *= space

        TnTs = {type: (None, None) for type in settings.keys()}
        TnT = ((trial, test) for trial, test in zip(*fes.TnT()))
        TnT = {type: TnT for type, TnT in zip(spaces.keys(), TnT)}
        TnTs.update(TnT)
        TnTs = cls.Settings(**TnTs)

        components = {type: idx for idx, type in enumerate(spaces)}
        components = cls.Settings(**components)

        return cls(fes, TnTs, components)


class Formulation(abc.ABC):

    @abc.abstractmethod
    def _initialize_FE_space(self) -> FiniteElementSpace: ...

    @abc.abstractmethod
    def add_time_bilinearform(self, blf) -> None: ...

    @abc.abstractmethod
    def add_convective_bilinearform(self, blf) -> None: ...

    @abc.abstractmethod
    def add_diffusive_bilinearform(self, blf) -> None: ...

    def density(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def velocity(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def momentum(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def pressure(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def temperature(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def energy(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def specific_energy(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def enthalpy(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def specific_enthalpy(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def kinetic_energy(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def specific_kinetic_energy(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def inner_energy(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def specific_inner_energy(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def speed_of_sound(self, U: Optional[CF] = None):
        raise NotImplementedError()

    def density_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def velocity_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def momentum_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def pressure_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def temperature_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def energy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def enthalpy_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def vorticity(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def deviatoric_strain_tensor(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        raise NotImplementedError()

    def _add_dirichlet_bilinearform(self, blf, boundary, condition):
        raise NotImplementedError()

    def _add_farfield_bilinearform(self, blf, boundary, condition):
        raise NotImplementedError()

    def _add_outflow_bilinearform(self, blf, boundary, condition):
        raise NotImplementedError()

    def _add_nonreflecting_outflow_bilinearform(self, blf, boundary, condition):
        raise NotImplementedError()

    def _add_nonreflecting_inflow_bilinearform(self, blf, boundary, condition):
        raise NotImplementedError()

    def _add_inviscid_wall_bilinearform(self, blf, boundary, condition):
        raise NotImplementedError()

    def _add_isothermal_wall_bilinearform(self, blf, boundary, condition):
        raise NotImplementedError()

    def _add_adiabatic_wall_bilinearform(self, blf, boundary, condition):
        raise NotImplementedError()

    def add_initial_linearform(self, lf):
        raise NotImplementedError()

    def add_perturbation_linearform(self, lf):
        raise NotImplementedError()

    def add_forcing_linearform(self, lf):
        raise NotImplementedError()

    def _add_sponge_bilinearform(self, blf, domain, condition, weight_function):
        raise NotImplementedError()

    def _add_psponge_bilinearform(self, blf, domain, condition, weight_function):
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__


class _Formulation(Formulation):

    def __init__(self, mesh: Mesh, solver_configuration: SolverConfiguration) -> None:

        if isinstance(mesh, Mesh):
            mesh = DreamMesh(mesh)

        self._mesh = mesh
        self._cfg = solver_configuration
        self._calc = IdealGasCalculator(self.cfg.heat_capacity_ratio)

        self.time_scheme = time_scheme_factory(solver_configuration.time)

        self._gfus = None
        self._fes = None
        self._TnT = None

        self.normal = specialcf.normal(mesh.dim)
        self.tangential = specialcf.tangential(mesh.dim)
        self.meshsize = specialcf.mesh_size

    @property
    def dmesh(self) -> DreamMesh:
        return self._mesh

    @property
    def mesh(self) -> Mesh:
        return self.dmesh.mesh

    @property
    def cfg(self) -> SolverConfiguration:
        return self._cfg

    @property
    def gfu(self) -> GridFunction:
        return self.gridfunctions['n+1']

    @property
    def gridfunctions(self) -> TimeLevelsGridfunction:
        if self._gfus is None:
            raise RuntimeError("Call 'formulation.initialize()' before accessing the GridFunction")
        return self._gfus

    @property
    def fes(self) -> ProductSpace:
        if self._fes is None:
            raise RuntimeError("Call 'formulation.initialize()' before accessing the Finite Element space")
        return self._fes.space

    @property
    def TnT(self):
        if self._fes is None:
            raise RuntimeError("Call 'formulation.initialize()' before accessing the TestAndTrialFunctions")
        return self._fes.TnT

    @property
    def calc(self) -> IdealGasCalculator:
        return self._calc

    def initialize(self):
        self._fes = self._initialize_FE_space()
        self._gfus = TimeLevelsGridfunction({level: GridFunction(self.fes) for level in self.time_scheme.time_levels})

    def update_gridfunctions(self, initial_value: bool = False):
        if initial_value:
            self.time_scheme.update_initial_solution(self._gfus)
        else:
            self.time_scheme.update_previous_solution(self._gfus)

    def add_boundary_conditions_bilinearform(self, blf):

        bcs = self.dmesh.bcs

        for name, condition in bcs.items():
            boundary = self.dmesh.boundary(name)

            if isinstance(condition, bcs.Dirichlet):
                self._add_dirichlet_bilinearform(blf, boundary, condition)

            elif isinstance(condition, bcs.FarField):
                self._add_farfield_bilinearform(blf, boundary, condition)

            elif isinstance(condition, bcs.Outflow):
                self._add_outflow_bilinearform(blf, boundary, condition)

            elif isinstance(condition, bcs.Outflow_NSCBC):
                self._add_nonreflecting_outflow_bilinearform(blf, boundary, condition)

            elif isinstance(condition, (bcs.InviscidWall, bcs.Symmetry)):
                self._add_inviscid_wall_bilinearform(blf, boundary, condition)

            elif isinstance(condition, bcs.IsothermalWall):
                self._add_isothermal_wall_bilinearform(blf, boundary, condition)

            elif isinstance(condition, bcs.AdiabaticWall):
                self._add_adiabatic_wall_bilinearform(blf, boundary, condition)

            elif isinstance(condition, bcs.Periodic):
                continue

            else:
                logger.warn(f"Boundary condition for '{name}' has not been set!")

    def add_domain_conditions_bilinearform(self, blf):

        sponge_layers = self.dmesh.dcs.sponge_layers
        if sponge_layers:
            weight_function = self.dmesh.get_sponge_weight_function()

            for name, condition in sponge_layers.items():
                domain = self.dmesh.domain(name)
                self._add_sponge_bilinearform(blf, domain, condition, weight_function)

        psponge_layers = self.dmesh.dcs.psponge_layers
        if psponge_layers:
            weight_function = self.dmesh.get_psponge_weight_function()
            for name, condition in psponge_layers.items():
                domain = self.dmesh.domain(name)
                self._add_psponge_bilinearform(blf, domain, condition, weight_function)

    def add_mass_bilinearform(self, blf):
        mixed_method = self.cfg.mixed_method

        U, V = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        blf += U * V * dx
        blf += Uhat * Vhat * dx(element_boundary=True)

        if mixed_method is not MixedMethods.NONE:
            Q, P = self.TnT.MIXED
            blf += Q * P * dx

    def mach_number(self, U: Optional[CF] = None):

        u = self.velocity(U)
        c = self.speed_of_sound(U)

        return sqrt(InnerProduct(u, u)) / c

    def convective_flux(self, U):
        """
        Convective flux F

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        dim = self.mesh.dim

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_H = self.enthalpy(U)
        u = self.velocity(U)
        p = self.pressure(U)

        flux = tuple([rho_u, OuterProduct(rho_u, rho_u)/rho + p * Id(dim), rho_H * u])

        return CF(flux, dims=(dim + 2, dim))

    def convective_stabilisation_matrix(self, Uhat, unit_vector):
        riemann_solver = self.cfg.riemann_solver
        un = InnerProduct(self.velocity(Uhat), unit_vector)
        c = self.speed_of_sound(Uhat)
        un_abs = IfPos(un, un, -un)
        splus = IfPos(un + c, un + c, 0)

        if riemann_solver is RiemannSolver.LAX_FRIEDRICH:
            lambda_max = un_abs + c
            stabilisation_matrix = lambda_max * Id(self.mesh.dim + 2)

        elif riemann_solver is RiemannSolver.ROE:
            Lambda_abs = self.characteristic_velocities(Uhat, unit_vector, type="absolute", as_matrix=True)
            stabilisation_matrix = self.DME_from_CHAR_matrix(Lambda_abs, Uhat, unit_vector)

        elif riemann_solver is RiemannSolver.HLL:
            stabilisation_matrix = splus * Id(self.mesh.dim + 2)

        elif riemann_solver is RiemannSolver.HLLEM:
            theta_0 = 1e-8
            theta = un_abs/(un_abs + c)
            theta = IfPos(theta - theta_0, theta, theta_0)

            if self.mesh.dim == 2:
                Theta = CF((1, 0, 0, 0,
                            0, theta, 0, 0,
                            0, 0, theta, 0,
                            0, 0, 0, 1), dims=(4, 4))

            elif self.mesh.dim == 3:
                Theta = CF((1, 0, 0, 0, 0,
                            0, theta, 0, 0, 0,
                            0, 0, theta, 0, 0,
                            0, 0, 0, theta, 0,
                            0, 0, 0, 0, 1), dims=(5, 5))

            Theta = self.DME_from_CHAR_matrix(Theta, Uhat, unit_vector)

            stabilisation_matrix = splus * Theta

        return stabilisation_matrix

    def diffusive_flux(self, U, Q):
        """
        Diffusive flux G

        Equation 2, page 5

        Literature:
        [1] - Vila-Pérez, J., Giacomini, M., Sevilla, R. et al.
              Hybridisable Discontinuous Galerkin Formulation of Compressible Flows.
              Arch Computat Methods Eng 28, 753–784 (2021).
              https://doi.org/10.1007/s11831-020-09508-z
        """
        dim = self.mesh.dim

        continuity = tuple(0 for i in range(dim))
        u = self.velocity(U)
        tau = self.deviatoric_stress_tensor(U, Q)
        heat_flux = self.heat_flux(U, Q)

        flux = CF((continuity, tau, tau*u - heat_flux), dims=(dim + 2, dim))

        return flux

    def diffusive_stabilisation_matrix(self, Uhat):

        Re = self.cfg.Reynolds_number
        Pr = self.cfg.Prandtl_number
        mu = self.dynamic_viscosity(Uhat)

        if self.mesh.dim == 2:

            tau_d = CF((0, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1/Pr), dims=(4, 4))

        elif self.mesh.dim == 3:

            tau_d = CF((0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 0, 1, 0, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 0, 1/Pr), dims=(5, 5))

        tau_d *= mu / Re
        if self.cfg.scaling is self.cfg.scaling.ACOUSTIC:
            tau_d *= self.cfg.Mach_number
        elif self.cfg.scaling is self.cfg.scaling.AEROACOUSTIC:
            tau_d *= self.cfg.Mach_number/(1 + self.cfg.Mach_number)

        return tau_d

    def heat_flux(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        Re = self.cfg.Reynolds_number
        Pr = self.cfg.Prandtl_number

        gradient_T = self.temperature_gradient(U, Q)
        mu = self.dynamic_viscosity(U)

        k = mu / (Re * Pr)

        if self.cfg.scaling is self.cfg.scaling.ACOUSTIC:
            k *= self.cfg.Mach_number
        elif self.cfg.scaling is self.cfg.scaling.AEROACOUSTIC:
            k *= self.cfg.Mach_number/(1 + self.cfg.Mach_number)

        return -k * gradient_T

    def deviatoric_stress_tensor(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        Re = self.cfg.Reynolds_number

        param = self.dynamic_viscosity(U)/Re
        if self.cfg.scaling is self.cfg.scaling.ACOUSTIC:
            param *= self.cfg.Mach_number
        elif self.cfg.scaling is self.cfg.scaling.AEROACOUSTIC:
            param *= self.cfg.Mach_number/(1 + self.cfg.Mach_number)

        return param * self.deviatoric_strain_tensor(U, Q)

    def dynamic_viscosity(self, U: Optional[CF] = None) -> CF:
        mu = self.cfg.dynamic_viscosity

        if isinstance(mu, Inviscid):
            raise TypeError('Dynamic Viscosity non existent for Inviscid flow')
        elif isinstance(mu, Constant):
            return 1
        elif isinstance(mu, Sutherland):
            M = self.cfg.Mach_number
            gamma = self.cfg.heat_capacity_ratio

            T_ = self.temperature(U)

            T_ref = mu.temperature_ref
            S0 = mu.temperature_0

            S_ = S0/(T_ref * (gamma - 1) * M**2)
            T_ref_ = 1/((gamma - 1) * M**2)

            return (T_/T_ref_)**(3/2) * (T_ref_ + S_)/(T_ + S_)
        else:
            raise NotImplementedError()

    def dynamic_viscosity_gradient(self, U: Optional[CF] = None, Q: Optional[CF] = None):
        mu = self.cfg.dynamic_viscosity

        if isinstance(mu, Inviscid):
            raise TypeError('Dynamic Viscosity non existent for Inviscid flow')
        elif isinstance(mu, Constant):
            return CF([0]*self.mesh.dim)
        else:
            raise NotImplementedError()

    def characteristic_variables(self, U, Q, Uhat, unit_vector: CF) -> tuple:
        """
        The charachteristic amplitudes are defined as

            Amplitudes = Lambda * L_inverse * dV/dn,

        where Lambda is the eigenvalue matrix, L_inverse is the mapping from
        primitive variables to charachteristic variables and dV/dn is the
        derivative normal to the boundary.
        """
        rho = self.density(Uhat)
        c = self.speed_of_sound(Uhat)

        gradient_rho_dir = InnerProduct(self.density_gradient(U, Q), unit_vector)
        gradient_p_dir = InnerProduct(self.pressure_gradient(U, Q), unit_vector)
        gradient_u_dir = self.velocity_gradient(U, Q) * unit_vector

        if self.mesh.dim == 2:

            variables = (
                gradient_p_dir - InnerProduct(gradient_u_dir, unit_vector) * c * rho,
                gradient_rho_dir * c**2 - gradient_p_dir,
                gradient_u_dir[0] * unit_vector[1] - gradient_u_dir[1] * unit_vector[0],
                gradient_p_dir + InnerProduct(gradient_u_dir, unit_vector) * c * rho
            )

        else:
            raise NotImplementedError()

        return variables

    def characteristic_velocities(self, U, unit_vector: CF, type: str = None, as_matrix: bool = False) -> CF:
        """
        The Lambda matrix contains the eigenvalues of the Jacobian matrices

        Equation E16.5.21, page 180

        Literature:
        [1] - C. Hirsch,
              Numerical Computation of Internal and External Flows Volume 2:
              Computational Methods for Inviscid and Viscous Flows,
              Vrije Universiteit Brussel, Brussels, Belgium
              ISBN: 978-0-471-92452-4
        """
        c = self.speed_of_sound(U)
        u_dir = InnerProduct(self.velocity(U), unit_vector)

        lam_m_c = u_dir - c
        lam = u_dir
        lam_p_c = u_dir + c

        if type is None:
            pass
        elif type == "absolute":
            lam_m_c = IfPos(lam_m_c, lam_m_c, -lam_m_c)
            lam = IfPos(lam, lam, -lam)
            lam_p_c = IfPos(lam_p_c, lam_p_c, -lam_p_c)
        elif type == "in":
            lam_m_c = IfPos(lam_m_c, 0, lam_m_c)
            lam = IfPos(lam, 0, lam)
            lam_p_c = IfPos(lam_p_c, 0, lam_p_c)
        elif type == "out":
            lam_m_c = IfPos(lam_m_c, lam_m_c, 0)
            lam = IfPos(lam, lam, 0)
            lam_p_c = IfPos(lam_p_c, lam_p_c, 0)
        else:
            raise ValueError(f"{str(type).capitalize()} invalid! Alternatives: {[None, 'absolute', 'in', 'out']}")

        if self.mesh.dim == 2:
            if as_matrix:
                Lambda = CF((lam_m_c, 0, 0, 0,
                             0, lam, 0, 0,
                             0, 0, lam, 0,
                             0, 0, 0, lam_p_c),
                            dims=(4, 4))
            else:
                Lambda = (lam_m_c, lam, lam, lam_p_c)
        elif self.mesh.dim == 3:
            if as_matrix:
                Lambda = CF((lam_m_c, 0, 0, 0, 0,
                             0, lam, 0, 0, 0,
                             0, 0, lam, 0, 0,
                             0, 0, 0, lam, 0,
                             0, 0, 0, 0, lam_p_c),
                            dims=(5, 5))
            else:
                Lambda = (lam_m_c, lam, lam, lam, lam_p_c)

        return Lambda

    def characteristic_amplitudes(self, U, Q, Uhat, unit_vector: CF, type: str = None):
        velocities = self.characteristic_velocities(Uhat, unit_vector, type, as_matrix=False)
        variables = self.characteristic_variables(U, Q, Uhat, unit_vector)
        return CF(tuple(vel * var for vel, var in zip(velocities, variables)))

    def identity_matrix_outgoing(self, U, unit_vector: CF) -> CF:
        c = self.speed_of_sound(U)
        u_dir = InnerProduct(self.velocity(U), unit_vector)

        lam_m_c = IfPos(u_dir - c, 1, 0)
        lam = IfPos(u_dir, 1, 0)
        lam_p_c = IfPos(u_dir + c, 1, 0)

        if self.mesh.dim == 2:

            identity = CF((lam_m_c, 0, 0, 0,
                           0, lam, 0, 0,
                           0, 0, lam, 0,
                           0, 0, 0, lam_p_c), dims=(4, 4))

        elif self.mesh.dim == 3:

            identity = CF((lam_m_c, 0, 0, 0, 0,
                           0, lam, 0, 0, 0,
                           0, 0, lam, 0, 0,
                           0, 0, 0, lam, 0,
                           0, 0, 0, 0, lam_p_c), dims=(5, 5))

        return identity

    def identity_matrix_incoming(self, U, unit_vector: CF) -> CF:
        c = self.speed_of_sound(U)
        u_dir = InnerProduct(self.velocity(U), unit_vector)

        lam_m_c = IfPos(u_dir - c, 0, 1)
        lam = IfPos(u_dir, 0, 1)
        lam_p_c = IfPos(u_dir + c, 0, 1)

        if self.mesh.dim == 2:

            identity = CF((lam_m_c, 0, 0, 0,
                           0, lam, 0, 0,
                           0, 0, lam, 0,
                           0, 0, 0, lam_p_c), dims=(4, 4))

        elif self.mesh.dim == 3:

            identity = CF((lam_m_c, 0, 0, 0, 0,
                           0, lam, 0, 0, 0,
                           0, 0, lam, 0, 0,
                           0, 0, 0, lam, 0,
                           0, 0, 0, 0, lam_p_c), dims=(5, 5))

        return identity

    def DVP_from_DME(self, U) -> CF:
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
        gamma = self.cfg.heat_capacity_ratio

        rho = self.density(U)
        u = self.velocity(U)

        if self.mesh.dim == 2:

            ux, uy = u[0], u[1]

            Minv = CF((1, 0, 0, 0,
                       -ux/rho, 1/rho, 0, 0,
                       -uy/rho, 0, 1/rho, 0,
                       (gamma - 1)/2 * InnerProduct(u, u), -(gamma - 1) * ux,
                       -(gamma - 1) * uy, gamma - 1), dims=(4, 4))
        else:
            raise NotImplementedError()

        return Minv

    def DVP_from_CHAR(self, U, unit_vector: CF) -> CF:
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
        rho = self.density(U)
        c = self.speed_of_sound(U)

        if self.mesh.dim == 2:

            d0, d1 = unit_vector[0], unit_vector[1]

            L = CF((0.5/c**2, 1/c**2, 0, 0.5/c**2,
                    -d0/(2*c*rho), 0, d1, d0/(2*c*rho),
                    -d1/(2*c*rho), 0, -d0, d1/(2*c*rho),
                    0.5, 0, 0, 0.5), dims=(4, 4))
        else:
            return NotImplementedError()

        return L

    def DVP_from_PVT(self, U):
        """ From low mach primitive to compressible primitive """
        gamma = self.cfg.heat_capacity_ratio
        R = (gamma - 1)/gamma

        p = self.pressure(U)
        T = self.temperature(U)

        if self.mesh.dim == 2:

            T_mat = CF((
                1/(R*T), 0, 0, -p/(R * T**2),
                0, 1, 0, 0,
                0, 0, 1, 0,
                1, 0, 0, 0
            ), dims=(4, 4))

        else:
            raise NotImplementedError()

        return T_mat

    def DVP_from_PVT_matrix(self, matrix, U) -> CF:
        return self.DVP_from_PVT(U) * matrix * self.PVT_from_DVP(U)

    def DVP_from_DME_matrix(self, matrix, U) -> CF:
        return self.DVP_from_DME(U) * matrix * self.DME_from_DVP(U)

    def DVP_from_CHAR_matrix(self, matrix, U) -> CF:
        return self.DVP_from_CHAR(U) * matrix * self.CHAR_from_DVP(U)

    def DVP_convective_jacobian_x(self, U) -> CF:

        rho = self.density(U)
        c = self.speed_of_sound(U)

        if self.mesh.dim == 2:
            u = self.velocity(U)[0]

            A = CF((
                u, rho, 0, 0,
                0, u, 0, 1/rho,
                0, 0, u, 0,
                0, rho*c**2, 0, u),
                dims=(4, 4))
        else:
            return NotImplementedError()

        return A

    def DVP_convective_jacobian_y(self, U) -> CF:

        rho = self.density(U)
        c = self.speed_of_sound(U)

        if self.mesh.dim == 2:
            v = self.velocity(U)[1]

            B = CF((v, 0, rho, 0,
                    0, v, 0, 0,
                    0, 0, v, 1/rho,
                    0, 0, rho*c**2, v),
                   dims=(4, 4))

        else:
            return NotImplementedError()

        return B

    def DVP_convective_jacobian(self, U, unit_vector: CF) -> CF:
        rho = self.density(U)
        c = self.speed_of_sound(U)
        un = InnerProduct(self.velocity(U), unit_vector)

        if self.mesh.dim == 2:
            d0, d1 = unit_vector

            JAC = CF((
                un, d0*rho, d1*rho, 0,
                0, un, 0, d0/rho,
                0, 0, un, d1/rho,
                0, d0*rho*c**2, d1*rho*c**2, un),
                dims=(4, 4))
        else:
            return NotImplementedError()

        return JAC

    def DME_from_DVP(self, U) -> CF:
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
        gamma = self.cfg.heat_capacity_ratio

        rho = self.density(U)
        u = self.velocity(U)
        ux, uy = u[0], u[1]

        if self.mesh.dim == 2:
            M = CF((1, 0, 0, 0,
                    ux, rho, 0, 0,
                    uy, 0, rho, 0,
                    0.5*InnerProduct(u, u), rho*ux, rho*uy, 1/(gamma - 1)), dims=(4, 4))
        else:
            raise NotImplementedError()

        return M

    def DME_from_CHAR(self, U, unit_vector: CF) -> CF:
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
        gamma = self.cfg.heat_capacity_ratio
        rho = self.density(U)
        c = self.speed_of_sound(U)

        if self.mesh.dim == 2:
            d0, d1 = unit_vector
            u = self.velocity(U)
            ux, uy = u[0], u[1]

            T = CF(
                (1 / (2 * c ** 2),
                 1 / c ** 2, 0, 1 / (2 * c ** 2),
                 -d0 / (2 * c) + ux / (2 * c ** 2),
                 ux / c ** 2, d1 * rho, d0 / (2 * c) + ux / (2 * c ** 2),
                 -d1 / (2 * c) + uy / (2 * c ** 2),
                 uy / c ** 2, -d0 * rho, d1 / (2 * c) + uy / (2 * c ** 2),
                 0.5 / (gamma - 1) - d0 * ux / (2 * c) - d1 * uy / (2 * c) + InnerProduct(u, u) / (4 * c ** 2),
                 InnerProduct(u, u) / (2 * c ** 2),
                 -d0 * rho * uy + d1 * rho * ux, 0.5 / (gamma - 1) + d0 * ux / (2 * c) + d1 * uy / (2 * c) +
                 InnerProduct(u, u) / (4 * c ** 2)), dims=(4, 4))
        else:
            raise NotImplementedError()
        return T

    def DME_from_PVT(self, U):

        gamma = self.cfg.heat_capacity_ratio
        R = (gamma - 1)/gamma

        p = self.pressure(U)
        T = self.temperature(U)
        E = self.specific_energy(U)
        Ek = self.specific_kinetic_energy(U)

        if self.mesh.dim == 2:
            u, v = self.velocity(U)
            M = CF((
                1, 0, 0, -p/T,
                u, p, 0, -p*u/T,
                v, 0, p, -p*v/T,
                E, p*u, p*v, -p*Ek/T
            ), dims=(4, 4))

            M *= 1/(R * T)
        else:
            raise NotImplementedError()

        return M

    def DME_from_DVP_matrix(self, matrix, U) -> CF:
        return self.DME_from_DVP(U) * matrix * self.DVP_from_DME(U)

    def DME_from_CHAR_matrix(self, matrix, U, unit_vector) -> CF:
        return self.DME_from_CHAR(U, unit_vector) * matrix * self.CHAR_from_DME(U, unit_vector)

    def DME_from_PVT_matrix(self, matrix, U) -> CF:
        return self.DME_from_PVT(U) * matrix * self.CHAR_from_PVT(U)

    def DME_convective_jacobian_x(self, U) -> CF:
        '''
        First Jacobian of the convective Euler Fluxes F_c = (f_c, g_c) for conservative variables U
        A = \partial f_c / \partial U
        input: u = (rho, rho * u, rho * E)
        See also Page 144 in C. Hirsch, Numerical Computation of Internal and External Flows: Vol.2
        '''

        gamma = self.cfg.heat_capacity_ratio
        velocity = self.velocity(U)
        ux, uy = velocity[0], velocity[1]
        u = InnerProduct(velocity, velocity)
        E = self.specific_energy(U)

        A = CF((
            0, 1, 0, 0,
            (gamma - 3)/2 * ux**2 + (gamma - 1)/2 * uy**2, (3 - gamma) * ux, -(gamma - 1) * uy, gamma - 1,
            -ux*uy, uy, ux, 0,
            -gamma*ux*E + (gamma - 1)*ux*u, gamma*E - (gamma - 1)/2 * (uy**2 + 3*ux**2), -(gamma - 1)*ux*uy, gamma*ux),
            dims=(4, 4))

        return A

    def DME_convective_jacobian_y(self, U) -> CF:
        '''
        Second Jacobian of the convective Euler Fluxes F_c = (f_c, g_c)for conservative variables U
        B = \partial g_c / \partial U
        input: u = (rho, rho * u, rho * E)
        See also Page 144 in C. Hirsch, Numerical Computation of Internal and External Flows: Vol.2
        '''
        gamma = self.cfg.heat_capacity_ratio
        velocity = self.velocity(U)
        ux, uy = velocity[0], velocity[1]
        u = InnerProduct(velocity, velocity)
        E = self.specific_energy(U)

        B = CF(
            (0, 0, 1, 0, -ux * uy, uy, ux, 0, (gamma - 3) / 2 * uy ** 2 + (gamma - 1) / 2 * ux ** 2, -(gamma - 1) * ux,
             (3 - gamma) * uy, gamma - 1, -gamma * uy * E + (gamma - 1) * uy * u, -(gamma - 1) * ux * uy, gamma * E -
             (gamma - 1) / 2 * (ux ** 2 + 3 * uy ** 2),
             gamma * uy),
            dims=(4, 4))

        return B

    def DME_convective_jacobian(self, U, unit_vector: CF) -> CF:
        A = self.DME_convective_jacobian_x(U)
        B = self.DME_convective_jacobian_y(U)
        return A * unit_vector[0] + B * unit_vector[1]

    def CHAR_from_DVP(self, U, unit_vector: CF) -> CF:
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
        rho = self.density(U)
        c = self.speed_of_sound(U)

        if self.mesh.dim == 2:

            d0, d1 = unit_vector[0], unit_vector[1]

            Linv = CF((0, -rho*c*d0, -rho*c*d1, 1,
                       c**2, 0, 0, -1,
                       0, d1, -d0, 0,
                       0, rho*c*d0, rho*c*d1, 1), dims=(4, 4))

        else:
            return NotImplementedError()

        return Linv

    def CHAR_from_DME(self, U, unit_vector: CF) -> CF:
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
        gamma = self.cfg.heat_capacity_ratio
        rho = self.density(U)
        c = self.speed_of_sound(U)

        if self.mesh.dim == 2:
            d0, d1 = unit_vector
            u = self.velocity(U)
            ux, uy = u[0], u[1]

            T = CF((
                c*d0*ux + c*d1*uy + (gamma - 1)*InnerProduct(u, u)/2, -c*d0 + ux*(1 - gamma), -c*d1 + uy*(1 - gamma), gamma - 1,
                c**2 - (gamma - 1)*InnerProduct(u, u)/2, -ux*(1 - gamma), -uy*(1 - gamma), 1 - gamma,
                d0*uy/rho - d1*ux/rho, d1/rho, -d0/rho, 0,
                -c*d0*ux - c*d1*uy + (gamma - 1)*InnerProduct(u, u)/2, c*d0 + ux*(1 - gamma), c*d1 + uy*(1 - gamma), gamma - 1),
                dims=(4, 4))
        else:
            raise NotImplementedError()
        return T

    def CHAR_from_PVT(self, U, unit_vector: CF) -> CF:
        return self.CHAR_from_DVP(U, unit_vector) * self.DVP_from_PVT(U)

    def CHAR_from_DVP_matrix(self, matrix, U, unit_vector) -> CF:
        return self.CHAR_from_DVP(U, unit_vector) * matrix * self.DVP_from_CHAR(U, unit_vector)

    def CHAR_from_DME_matrix(self, matrix, U, unit_vector) -> CF:
        return self.CHAR_from_DME(U, unit_vector) * matrix * self.DME_from_CHAR(U, unit_vector)

    def CHAR_from_PVT_matrix(self, matrix, U, unit_vector) -> CF:
        return self.CHAR_from_PVT(U, unit_vector) * matrix * self.PVT_from_CHAR(U, unit_vector)

    def PVT_from_DVP(self, U):
        """ From compressible primitive to low mach primitive """
        gamma = self.cfg.heat_capacity_ratio
        R = (gamma - 1)/gamma

        p = self.pressure(U)
        rho = self.density(U)

        if self.mesh.dim == 2:

            Tinv = CF((
                0, 0, 0, 1,
                0, 1, 0, 0,
                0, 0, 1, 0,
                -p/(R*rho**2), 0, 0, 1/(R * rho),
            ), dims=(4, 4))

        else:
            raise NotImplementedError()

        return Tinv

    def PVT_from_DME(self, U):

        gamma = self.cfg.heat_capacity_ratio
        R = (gamma - 1)/gamma

        rho = self.density(U)
        T = self.temperature(U)
        Ek = self.specific_kinetic_energy(U)

        if self.mesh.dim == 2:
            u, v = self.velocity(U)

            Minv = CF((
                R*Ek*rho, -R*rho*u, -R*rho*v, R*rho,
                -u/gamma, 1/gamma, 0, 0,
                -v/gamma, 0, 1/gamma, 0,
                -T/gamma + Ek, -u, -v, 1
            ), dims=(4, 4))

            Minv *= gamma/rho

        else:
            raise NotImplementedError()

        return Minv

    def PVT_from_CHAR(self, U, unit_vector: CF) -> CF:
        return self.PVT_from_DVP(U) * self.DVP_from_CHAR(U, unit_vector)

    def PVT_from_DVP_matrix(self, matrix, U) -> CF:
        return self.PVT_from_DVP(U) * matrix * self.DVP_from_PVT(U)

    def PVT_from_DME_matrix(self, matrix, U) -> CF:
        return self.PVT_from_DME(U) * matrix * self.DME_from_PVT(U)

    def PVT_from_CHAR_matrix(self, matrix, U, unit_vector) -> CF:
        return self.PVT_from_CHAR(U, unit_vector) * matrix * self.CHAR_from_PVT(U, unit_vector)

    def _if_none_replace_with_gfu(self, cf, component: int = 0):
        if cf is None:
            cf = self.gfu.components[component]
        return cf
