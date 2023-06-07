from __future__ import annotations
import abc
import enum
import dataclasses
from typing import Optional, TYPE_CHECKING

from ngsolve import *

from ..time_schemes import time_scheme_factory, TimeLevelsGridfunction
from .. import conditions as co
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


@dataclasses.dataclass
class TestAndTrialFunction:
    PRIMAL: tuple[Optional[ProxyFunction], Optional[ProxyFunction]] = (None, None)
    PRIMAL_FACET: tuple[Optional[ProxyFunction], Optional[ProxyFunction]] = (None, None)
    MIXED: tuple[Optional[ProxyFunction], Optional[ProxyFunction]] = (None, None)


class Formulation(abc.ABC):

    def __init__(self, mesh: Mesh, solver_configuration: SolverConfiguration) -> None:

        self._mesh = mesh
        self._cfg = solver_configuration

        self.bcs = co.BoundaryConditions(mesh.GetBoundaries(), solver_configuration)
        self.dcs = co.DomainConditions(mesh.GetMaterials(), solver_configuration)
        self.time_scheme = time_scheme_factory(solver_configuration.time)

        self._gfus = None
        self._fes = None
        self._TnT = None

        self.normal = specialcf.normal(mesh.dim)
        self.tangential = specialcf.tangential(mesh.dim)
        self.meshsize = specialcf.mesh_size

    @property
    def mesh(self) -> Mesh:
        return self._mesh

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
        return self._fes

    @property
    def TnT(self) -> TestAndTrialFunction:
        if self._fes is None:
            raise RuntimeError("Call 'formulation.initialize()' before accessing the TestAndTrialFunctions")
        return self._TnT

    @abc.abstractmethod
    def _initialize_FE_space(self) -> ProductSpace: ...

    @abc.abstractmethod
    def _initialize_TnT(self) -> TestAndTrialFunction: ...

    @abc.abstractmethod
    def add_time_bilinearform(self, blf) -> None: ...

    @abc.abstractmethod
    def add_convective_bilinearform(self, blf) -> None: ...

    @abc.abstractmethod
    def add_diffusive_bilinearform(self, blf) -> None: ...

    def initialize(self):
        self._fes = self._initialize_FE_space()
        self._TnT = self._initialize_TnT()
        self._gfus = TimeLevelsGridfunction({level: GridFunction(self.fes) for level in self.time_scheme.time_levels})

    def update_gridfunctions(self, initial_value: bool = False):
        if initial_value:
            self.time_scheme.update_initial_solution(self._gfus)
        else:
            self.time_scheme.update_previous_solution(self._gfus)

    def add_boundary_conditions_bilinearform(self, blf):

        for boundary, condition in self.bcs.items():

            if not isinstance(condition, co.Condition):
                logger.warn(f"Boundary condition for '{boundary}' has not been set!")

            elif isinstance(condition, co.Dirichlet):
                self._add_dirichlet_bilinearform(blf, boundary, condition)

            elif isinstance(condition, co.FarField):
                self._add_farfield_bilinearform(blf, boundary, condition)

            elif isinstance(condition, co.Outflow):
                self._add_outflow_bilinearform(blf, boundary, condition)

            elif isinstance(condition, co.NRBC_Outflow):
                self._add_nonreflecting_outflow_bilinearform(blf, boundary, condition)

            elif isinstance(condition, co.NRBC_Inflow):
                self._add_nonreflecting_inflow_bilinearform(blf, boundary, condition)

            elif isinstance(condition, co.InviscidWall):
                self._add_inviscid_wall_bilinearform(blf, boundary, condition)

            elif isinstance(condition, co.IsothermalWall):
                self._add_isothermal_wall_bilinearform(blf, boundary, condition)

            elif isinstance(condition, co.AdiabaticWall):
                self._add_adiabatic_wall_bilinearform(blf, boundary, condition)

    def add_domain_conditions_bilinearform(self, blf):

        for domain, condition in self.dcs.items():
            if isinstance(condition.sponge, co.SpongeLayer):
                self._add_sponge_bilinearform(blf, domain, condition.sponge)

    def add_mass_bilinearform(self, blf):
        mixed_method = self.cfg.mixed_method

        U, V = self.TnT.PRIMAL
        Uhat, Vhat = self.TnT.PRIMAL_FACET

        blf += U * V * dx
        blf += Uhat * Vhat * dx(element_boundary=True)

        if mixed_method is not MixedMethods.NONE:
            Q, P = self.TnT.MIXED
            blf += Q * P * dx

    def add_perturbation_linearform(self, lf, perturbation: co.Perturbation):
        for domain, _ in self.dcs.items():
            self._add_linearform(lf, domain, perturbation)

    def add_initial_linearform(self, lf):
        for domain, condition in self.dcs.items():

            if isinstance(condition.initial, co.Initial):
                self._add_linearform(lf, domain, condition.initial)
            else:
                logger.warn(f"Initial condition for '{domain}' has not been set!")

    def add_forcing_linearform(self, lf, force: Optional[CF]):
        bonus_int_order = self.cfg.bonus_int_order_vol
        _, V = self.TnT.PRIMAL

        if force is not None:
            lf += InnerProduct(force, V) * dx(bonus_intorder=bonus_int_order)

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

    def heat_flux(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        Re = self.cfg.Reynolds_number
        Pr = self.cfg.Prandtl_number

        gradient_T = self.temperature_gradient(U, Q)
        mu = self.dynamic_viscosity(U)

        k = mu / (Re * Pr)

        return -k * gradient_T

    def deviatoric_stress_tensor(self, U: Optional[CF] = None, Q: Optional[CF] = None):

        Re = self.cfg.Reynolds_number
        return self.dynamic_viscosity(U)/Re * self.deviatoric_strain_tensor(U, Q)

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

    def _if_none_replace_with_gfu(self, cf, component: int = 0):
        if cf is None:
            cf = self.gfu.components[component]
        return cf

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

    def _add_linearform(self, lf, domain, condition):
        raise NotImplementedError()

    def _add_sponge_bilinearform(self, blf, domain, condition):
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.__class__.__name__
