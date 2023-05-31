from __future__ import annotations
import abc
import enum
import dataclasses
from typing import Optional, TYPE_CHECKING

from ngsolve import *

from ..time_schemes import time_scheme_factory, TimeLevelsGridfunction
from .. import conditions as co
from .. import viscosity as mu

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
class VectorIndices:
    X: Optional[int] = None
    Y: Optional[int] = None
    Z: Optional[int] = None

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
        self.mu = mu.viscosity_factory(self)
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

    def initialize(self):
        self._fes = self._initialize_FE_space()
        self._TnT = self._initialize_TnT()
        self._gfus = TimeLevelsGridfunction({level: GridFunction(self.fes) for level in self.time_scheme.time_levels})

    def add_bcs_bilinearform(self, blf):

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

    def add_perturbation_linearform(self, lf, perturbation: co.Perturbation):
        for domain, _ in self.dcs.items():
            self._add_linearform(lf, domain, perturbation)

    def add_initial_linearform(self, lf):
        for domain, condition in self.dcs.items():

            if isinstance(condition.initial, co.Initial):
                self._add_linearform(lf, domain, condition.initial)
            else:
                logger.warn(f"Initial condition for '{domain}' has not been set!")

    def add_dcs_bilinearform(self, blf):

        for domain, condition in self.dcs.items():
            if isinstance(condition.sponge, co.SpongeLayer):
                self._add_sponge_bilinearform(blf, domain, condition.sponge)

    def update_gridfunctions(self, initial_value: bool = False):
        if initial_value:
            self.time_scheme.update_initial_solution(self._gfus)
        else:
            self.time_scheme.update_previous_solution(self._gfus)

    @abc.abstractmethod
    def _add_dirichlet_bilinearform(self, blf, boundary, condition): ...

    @abc.abstractmethod
    def _add_farfield_bilinearform(self, blf, boundary, condition): ...

    @abc.abstractmethod
    def _add_outflow_bilinearform(self, blf, boundary, condition): ...

    @abc.abstractmethod
    def _add_nonreflecting_outflow_bilinearform(self, blf, boundary, condition): ...

    @abc.abstractmethod
    def _add_nonreflecting_inflow_bilinearform(self, blf, boundary, condition): ...

    @abc.abstractmethod
    def _add_inviscid_wall_bilinearform(self, blf, boundary, condition): ...

    @abc.abstractmethod
    def _add_isothermal_wall_bilinearform(self, blf, boundary, condition): ...

    @abc.abstractmethod
    def _add_adiabatic_wall_bilinearform(self, blf, boundary, condition): ...

    @abc.abstractmethod
    def _add_linearform(self, lf, domain, condition): ...

    @abc.abstractmethod
    def _add_sponge_bilinearform(self, blf, domain, condition): ...

    @abc.abstractmethod
    def _initialize_FE_space(self) -> ProductSpace: ...

    @abc.abstractmethod
    def _initialize_TnT(self) -> TestAndTrialFunction: ...

    @abc.abstractmethod
    def add_mass_bilinearform(self, blf): ...

    @abc.abstractmethod
    def add_time_bilinearform(self, blf): ...

    @abc.abstractmethod
    def add_convective_bilinearform(self, blf): ...

    @abc.abstractmethod
    def add_diffusive_bilinearform(self, blf): ...

    @abc.abstractmethod
    def density(self, U): ...

    @abc.abstractmethod
    def momentum(self, U): ...

    @abc.abstractmethod
    def energy(self, U): ...

    @abc.abstractmethod
    def pressure(self, U): ...

    @abc.abstractmethod
    def velocity(self, U): ...

    @abc.abstractmethod
    def temperature(self, U): ...

    @abc.abstractmethod
    def enthalpy(self, U): ...

    @abc.abstractmethod
    def specific_enthalpy(self, U): ...

    @abc.abstractmethod
    def specific_energy(self, U): ...

    @abc.abstractmethod
    def specific_inner_energy(self, U): ...

    @abc.abstractmethod
    def speed_of_sound(self, U): ...

    @abc.abstractmethod
    def mach_number(self, U): ...

    @abc.abstractmethod
    def density_gradient(self, U, Q=None): ...

    @abc.abstractmethod
    def momentum_gradient(self, U, Q=None): ...

    @abc.abstractmethod
    def energy_gradient(self, U, Q=None): ...

    @abc.abstractmethod
    def enthalpy_gradient(self, U, Q=None): ...

    @abc.abstractmethod
    def pressure_gradient(self, U, Q=None): ...

    @abc.abstractmethod
    def temperature_gradient(self, U, Q=None): ...

    @abc.abstractmethod
    def velocity_gradient(self, U, Q=None): ...

    @abc.abstractmethod
    def vorticity(self, U, Q=None): ...

    @abc.abstractmethod
    def heat_flux(self, U, Q=None): ...

    @abc.abstractmethod
    def deviatoric_strain_tensor(self, U, Q=None): ...

    @abc.abstractmethod
    def deviatoric_stress_tensor(self, U, Q=None): ...

    def __str__(self) -> str:
        return self.__class__.__name__
