from __future__ import annotations
import abc
import enum
import dataclasses
from typing import Optional, NamedTuple, TYPE_CHECKING

from ngsolve import *

from ..time_schemes import time_scheme_factory
from .. import boundary_conditions as bc
from .. import viscosity as mu

if TYPE_CHECKING:
    from configuration import SolverConfiguration


class CompressibleFormulations(enum.Enum):
    PRIMITIVE = "primitive"
    CONSERVATIVE = "conservative"


class MixedMethods(enum.Enum):
    NONE = None
    GRADIENT = "gradient"
    STRAIN_HEAT = "strain_heat"


@dataclasses.dataclass
class VectorCoordinates:
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
class TensorCoordinates:
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


class Indices(NamedTuple):
    DENSITY: Optional[int] = None
    MOMENTUM: Optional[VectorCoordinates] = None
    VELOCITY: Optional[VectorCoordinates] = None
    PRESSURE: Optional[int] = None
    ENERGY: Optional[int] = None
    TEMPERATURE_GRADIENT: Optional[VectorCoordinates] = None
    STRAIN: Optional[TensorCoordinates] = None


class Formulation(abc.ABC):

    _indices: Indices

    def __init__(self, mesh: Mesh, solver_configuration: SolverConfiguration) -> None:

        self.mesh = mesh
        self._solver_configuration = solver_configuration

        self.bcs = bc.BoundaryConditions(mesh.GetBoundaries(), solver_configuration)
        self.ic = bc.InitialCondition(mesh.GetMaterials(), solver_configuration)
        self.mu = mu.viscosity_factory(self)
        self.time_scheme = time_scheme_factory(solver_configuration)

        self._fes = self.get_FESpace()
        self._TnT = self.get_TnT()

        self.normal = specialcf.normal(mesh.dim)
        self.tangential = specialcf.tangential(mesh.dim)
        self.meshsize = specialcf.mesh_size

    @property
    def solver_configuration(self) -> SolverConfiguration:
        return self._solver_configuration

    @property
    def fes(self) -> ProductSpace:
        return self._fes

    @property
    def TnT(self):
        return self._TnT

    def add_bcs_bilinearform(self, blf, old_components):

        for boundary, value in self.bcs.boundaries.items():

            if value is None:
                raise ValueError(f"Boundary condition for {boundary} has not been set!")

            elif isinstance(value, bc.Dirichlet):
                self._add_dirichlet_bilinearform(blf, boundary, value)

            elif isinstance(value, bc.FarField):
                self._add_farfield_bilinearform(blf, boundary, value)

            elif isinstance(value, bc.Outflow):
                self._add_outflow_bilinearform(blf, boundary, value)

            elif isinstance(value, bc.NonReflectingOutflow):
                self._add_nonreflecting_outflow_bilinearform(blf, boundary, value, old_components)

            elif isinstance(value, bc.InviscidWall):
                self._add_inviscid_wall_bilinearform(blf, boundary, value)

            elif isinstance(value, bc.IsothermalWall):
                self._add_isothermal_wall_bilinearform(blf, boundary, value)

            elif isinstance(value, bc.AdiabaticWall):
                self._add_adiabatic_wall_bilinearform(blf, boundary, value)

    def add_ic_linearform(self, lf):

        for domain, value in self.ic.domains.items():

            if value is None:
                raise ValueError(f"Initial condition for {domain} has not been set!")

            elif isinstance(value, bc.Initial):
                self._add_initial_linearform(lf, domain, value)

    @abc.abstractmethod
    def _add_dirichlet_bilinearform(self, blf, boundary, value): ...

    @abc.abstractmethod
    def _add_farfield_bilinearform(self, blf, boundary, value): ...

    @abc.abstractmethod
    def _add_outflow_bilinearform(self, blf, boundary, value): ...

    @abc.abstractmethod
    def _add_nonreflecting_outflow_bilinearform(self, blf, boundary, value): ...

    @abc.abstractmethod
    def _add_inviscid_wall_bilinearform(self, blf, boundary, value): ...

    @abc.abstractmethod
    def _add_isothermal_wall_bilinearform(self, blf, boundary, value): ...

    @abc.abstractmethod
    def _add_adiabatic_wall_bilinearform(self, blf, boundary, value): ...

    @abc.abstractmethod
    def get_FESpace(self) -> ProductSpace: ...

    @abc.abstractmethod
    def get_TnT(self): ...

    @abc.abstractmethod
    def _add_initial_linearform(self, lf, domain, value): ...

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
    def density_gradient(self, U, Q): ...

    @abc.abstractmethod
    def momentum_gradient(self, U, Q): ...

    @abc.abstractmethod
    def energy_gradient(self, U, Q): ...

    @abc.abstractmethod
    def enthalpy_gradient(self, U, Q): ...

    @abc.abstractmethod
    def pressure_gradient(self, U, Q): ...

    @abc.abstractmethod
    def temperature_gradient(self, U, Q): ...

    @abc.abstractmethod
    def velocity_gradient(self, U, Q): ...

    @abc.abstractmethod
    def vorticity(self, U, Q): ...

    @abc.abstractmethod
    def deviatoric_strain_tensor(self, U, Q): ...

    @abc.abstractmethod
    def deviatoric_stress_tensor(self, U, Q): ...

    def __str__(self) -> str:
        return self.__class__.__name__
