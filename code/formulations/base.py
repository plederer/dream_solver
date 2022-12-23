from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, NamedTuple
from configuration import SolverConfiguration

from . import MixedMethods
import boundary_conditions as bc
from time_schemes import time_scheme_factory
from ngsolve import *


class Coordinates(NamedTuple):
    X: Optional[int] = None
    Y: Optional[int] = None
    Z: Optional[int] = None

    def __iter__(self):
        for iter in (self.X, self.Y, self.Z):
            if iter is not None:
                yield iter


class Indices(NamedTuple):
    DENSITY: Optional[int] = None
    MOMENTUM: Optional[Coordinates] = None
    VELOCITY: Optional[Coordinates] = None
    PRESSURE: Optional[int] = None
    ENERGY: Optional[int] = None
    TEMPERATURE_GRADIENT: Optional[Coordinates] = None


class Formulation(ABC):

    _indices: Indices

    def __init__(self, mesh, solver_configuration: SolverConfiguration) -> None:
        self.mesh = mesh
        self.bcs = bc.BoundaryConditions(mesh.GetBoundaries(), solver_configuration)
        self.time_scheme = time_scheme_factory(solver_configuration)
        self.solver_configuration = solver_configuration

        self._fes = self.get_FESpace()
        self._TnT = self.get_TnT()

        self.normal = specialcf.normal(mesh.dim)
        self.tangential = specialcf.tangential(mesh.dim)
        self.meshsize = specialcf.mesh_size

    @property
    def fes(self) -> ProductSpace:
        return self._fes

    @property
    def TnT(self):
        return self._TnT

    def set_boundary_conditions_bilinearform(self, blf):

        for boundary, value in self.bcs.boundaries.items():

            if value is None:
                raise ValueError(f"Boundary condition for {boundary} has not been set!")

            elif isinstance(value, bc.Dirichlet):
                self._set_dirichlet(blf, boundary, value)

            elif isinstance(value, bc.FarField):
                self._set_farfield(blf, boundary, value)

            elif isinstance(value, bc.Outflow):
                self._set_outflow(blf, boundary, value)

            elif isinstance(value, bc.NonReflectingOutflow):
                self._set_nonreflecting_outflow(blf, boundary, value)

            elif isinstance(value, bc.InviscidWall):
                self._set_inviscid_wall(blf, boundary, value)

            elif isinstance(value, bc.IsothermalWall):
                self._set_isothermal_wall(blf, boundary, value)

            elif isinstance(value, bc.AdiabaticWall):
                self._set_adiabatic_wall(blf, boundary, value)

    @abstractmethod
    def _set_dirichlet(self, blf, boundary, value): ...

    @abstractmethod
    def _set_farfield(self, blf, boundary, value): ...

    @abstractmethod
    def _set_outflow(self, blf, boundary, value): ...

    @abstractmethod
    def _set_nonreflecting_outflow(self, blf, boundary, value): ...

    @abstractmethod
    def _set_inviscid_wall(self, blf, boundary, value): ...

    @abstractmethod
    def _set_isothermal_wall(self, blf, boundary, value): ...

    @abstractmethod
    def _set_adiabatic_wall(self, blf, boundary, value): ...

    @abstractmethod
    def get_FESpace(self) -> ProductSpace: ...

    @abstractmethod
    def get_TnT(self): ...

    @abstractmethod
    def set_time_bilinearform(self, blf): ...

    @abstractmethod
    def set_mixed_bilinearform(self, blf): ...

    @abstractmethod
    def set_convective_bilinearform(self, blf): ...

    @abstractmethod
    def set_diffusive_bilinearform(self, blf): ...

    @abstractmethod
    def density(self, U): ...

    @abstractmethod
    def momentum(self, U): ...

    @abstractmethod
    def energy(self, U): ...

    @abstractmethod
    def pressure(self, U): ...

    @abstractmethod
    def velocity(self, U): ...

    @abstractmethod
    def temperature(self, U): ...

    @abstractmethod
    def enthalpy(self, U): ...

    @abstractmethod
    def specific_energy(self, U): ...

    @abstractmethod
    def specific_inner_energy(self, U): ...

    @abstractmethod
    def speed_of_sound(self, U): ...

    @abstractmethod
    def density_gradient(self, U, Q): ...

    @abstractmethod
    def momentum_gradient(self, U, Q): ...

    @abstractmethod
    def energy_gradient(self, U, Q): ...

    @abstractmethod
    def pressure_gradient(self, U, Q): ...

    @abstractmethod
    def temperature_gradient(self, U, Q): ...

    @abstractmethod
    def velocity_gradient(self, U, Q): ...

    def __str__(self) -> str:
        return self.__class__.__name__


class ConservativeFormulation(Formulation):

    def density(self, U):
        return U[self._indices.DENSITY]

    def momentum(self, U):
        return CF(tuple(U[index] for index in self._indices.MOMENTUM))

    def energy(self, U):
        return U[self._indices.ENERGY]

    def pressure(self, U):

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)
        gamma = self.solver_configuration.heat_capacity_ratio

        return (gamma-1) * (rho_E - InnerProduct(rho_u, rho_u)/(2 * rho))

    def velocity(self, U):

        rho = self.density(U)
        rho_u = self.momentum(U)

        return rho_u/rho

    def temperature(self, U):

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)
        gamma = self.solver_configuration.heat_capacity_ratio

        return gamma/rho * (rho_E - InnerProduct(rho_u, rho_u)/(2 * rho))

    def enthalpy(self, U):

        p = self.pressure(U)
        rho = self.density(U)
        rho_E = self.energy(U)

        return rho_E + p/rho

    def specific_energy(self, U):

        rho = self.density(U)
        rho_E = self.energy(U)

        return rho_E/rho

    def specific_inner_energy(self, U):

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)

        return (rho_E - InnerProduct(rho_u, rho_u)/(2 * rho))/rho

    def speed_of_sound(self, U):

        p = self.pressure(U)
        rho = self.density(U)
        gamma = self.solver_configuration.heat_capacity_ratio

        return sqrt(gamma * p/rho)

    def mach_number(self, U):

        u = self.velocity(U)
        c = self.speed_of_sound(U)

        return sqrt(InnerProduct(u, u)) / c

    def density_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            gradient_rho = Q[self._indices.DENSITY]
        else:
            gradient_U = grad(U)
            gradient_rho = CF(tuple(gradient_U[self._indices.DENSITY, index] for index in range(self.mesh.dim)))

        return gradient_rho

    def momentum_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method
        dim = self.mesh.dim

        if mixed_method is MixedMethods.GRADIENT:
            gradient_rho_u = CF(tuple(Q[index] for index in self._indices.MOMENTUM), dims=(dim, dim))
        else:
            gradient_U = grad(U)
            gradient_rho_u = CF(tuple(gradient_U[index, dir]
                                for index in self._indices.MOMENTUM
                                for dir in range(dim)), dims=(dim, dim))

        return gradient_rho_u

    def energy_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method

        if mixed_method is MixedMethods.GRADIENT:
            gradient_rho_E = Q[self._indices.ENERGY]
        else:
            gradient_U = grad(U)
            gradient_rho_E = CF(tuple(gradient_U[self._indices.ENERGY, index] for index in range(self.mesh.dim)))

        return gradient_rho_E

    def pressure_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method
        gamma = self.solver_configuration.heat_capacity_ratio

        rho = self.density(U)
        gradient_rho = self.density_gradient(U, Q)

        if mixed_method is MixedMethods.STRAIN_HEAT:
            T = self.temperature(U)
            gradient_T = self.temperature_gradient(U, Q)

            gradient_p = (gamma - 1)/gamma * (T * gradient_rho + rho * gradient_T)

        else:
            rho_u = self.momentum(U)

            gradient_rho_u = self.momentum_gradient(U, Q)
            gradient_rho_E = self.energy_gradient(U, Q)

            gradient_p = (gamma - 1) * (gradient_rho_E -
                                        gradient_rho_u.trans*rho_u/rho +
                                        InnerProduct(rho_u, rho_u)*gradient_rho/(2*rho**2))

        return gradient_p

    def temperature_gradient(self, U, Q):

        mixed_method = self.solver_configuration.mixed_method
        gamma = self.solver_configuration.heat_capacity_ratio

        if mixed_method is MixedMethods.STRAIN_HEAT:
            gradient_T = CF(tuple(Q[index] for index in self._indices.TEMPERATURE_GRADIENT))

        else:
            rho = self.density(U)
            rho_u = self.momentum(U)
            rho_E = self.energy(U)

            gradient_rho = self.density_gradient(U, Q)
            gradient_rho_m = self.momentum_gradient(U, Q)
            gradient_rho_E = self.energy_gradient(U, Q)

            gradient_T = gamma * (gradient_rho_E/rho -
                                  gradient_rho*rho_E/rho**2 -
                                  gradient_rho_m.trans*rho_u/rho**2 +
                                  InnerProduct(rho_u, rho_u)*gradient_rho/(rho**3))

        return gradient_T

    def velocity_gradient(self, U, Q):

        dim = self.mesh.dim

        rho = self.density(U)
        rho_u = self.momentum(U)

        gradient_rho = self.density_gradient(U, Q)
        gradient_rho_u = self.momentum_gradient(U, Q)

        rho_u_outer_gradient_rho = CF(tuple(rho_u[dir]*gradient_rho for dir in range(dim)), dims=(dim, dim))

        gradient_u = gradient_rho_u/rho - rho_u_outer_gradient_rho/rho**2

        return gradient_u

    def reflect(self, U):

        n = self.normal

        rho = self.density(U)
        rho_u = self.momentum(U)
        rho_E = self.energy(U)

        return CF(tuple(rho, rho_u - InnerProduct(rho_u, n)*n, rho_E))
