from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from .configuration import SolverConfiguration
from ngsolve import InnerProduct


class TimeSchemes(Enum):
    IE = "IE"
    BDF2 = "BDF2"


def time_scheme_factory(mesh, solver_configuration: SolverConfiguration) -> TimeSchemesImplementation:

    time_scheme = solver_configuration.time_scheme

    if time_scheme is TimeSchemes.IE:
        return ImplicitEuler(solver_configuration)
    elif time_scheme is TimeSchemes.BDF2:
        return BDF2(solver_configuration)


class TimeSchemesImplementation(ABC):

    num_temporary_vectors: int

    def __init__(self, solver_configuration: SolverConfiguration) -> None:
        self.solver_configuration = solver_configuration

    @abstractmethod
    def apply_scheme(self, U, V, *old_components): ...

    @abstractmethod
    def update_temporary_vectors(new, *old_components): ...

    def __call__(self, U, V, *old_components):
        return self.apply_scheme(U, V, *old_components)


class ImplicitEuler(TimeSchemesImplementation):

    num_temporary_vectors: int = 1

    def apply_scheme(self, U, V, *old_components):
        dt = self.solver_configuration.time_step
        return 1/dt * InnerProduct(U - old_components[0], V)

    def update_temporary_vectors(new, *old_components):
        old_components[0].vec.data = new.vec


class BDF2(TimeSchemesImplementation):

    num_temporary_vectors: int = 2

    def apply_scheme(self, U, V, *old_components):
        dt = self.solver_configuration.time_step
        return 3/2 * 1/dt * InnerProduct(U - 4/3 * old_components[0] + 1/3 * old_components[1], V)

    def update_temporary_vectors(new, *old_components):
        old_components[1].vec.data = old_components[0].vec
        old_components[0].vec.data = new.vec
