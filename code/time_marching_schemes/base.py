from __future__ import annotations
from abc import ABC, abstractmethod
from configuration import SolverConfiguration
from ngsolve import InnerProduct


class TimeMarchingSchemes(ABC):

    def __init__(self, solver_configuration: SolverConfiguration) -> None:
        self.solver_configuration = solver_configuration

    @abstractmethod
    def get_scheme(self, U, V, *old_components): ...


class ImplicitEuler(TimeMarchingSchemes):

    def get_scheme(self, U, V, *old_components):
        dt = self.solver_configuration.time_step
        return 1/dt * InnerProduct(U - old_components[0], V)


class BDF2(TimeMarchingSchemes):

    def get_scheme(self, U, V, *old_components):
        dt = self.solver_configuration.time_step
        return 3/2 * 1/dt * InnerProduct(U - 4/3 * old_components[0] + 1/3 * old_components[1], V)
