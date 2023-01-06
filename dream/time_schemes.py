from __future__ import annotations
import abc
import typing
import enum

if typing.TYPE_CHECKING:
    from .configuration import SolverConfiguration


class TimeSchemes(enum.Enum):
    IMPLICIT_EULER = "IE"
    BDF2 = "BDF2"


def time_scheme_factory(solver_configuration: SolverConfiguration) -> _TimeSchemes:

    time_scheme = solver_configuration.time_scheme

    if time_scheme is TimeSchemes.IMPLICIT_EULER:
        return ImplicitEuler(solver_configuration)
    elif time_scheme is TimeSchemes.BDF2:
        return BDF2(solver_configuration)


class _TimeSchemes(abc.ABC):

    num_temporary_vectors: int

    def __init__(self, solver_configuration: SolverConfiguration) -> None:
        self.solver_configuration = solver_configuration

    @abc.abstractmethod
    def apply_scheme(self, U, *old_components): ...

    @abc.abstractmethod
    def update_previous_solution(self, new, *old_components): ...

    @abc.abstractmethod
    def set_initial_solution(self, new, *old_components): ...

    def __call__(self, U, *old_components):
        return self.apply_scheme(U, *old_components)


class ImplicitEuler(_TimeSchemes):

    num_temporary_vectors: int = 1

    def apply_scheme(self, U, *old_components):
        dt = self.solver_configuration.time_step
        return 1/dt * (U - old_components[0])

    def update_previous_solution(self, new, *old_components):
        old_components[0].vec.data = new.vec

    def set_initial_solution(self, new, *old_components):
        self.update_previous_solution(new, *old_components)


class BDF2(_TimeSchemes):

    num_temporary_vectors: int = 2

    def apply_scheme(self, U, *old_components):
        dt = self.solver_configuration.time_step
        return (3*U - 4 * old_components[0] + old_components[1]) / (2 * dt)

    def update_previous_solution(self, new, *old_components):
        old_components[1].vec.data = old_components[0].vec
        old_components[0].vec.data = new.vec

    def set_initial_solution(self, new, *old_components):
        old_components[0].vec.data = new.vec
        old_components[1].vec.data = new.vec
