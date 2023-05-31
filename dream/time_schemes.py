from __future__ import annotations
import abc
import enum

from typing import ItemsView, Any, TYPE_CHECKING
from collections import UserDict
from ngsolve import GridFunction, CF

if TYPE_CHECKING:
    from .configuration import TimeConfiguration


class TimeSchemes(enum.Enum):
    IMPLICIT_EULER = "IE"
    BDF2 = "BDF2"


class Simulation(enum.Enum):
    STATIONARY = "stationary"
    TRANSIENT = "transient"


def time_scheme_factory(time_configuration: TimeConfiguration) -> _TimeSchemes:

    scheme = time_configuration.scheme

    if scheme is TimeSchemes.IMPLICIT_EULER:
        return ImplicitEuler(time_configuration)
    elif scheme is TimeSchemes.BDF2:
        return BDF2(time_configuration)


class TimeLevelsGridfunction(UserDict):

    def get_component(self, component: int) -> TimeLevelsGridfunction:
        components = {level: gfu.components[component] for level, gfu in self.items()}
        return type(self)(components)

    def __getitem__(self, level: str) -> GridFunction:
        return super().__getitem__(level)

    def items(self) -> ItemsView[str, GridFunction]:
        return super().items()


class _TimeSchemes(abc.ABC):

    time_levels: tuple[str, ...] = None

    def __init__(self, time_configuration: TimeConfiguration) -> None:
        self.cfg = time_configuration

    @abc.abstractmethod
    def apply(self, cf: TimeLevelsGridfunction) -> CF: ...

    @abc.abstractmethod
    def update_previous_solution(self, cf: TimeLevelsGridfunction): ...

    @abc.abstractmethod
    def update_initial_solution(self, cf: TimeLevelsGridfunction): ...


class ImplicitEuler(_TimeSchemes):

    time_levels = ('n+1', 'n')

    def apply(self, cf: TimeLevelsGridfunction) -> CF:
        dt = self.cfg.step
        return 1/dt * (cf['n+1'] - cf['n'])

    def update_previous_solution(self, cf: TimeLevelsGridfunction):
        cf['n'].vec.data = cf['n+1'].vec

    def update_initial_solution(self, cf: TimeLevelsGridfunction):
        self.update_previous_solution(cf)


class BDF2(_TimeSchemes):

    time_levels = ('n+1', 'n', 'n-1')

    def apply(self, cf: TimeLevelsGridfunction) -> CF:
        dt = self.cfg.step
        return (3*cf['n+1'] - 4 * cf['n'] + cf['n-1']) / (2 * dt)

    def update_previous_solution(self, cf: TimeLevelsGridfunction):
        cf['n-1'].vec.data = cf['n'].vec
        cf['n'].vec.data = cf['n+1'].vec

    def update_initial_solution(self, cf: TimeLevelsGridfunction):
        cf['n'].vec.data = cf['n+1'].vec
        cf['n-1'].vec.data = cf['n+1'].vec
