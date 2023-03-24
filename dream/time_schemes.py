from __future__ import annotations
import abc
import typing
import enum
import numpy as np
import dataclasses
from collections import UserDict
from ngsolve import Parameter, GridFunction, CF

if typing.TYPE_CHECKING:
    from .configuration import SolverConfiguration


@dataclasses.dataclass
class TimePeriod:
    start: float
    end: float
    step: Parameter

    def __iter__(self):
        for t in self.range(step=1):
            yield t

    def range(self, step: int = 1):
        dt = self.step.Get()
        num = round((self.end - self.start)/dt) + 1
        for i in range(1, num, step):
            yield self.start + i * dt

    def array(self, include_start_time: bool = False) -> np.ndarray:
        dt = self.step.Get()
        num = round((self.end - self.start)/dt) + 1

        time_period = np.linspace(self.start, self.end, num)
        if not include_start_time:
            time_period = time_period[1:]
        return time_period

    def __repr__(self) -> str:
        return f"Interval: ({self.start}, {self.end}], Time Step: {self.step.Get()}"


class TimeLevelsGridfunction(UserDict):

    def __init__(self, gfus: tuple[GridFunction, ...]) -> None:
        time_levels = ['n+1', 'n'] + [f'n{i}' for i in range(-1, 1-len(gfus), -1)]
        super().__init__({time_level: gfu for time_level, gfu in zip(time_levels, gfus)})

    def get_component(self, component: int) -> TimeLevelsGridfunction:
        components = tuple(gfu.components[component] for gfu in self.values())
        return type(self)(components)

    def __getitem__(self, level: str) -> GridFunction:
        return super().__getitem__(level)

    def items(self) -> typing.ItemsView[str, GridFunction]:
        return super().items()


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

    time_levels: int = 2

    def __init__(self, solver_configuration: SolverConfiguration) -> None:
        self.solver_configuration = solver_configuration

    @abc.abstractmethod
    def apply(self, cf: TimeLevelsGridfunction) -> CF: ...

    @abc.abstractmethod
    def update_previous_solution(self, cf: TimeLevelsGridfunction): ...

    @abc.abstractmethod
    def update_initial_solution(self, cf: TimeLevelsGridfunction): ...


class ImplicitEuler(_TimeSchemes):

    def apply(self, cf: TimeLevelsGridfunction) -> CF:
        dt = self.solver_configuration.time_step
        return 1/dt * (cf['n+1'] - cf['n'])

    def update_previous_solution(self, cf: TimeLevelsGridfunction):
        cf['n'].vec.data = cf['n+1'].vec

    def update_initial_solution(self, cf: TimeLevelsGridfunction):
        self.update_previous_solution(cf)


class BDF2(_TimeSchemes):

    time_levels: int = 3

    def apply(self, cf: TimeLevelsGridfunction) -> CF:
        dt = self.solver_configuration.time_step
        return (3*cf['n+1'] - 4 * cf['n'] + cf['n-1']) / (2 * dt)

    def update_previous_solution(self, cf: TimeLevelsGridfunction):
        cf['n-1'].vec.data = cf['n'].vec
        cf['n'].vec.data = cf['n+1'].vec

    def update_initial_solution(self, cf: TimeLevelsGridfunction):
        cf['n'].vec.data = cf['n+1'].vec
        cf['n-1'].vec.data = cf['n+1'].vec
