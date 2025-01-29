# %%

from __future__ import annotations
import numpy as np
import ngsolve as ngs
import logging
import typing

from dream.config import UniqueConfiguration, InterfaceConfiguration, parameter, configuration, interface, unique

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)


class Timer(UniqueConfiguration):

    @configuration(default=(0.0, 1.0))
    def interval(self, interval):
        start, end = interval

        if start < 0 or end < 0:
            raise ValueError(f"Start and end time must be positive!")

        if start >= end:
            raise ValueError(f"Start time must be smaller than end time!")

        return (float(start), float(end))

    @parameter(default=1e-4)
    def step(self, step):
        self._set_digit(step)
        return step

    @parameter(default=0.0)
    def t(self, t):
        return t

    def start(self, include_start: bool = False, stride: int = 1):

        start, end = self.interval
        step = self.step.Get()

        N = round((end - start)/(stride*step)) + 1

        for i in range(1 - include_start, N):
            self.t = start + stride*i*step
            yield round(self.t.Get(), self.digit)

    def to_array(self, include_start: bool = False, stride: int = 1) -> np.ndarray:
        return np.array(list(self.start(include_start, stride)))

    def __call__(self, **kwargs):
        self.update(**kwargs)
        for t in self.start(stride=1):
            yield t

    def _set_digit(self, step: float):
        digit = f"{step:.16f}".split(".")[1]
        self.digit = len(digit.rstrip("0"))

    interval: tuple[float, float]
    step: ngs.Parameter
    t: ngs.Parameter


class TimeSchemes(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration
    time_levels: tuple[str, ...]

    @property
    def dt(self) -> ngs.Parameter:
        return self.cfg.time.timer.step

    def get_transient_gridfunctions(self, gfu: ngs.GridFunction) -> dict[str, ngs.GridFunction]:
        gfus = [ngs.GridFunction(gfu.space) for _ in self.time_levels[:-1]] + [gfu]
        return {level: gfu for level, gfu in zip(self.time_levels, gfus)}

    def update_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]):

        for gfu in gfus.values():
            for old, new in zip(self.time_levels[:-1], self.time_levels[1:]):
                gfu[old].vec.data = gfu[new].vec

    def set_initial_conditions(self, gfus: dict[str, dict[str, ngs.GridFunction]]):

        for gfu in gfus.values():
            for old in list(gfu.values())[:-1]:
                old.vec.data = gfu['n+1'].vec

    def get_discrete_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_implicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_explicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_time_step(self) -> ngs.CF:
        raise NotImplementedError()

    def get_normalized_explicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_normalized_time_step(self) -> ngs.CF:
        raise NotImplementedError()


class ImplicitEuler(TimeSchemes):

    name: str = "implicit_euler"
    aliases = ("ie", )

    time_levels = ('n', 'n+1')

    def get_discrete_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return (gfus['n+1'] - gfus['n'])/self.dt

    def get_implicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return gfus['n+1']

    def get_explicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return gfus['n']

    def get_time_step(self) -> ngs.CF:
        return self.dt

    def get_normalized_explicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return self.get_explicit_terms(gfus)

    def get_normalized_time_step(self) -> ngs.CF:
        return self.dt


class BDF2(TimeSchemes):

    name: str = "bdf2"

    time_levels = ('n-1', 'n', 'n+1')

    def get_discrete_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return (3*gfus['n+1'] - 4*gfus['n'] + gfus['n-1'])/(2*self.dt)

    def get_implicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return 3 * gfus['n+1']

    def get_explicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return 4 * gfus['n'] - gfus['n-1']

    def get_time_step(self):
        return 2*self.dt

    def get_normalized_explicit_terms(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return 4/3*gfus['n'] - 1/3*gfus['n-1']

    def get_normalized_time_step(self):
        return 2/3*self.dt


class TimeConfig(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def is_stationary(self) -> bool:
        return isinstance(self, StationaryConfig)

    def start_solution_routine(self) -> typing.Generator[float | None, None, None]:
        raise NotImplementedError("Solution Routine not implemented!")

    def solver_iteration_update(self, it: int):
        pass


class StationaryConfig(TimeConfig):

    name: str = "stationary"

    def start_solution_routine(self) -> typing.Generator[float | None, None, None]:

        with self.cfg.io as io:
            io.save_pre_time_routine()

            # Solution routine starts here
            yield None
            # Solution routine ends here

            io.save_post_time_routine()


class TransientConfig(TimeConfig):

    name: str = "transient"

    @interface(default=ImplicitEuler)
    def scheme(self, scheme):
        return scheme

    @unique(default=Timer)
    def timer(self, timer):
        return timer

    def start_solution_routine(self) -> typing.Generator[float | None, None, None]:

        with self.cfg.io as io:
            io.save_pre_time_routine(self.timer.t.Get())

            # Solution routine starts here
            for it, t in enumerate(self.timer()):
                self.scheme.update_transient_gridfunctions(self.cfg.pde.transient_gfus)

                yield t

                io.save_in_time_routine(t, it)
                self.cfg.pde.redraw()
            # Solution routine ends here

            io.save_post_time_routine(t, it)

    scheme: ImplicitEuler | BDF2
    timer: Timer


class PseudoTimeSteppingConfig(TimeConfig):

    name: str = "pseudo_time_stepping"
    aliases = ("pseudo", )

    @interface(default=ImplicitEuler)
    def scheme(self, scheme):
        return scheme

    @unique(default=Timer)
    def timer(self, timer):
        return timer

    @configuration(default=1.0)
    def max_time_step(self, max_time_step):
        return float(max_time_step)

    @configuration(default=10)
    def increment_at(self, increment_at):
        return int(increment_at)

    @configuration(default=10)
    def increment_factor(self, increment_factor):
        return int(increment_factor)

    def start_solution_routine(self) -> typing.Generator[float | None, None, None]:

        with self.cfg.io as io:
            io.save_pre_time_routine(self.timer.t.Get())

            # Solution routine starts here
            self.scheme.update_transient_gridfunctions(self.cfg.pde.transient_gfus)

            yield None
            # Solution routine ends here
            io.save_in_time_routine(self.timer.t.Get(), it=0)

            self.cfg.pde.redraw()

            io.save_post_time_routine(self.timer.t.Get())

    def solver_iteration_update(self, it: int):
        self.scheme.update_transient_gridfunctions(self.cfg.pde.transient_gfus)

        old_time_step = self.timer.step.Get()

        if self.max_time_step > old_time_step:
            if (it % self.increment_at == 0) and (it > 0):
                new_time_step = old_time_step * self.increment_factor

                if new_time_step > self.max_time_step:
                    new_time_step = self.max_time_step

                self.timer.step = new_time_step
                logger.info(f"Successfully updated time step at iteration {it}")
                logger.info(f"Updated time step 𝚫t = {new_time_step}. Previous time step 𝚫t = {old_time_step}")

        self.cfg.pde.redraw()

    scheme: ImplicitEuler | BDF2
    timer: Timer
    max_time_step: float
    increment_at: int
    increment_factor: int
