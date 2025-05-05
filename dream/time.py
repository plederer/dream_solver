# %%

from __future__ import annotations
import numpy as np
import ngsolve as ngs
import logging
import typing

from dream.config import Integrals, Configuration, dream_configuration

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)


class Timer(Configuration):

    def __init__(self, mesh=None, root=None, **default):

        self._step = ngs.Parameter(1e-4)
        self._t = ngs.Parameter(0.0)

        DEFAULT = {
            "interval": (0.0, 1.0),
            "step": 1e-4,
            "t": 0.0
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def interval(self) -> tuple[float, float]:
        return self._interval

    @interval.setter
    def interval(self, interval: tuple[float, float]):
        start, end = interval

        if start < 0 or end < 0:
            raise ValueError(f"Start and end time must be positive!")

        if start >= end:
            raise ValueError(f"Start time must be smaller than end time!")

        self._interval = (float(start), float(end))

    @dream_configuration
    def step(self) -> ngs.Parameter:
        return self._step

    @step.setter
    def step(self, step: float):
        self._step.Set(step)
        self._set_digit(self._step.Get())

    @dream_configuration
    def t(self) -> ngs.Parameter:
        return self._t

    @t.setter
    def t(self, t: float):
        self._t.Set(t)

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

    def __iter__(self):
        for t in self.start(False, stride=1):
            yield t

    def _set_digit(self, step: float):
        digit = f"{step:.16f}".split(".")[1]
        self.digit = len(digit.rstrip("0"))


class TimeSchemes(Configuration, is_interface=True):

    root: SolverConfiguration
    time_levels: tuple[str, ...]

    @property
    def dt(self) -> ngs.Parameter:
        return self.root.time.timer.step

    def add_symbolic_temporal_forms(self,
                                    space: str,
                                    blf: Integrals,
                                    lf: Integrals) -> None:
        raise NotImplementedError()

    def add_sum_of_integrals(self,
                             form: ngs.LinearForm | ngs.BilinearForm,
                             integrals: Integrals,
                             *pass_terms: tuple[str, ...],
                             fespace: str = None) -> None:

        compile = self.root.optimizations.compile

        # Determine which spaces to iterate over.
        spaces = [fespace] if fespace else integrals.keys()

        for space in spaces:
            if space not in integrals:
                raise KeyError(f"Error: '{space}' not found in integrals.")

            for term, cf in integrals[space].items():
                if term in pass_terms:

                    logger.debug(f"Skipping {term} for space {space}!")
                    continue

                logger.debug(f"Adding {term} term for space {space}!")

                if compile.realcompile:
                    form += cf.Compile(compile.realcompile, compile.wait, compile.keep_files)
                else:
                    form += cf

    def assemble(self) -> None:
        raise NotImplementedError()

    def initialize(self):

        self.dx = self.root.fem.get_temporal_integrators()
        self.spaces = {}
        self.TnT = {}
        self.gfus = {}

        for variable in self.dx:
            self.spaces[variable] = self.root.spaces[variable]
            self.TnT[variable] = self.root.TnT[variable]
            self.gfus[variable] = self.initialize_level_gridfunctions(self.root.gfus[variable])

    def initialize_level_gridfunctions(self, gfu: ngs.GridFunction) -> dict[str, ngs.GridFunction]:
        gfus = [ngs.GridFunction(gfu.space) for _ in self.time_levels[:-1]] + [gfu]
        return {level: gfu for level, gfu in zip(self.time_levels, gfus)}

    def set_initial_conditions(self):
        for gfu in self.gfus.values():
            for old in list(gfu.values())[:-1]:
                old.vec.data = gfu['n+1'].vec

    def solve_current_time_level(self) -> typing.Generator[int | None, None, None]:
        raise NotImplementedError()

    def update_gridfunctions(self):
        for gfu in self.gfus.values():
            for old, new in zip(self.time_levels[:-1], self.time_levels[1:]):
                gfu[old].vec.data = gfu[new].vec




class TimeConfig(Configuration, is_interface=True):

    root: SolverConfiguration

    @property
    def is_stationary(self) -> bool:
        return isinstance(self, StationaryConfig)

    def assemble(self) -> None:
        raise NotImplementedError("Symbolic Forms not implemented!")

    def add_symbolic_temporal_forms(self, blf, lf) -> None:
        pass
    
    def scheme(self) -> None:
        pass

    def initialize(self):
        pass

    def set_initial_conditions(self):
        self.root.fem.set_initial_conditions()

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:
        raise NotImplementedError("Solution Routine not implemented!")


class StationaryConfig(TimeConfig):

    name: str = "stationary"

    def assemble(self) -> None:

        condense = self.root.optimizations.static_condensation
        compile = self.root.optimizations.compile

        self.blf = ngs.BilinearForm(self.root.fes, condense=condense)
        self.lf = ngs.LinearForm(self.root.fes)

        for name, cf in self.root.blf.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blf += cf.Compile(**compile)
            else:
                self.blf += cf

        for name, cf in self.root.lf.items():
            logger.debug(f"Adding {name} to the LinearForm!")

            if compile.realcompile:
                self.lf += cf.Compile(**compile)
            else:
                self.lf += cf

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        if reassemble:
            self.assemble()

        with self.root.io as io:
            io.save_pre_time_routine()

            # Solution routine starts here
            self.solve()
            yield None
            # Solution routine ends here

            io.save_post_time_routine()


class TransientConfig(TimeConfig):

    name: str = "transient"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "timer": Timer(mesh, root)
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @property
    def scheme(self) -> TimeSchemes:
        raise NotImplementedError("Overload this configuration in derived class!")

    @dream_configuration
    def timer(self) -> Timer:
        return self._timer

    @timer.setter
    def timer(self, timer: Timer):

        if not isinstance(timer, Timer):
            raise TypeError(f"Timer must be of type {Timer}!")

        self._timer = timer

    def assemble(self):
        self.scheme.assemble()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals):
        self.root.fem.add_symbolic_temporal_forms(blf, lf)

    def initialize(self):
        super().initialize()
        self.scheme.initialize()

    def set_initial_conditions(self):
        super().set_initial_conditions()
        self.scheme.set_initial_conditions()

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        if reassemble:
            self.assemble()

        with self.root.io as io:
            io.save_pre_time_routine(self.timer.t.Get())

            # Solution routine starts here
            for it, t in enumerate(self.timer()):

                for _ in self.scheme.solve_current_time_level(t):
                    continue

                self.scheme.update_gridfunctions()

                yield t

                io.save_in_time_routine(t, it)
                io.redraw()
            # Solution routine ends here

            io.save_post_time_routine(t, it)


class PseudoTimeSteppingConfig(TimeConfig):

    name: str = "pseudo_time_stepping"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "timer": Timer(mesh, root),
            "max_time_step": 1.0,
            "increment_at": 10,
            "increment_factor": 10,
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @property
    def scheme(self) -> TimeSchemes:
        raise NotImplementedError("Overload this configuration in derived class!")

    @dream_configuration
    def timer(self) -> Timer:
        return self._timer

    @timer.setter
    def timer(self, timer: Timer):

        if not isinstance(timer, Timer):
            raise TypeError(f"Timer must be of type {Timer}!")

        self._timer = timer

    @dream_configuration
    def max_time_step(self) -> float:
        return self._max_time_step

    @max_time_step.setter
    def max_time_step(self, max_time_step: float):
        self._max_time_step = float(max_time_step)

    @dream_configuration
    def increment_at(self) -> int:
        return self._increment_at

    @increment_at.setter
    def increment_at(self, increment_at: int):
        self._increment_at = int(increment_at)

    @dream_configuration
    def increment_factor(self) -> int:
        return self._increment_factor

    @increment_factor.setter
    def increment_factor(self, increment_factor: int):
        self._increment_factor = int(increment_factor)

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals):
        self.root.fem.add_symbolic_temporal_forms(blf, lf)

    def assemble(self):
        self.scheme.assemble()

    def initialize(self):
        super().initialize()
        self.scheme.initialize()

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        if reassemble:
            self.scheme.assemble()

        with self.root.io as io:
            io.save_pre_time_routine(self.timer.t.Get())

            # Solution routine starts here
            for it in self.scheme.solve_current_time_level():
                self.scheme.update_gridfunctions()
                self.solver_iteration_update(it)
                io.redraw()

            yield None

            # Solution routine ends here
            io.save_in_time_routine(self.timer.t.Get(), it=0)

            io.redraw()

            io.save_post_time_routine(self.timer.t.Get())

    def solver_iteration_update(self, it: int):
        old_time_step = self.timer.step.Get()

        if self.max_time_step > old_time_step:
            if (it % self.increment_at == 0) and (it > 0):
                new_time_step = old_time_step * self.increment_factor

                if new_time_step > self.max_time_step:
                    new_time_step = self.max_time_step

                self.timer.step = new_time_step
                logger.info(f"Successfully updated time step at iteration {it}")
                logger.info(f"Updated time step 𝚫t = {new_time_step}. Previous time step 𝚫t = {old_time_step}")

    def set_initial_conditions(self):
        super().set_initial_conditions()
        self.scheme.set_initial_conditions()
