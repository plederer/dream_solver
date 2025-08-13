# %%

from __future__ import annotations
import numpy as np
import ngsolve as ngs
import logging
import typing

from dream.config import Integrals, Log, Configuration, dream_configuration

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

        start, _ = self.interval
        step = self.step.Get()
        size = self.num_steps(include_start, stride)

        for i in range(1 - include_start, size + 1):
            self.t = start + stride*i*step
            yield self.t.Get()

    def to_array(self, include_start: bool = False, stride: int = 1) -> np.ndarray:
        return np.array(list(self.start(include_start, stride)))

    def reset(self):
        self.t = self.interval[0]

    def num_steps(self, include_start: bool = False, stride: int = 1) -> int:
        start, end = self.interval
        step = self.step.Get()

        size = round((end - start)/(stride * step))
        if include_start:
            size += 1

        return size

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


class Scheme(Configuration, is_interface=True):

    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):

        self._compile = {'realcompile': False, 'wait': False, 'keep_files': False}

        DEFAULT = {
            "compile": False,
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def compile(self) -> dict[str, bool]:
        return self._compile

    @compile.setter
    def compile(self, compile: bool):

        if bool(compile):
            self._compile['realcompile'] = True
        else:
            self._compile['realcompile'] = False

    def add_sum_of_integrals(self,
                             form: ngs.LinearForm | ngs.BilinearForm,
                             integrals: Integrals,
                             name: str = "form") -> None:

        logger.debug(f"Adding forms to {name}...")

        for space, terms in integrals.items():

            for term, cf in terms.items():

                logger.debug(f"  Adding {term} term for space {space} to {name}!")

                if self.compile['realcompile']:
                    form += cf.Compile(**self.compile)
                else:
                    form += cf

        logger.debug("Done.")

    def parse_sum_of_integrals(self,
                               integrals: Integrals,
                               include_spaces: tuple[str, ...] = None,
                               exclude_spaces: tuple[str, ...] = None,
                               include_terms: tuple[str, ...] = None,
                               exclude_terms: tuple[str, ...] = None) -> Integrals:
        """ Parse the sum of integrals dictionary to include or exclude specific spaces and terms.

            By default, it includes all spaces and terms in the integrals. You can specify which spaces to include 
            or exclude, and which terms to include or exclude. If a space in the include container 
            is not found in the integrals dictionary, it will raise an error.
        """

        # Get the spaces to iterate over. If None, include all spaces.
        spaces = include_spaces
        if spaces is None:
            spaces = tuple(integrals)

        # Exclude spaces if specified
        if exclude_spaces is not None:
            spaces = tuple(space for space in spaces if space not in exclude_spaces)

        # Prevent modification of the original integrals dictionary and throw an error if a space is not found
        integrals = {space: integrals[space].copy() for space in spaces}

        # Parse exclude terms
        if exclude_terms is not None:

            for space in list(integrals):
                for term in exclude_terms:
                    if term in integrals[space]:
                        integrals[space].pop(term)

                if not integrals[space]:
                    integrals.pop(space)

        # Parse include terms. Throws an error if a term is not found
        if include_terms is not None:
            integrals = {space: {term: integral[term] for term in include_terms if term in integral}
                         for space, integral in integrals.items()}

        # Omit empty integrals
        integrals = {space: {term: cf for term, cf in integral.items() if cf} for space, integral in integrals.items() if integral}

        if not integrals:
            raise ValueError("No integrals found after parsing! Check your include/exclude terms/spaces.")

        return integrals

    def assemble(self) -> None:
        raise NotImplementedError("Overload this method in derived class!")

    def solve_stationary(self) -> typing.Generator[Log, None, None]:
        raise NotImplementedError("Overload this method in derived class!")


class TimeSchemes(Scheme):

    time_levels: tuple[str, ...]

    @property
    def dt(self) -> ngs.Parameter:
        return self.root.time.timer.step

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        raise NotImplementedError("Overload this method in derived class!")

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:
        raise NotImplementedError("Overload this method in derived class!")

    def get_level_gridfunctions(self, gfu: ngs.GridFunction) -> dict[str, ngs.GridFunction]:
        gfus = [ngs.GridFunction(gfu.space) for _ in self.time_levels[:-1]] + [gfu]
        return {level: gfu for level, gfu in zip(self.time_levels, gfus)}

    def initialize_gridfunctions(self, gfus: dict[str, ngs.GridFunction]) -> None:
        self.gfus = {space: self.get_level_gridfunctions(gfu) for space, gfu in gfus.items()}

    def set_initial_conditions(self):
        for gfu in self.gfus.values():
            for old in list(gfu.values())[:-1]:
                old.vec.data = gfu['n+1'].vec

    def update_gridfunctions(self):
        for gfu in self.gfus.values():
            for old, new in zip(self.time_levels[:-1], self.time_levels[1:]):
                gfu[old].vec.data = gfu[new].vec


class TimeRoutine(Configuration, is_interface=True):

    root: SolverConfiguration

    @property
    def is_stationary(self) -> bool:
        return isinstance(self, StationaryRoutine)

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:
        raise NotImplementedError("Overload this method in derived class!")

    def parse_routine_log(self,
                          it: int | None = None,
                          error: float | None = None, **kwargs):
        """ Parse the routine log and return a formatted string. """

        fem = self.root.fem

        msg = f"{self.name} | {fem.name} {fem.scheme.name}"
        if it is not None:
            msg += f" | it: {it}"
        if error is not None:
            msg += f" | error: {error:8e}"

        return msg


class StationaryRoutine(TimeRoutine):

    name: str = "stationary"

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        scheme = self.root.fem.scheme

        if reassemble:
            scheme.assemble()

        with self.root.io as io:
            io.save_pre_time_routine()

            # Solution routine starts here
            for log in scheme.solve_stationary():
                logger.info(self.parse_routine_log(**log))

            yield None
            # Solution routine ends here

            io.save_post_time_routine()
        io.redraw()


class TransientRoutine(TimeRoutine):

    name: str = "transient"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "timer": Timer(mesh, root)
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def timer(self) -> Timer:
        return self._timer

    @timer.setter
    def timer(self, timer: Timer):

        if not isinstance(timer, Timer):
            raise TypeError(f"Timer must be of type {Timer}!")

        self._timer = timer

    def parse_routine_log(self,
                          stage: int = None,
                          t: float = None,
                          **kwargs):
        """ Parse the routine log and return a formatted string. """

        msg = super().parse_routine_log(**kwargs)

        if stage is not None:
            msg += f" | stage: {stage}"
        if t is not None:
            msg += f" | t: {t:.{self.timer.digit}f}"

        return msg

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        scheme = self.root.fem.scheme
        timer = self.timer

        timer.reset()

        if reassemble:
            scheme.assemble()

        with self.root.io as io:
            io.save_pre_time_routine(self.timer.t.Get())

            # Solution routine starts here
            for rate, t in enumerate(self.timer()):

                for log in scheme.solve_current_time_level():
                    logger.info(self.parse_routine_log(t=t, **log))

                scheme.update_gridfunctions()

                yield t

                io.save_in_time_routine(t, rate)
                io.redraw(rate)
            # Solution routine ends here

            io.save_post_time_routine(t, rate)


class PseudoTimeSteppingRoutine(TimeRoutine):

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

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        self.timer.reset()

        scheme = self.root.fem.scheme

        if reassemble:
            scheme.assemble()

        with self.root.io as io:
            io.save_pre_time_routine()

            # Solution routine starts here
            for log in scheme.solve_current_time_level():
                logger.info(self.parse_routine_log(**log))

                scheme.update_gridfunctions()
                self.solver_iteration_update(log['it'])
                io.redraw()

            yield None

            # Solution routine ends here
            io.save_post_time_routine()

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



class MultizoneIMEXTimeRoutine(TimeRoutine):

    name: str = "multizone_imex_transient"

    def __init__(self, cfg_implicit=None, cfg_explicit=None, root=None, **default):

        DEFAULT = {
            "timer": Timer(root)
        }
        DEFAULT.update(default)

        super().__init__(root, **DEFAULT)

        # Keep references for the configurations.
        self.cfg_implicit = cfg_implicit
        self.cfg_explicit = cfg_explicit

       
    # Function that checks whether the specified schemes form a viable IMEX pair.
    def is_valid_imex_schemes(self) -> bool:

        n_imp = self.cfg_implicit.fem.scheme.get_num_stages()
        n_exp = self.cfg_explicit.fem.scheme.get_num_stages()
        
        if n_imp != n_exp:
            return False

        cdt_imp = self.cfg_implicit.fem.scheme.get_stage_dt()
        cdt_exp = self.cfg_explicit.fem.scheme.get_stage_dt()

        # NOTE, if AIMEX schemes are used, this condition probably will fail.
        if not np.allclose(cdt_imp, cdt_exp, rtol=1e-10, atol=1e-10):
            return False

        # NOTE, all first stages are padded, since they start explicitly, i.e. cdt[0] = 0.
        if (n_imp != len(cdt_imp)-1) or (n_exp != len(cdt_exp)-1):
            return False
        
        # Book-keep number of stages, including first (padded) stage.
        self.nStage = len(cdt_imp)

        # All good.
        return True

    @dream_configuration
    def timer(self) -> Timer:
        return self._timer

    @timer.setter
    def timer(self, timer: Timer):

        if not isinstance(timer, Timer):
            raise TypeError(f"Timer must be of type {Timer}!")

        self._timer = timer

    def parse_routine_log(self,
                          stage: int = None,
                          t: float = None,
                          **kwargs):
        """ Parse the routine log and return a formatted string. """

        msg = super().parse_routine_log(**kwargs)

        if stage is not None:
            msg += f" | stage: {stage}"
        if t is not None:
            msg += f" | t: {t:.{self.timer.digit}f}"

        return msg

    
    def solve(self, reassemble: bool=True):

        timer = self.timer
        timer.reset()
        
        if reassemble:
            self.cfg_implicit.fem.scheme.assemble()
            self.cfg_explicit.fem.scheme.assemble()

        if not self.is_valid_imex_schemes():
            raise TypeError(f"Specified time schemes do not form an IMEX pair.")

        with self.cfg_implicit.io as io_imp, self.cfg_explicit.io as io_exp:

            io_imp.save_pre_time_routine(self.timer.t.Get())
            io_exp.save_pre_time_routine(self.timer.t.Get())

            for rate, t in enumerate(self.timer()):
                
                # We loop over each stage and solve explicit regions first, then implicit ones.
                for iStage in range(1, self.nStage):
                    for log_explicit in self.cfg_explicit.fem.scheme.solve_stage(iStage):
                        logger.info(self.parse_routine_log(t=t, **log_explicit))
                    for log_implicit in self.cfg_implicit.fem.scheme.solve_stage(iStage):
                        logger.info(self.parse_routine_log(t=t, **log_implicit))

                # These are needed in case the schemes aren't stiffly accurate.
                self.cfg_explicit.fem.scheme.update_solution()
                self.cfg_implicit.fem.scheme.update_solution()
                
                print() # One empty line to rule them all and ruin Jan's day..
                
                self.cfg_explicit.fem.scheme.update_gridfunctions()
                self.cfg_implicit.fem.scheme.update_gridfunctions()

                io_exp.save_in_time_routine(t, rate)
                io_imp.save_in_time_routine(t, rate)
                
                io_exp.redraw(rate)
                io_imp.redraw(rate)
                
            io_exp.save_post_time_routine(t, rate)
            io_imp.save_post_time_routine(t, rate)













