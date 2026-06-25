from __future__ import annotations
import numpy as np
import ngsolve as ngs
import logging
import typing
from functools import wraps
import time

from dream.config import Integrals, Log, Configuration, dream_configuration

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)


def time_generator(label):

    def decorator(generator):

        @wraps(generator)
        def wrap(self, *args, **kwargs):
            cfg: SolverConfiguration = self.root

            if not cfg.io.log.time_routines:
                yield from generator(self, *args, **kwargs)
                return

            start = time.perf_counter()
            yield from generator(self, *args, **kwargs)
            logger.info(f"Solve {label.format(*args, **kwargs)} runtime: {time.perf_counter() - start:.6f}s")

        return wrap

    return decorator


class Timer(Configuration):
    r""" Keeps track of the simulation time and time step used by a :class:`TimeRoutine`.

        The timer holds the time interval :math:`(t_0, t_{\mathrm{end}})` to be simulated, the current
        simulation time :math:`t` and the time step :math:`\Delta t`. Calling the timer as a generator,
        see :meth:`__call__`, advances :math:`t` from :math:`t_0` to :math:`t_{\mathrm{end}}` in increments
        of :math:`\Delta t`, yielding the iteration number together with the time levels :math:`t_n` and
        :math:`t_{n+1}` bracketing each step.
    """

    name: str = "timer"

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
        r""" Sets the time interval :math:`(t_0, t_{\mathrm{end}})` to be simulated.

            :getter: Returns the time interval
            :setter: Sets the time interval, defaults to (0.0, 1.0)
        """
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
        r""" Sets the time step :math:`\Delta t` used to advance the simulation.

            :getter: Returns the time step
            :setter: Sets the time step, defaults to 1e-4
        """
        return self._step

    @step.setter
    def step(self, step: float):
        self._step.Set(step)
        self._set_digit(self._step.Get())

    @dream_configuration
    def t(self) -> ngs.Parameter:
        r""" Sets the current simulation time :math:`t`.

            :getter: Returns the current simulation time
            :setter: Sets the current simulation time, defaults to 0.0
        """
        return self._t

    @t.setter
    def t(self, t: float):
        self._t.Set(t)

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

    def __call__(self, stride: int = 1, **kwargs):
        self.update(**kwargs)
        t0, end = self.interval
        step = self.step.Get() * stride

        self.t = t0
        for rate in range(1, round((end - t0)/(step))+1):
            tn = t0 + step
            yield rate, t0, tn
            t0 = tn

    def __iter__(self):
        for t in self.start(False, stride=1):
            yield t

    def _set_digit(self, step: float):
        digit = f"{step:.16f}".split(".")[1]
        self.digit = len(digit.rstrip("0"))


class Scheme(Configuration, is_interface=True):
    r""" Base interface for a spatial/temporal discretization scheme.

        A :class:`Scheme` is responsible for assembling the bilinear and linear forms of the
        discretized problem, see :meth:`assemble`, and for solving either a stationary problem,
        see :meth:`solve_stationary`, or a single time level/stage, see :class:`TimeSchemes`.
    """

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
        r""" Sets the NGSolve compile options used when assembling the symbolic forms.

            :getter: Returns the compile options
            :setter: Sets the compile options, defaults to False
        """
        return self._compile

    @compile.setter
    def compile(self, compile: bool):

        if isinstance(compile, dict):
            self._compile.update(compile)
        elif bool(compile):
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

                form += cf.Compile(**self.compile)

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
        integrals = {space: {term: cf for term, cf in integral.items() if cf}
                     for space, integral in integrals.items() if integral}

        if not integrals:
            raise ValueError("No integrals found after parsing! Check your include/exclude terms/spaces.")

        return integrals

    def assemble(self) -> None:
        raise NotImplementedError("Overload this method in derived class!")

    def solve_stationary(self) -> typing.Generator[Log, None, None]:
        raise NotImplementedError("Overload this method in derived class!")


class TimeSchemes(Scheme):
    r""" Base interface for a concrete (multi-step or multi-stage) time-stepping scheme.

        A :class:`TimeSchemes` discretizes the time derivative of the semi-discrete problem and
        advances the solution either over a single time level, see :meth:`solve_current_time_level`,
        or over a single Runge-Kutta stage, see :meth:`solve_stage`. Multi-step schemes (e.g. BDF) keep
        track of the solutions of the previous :attr:`number_of_steps` time levels via
        :meth:`get_step_gridfunctions`, while multi-stage schemes (e.g. Runge-Kutta) evaluate the
        right-hand side at the intermediate stage times given by :attr:`time_of_stages`.

        Concrete implementations are provided per solver, e.g. in :mod:`dream.scalar_transport.time`
        and :mod:`dream.compressible_flow.conservative.time`.
    """

    number_of_steps: int = 1
    number_of_stages: int = 1
    time_of_stages: tuple[float, ...]

    @property
    def dt(self) -> ngs.Parameter:
        return self.root.time.timer.step

    @property
    def t(self) -> ngs.Parameter:
        return self.root.time.timer.t

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        raise NotImplementedError("Overload this method in derived class!")

    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:
        raise NotImplementedError("Overload this method in derived class!")

    def solve_stage(self, stage: int, t0: float) -> typing.Generator[Log, None, None]:
        raise NotImplementedError("Overload this method in derived class!")

    def get_step_key(self, step: int):
        if step == 0:
            return 'n'
        return f"n{step:+d}"

    def get_step_gridfunctions(self, gfu: ngs.GridFunction) -> dict[str, ngs.GridFunction]:
        gfus = [gfu] + [ngs.GridFunction(gfu.space) for _ in range(self.number_of_steps-1)]
        return {self.get_step_key(step): gfu for step, gfu in zip(range(1, 1-self.number_of_steps, -1), gfus)}

    def get_stage_time_steps(self, scaling: float = None, padded: bool = True) -> tuple[float, ...]:

        if scaling is None:
            scaling = self.dt.Get()

        dts = tuple(scaling * (tn - t0) for t0, tn in zip(self.time_of_stages[:-1], self.time_of_stages[1:]))

        if padded:
            return (0.0,) + dts

        return dts

    def get_stage_times(self, scaling: float = None) -> tuple[float, ...]:

        if scaling is None:
            scaling = self.dt.Get()

        return tuple(scaling * t for t in self.time_of_stages)

    def initialize_step_gridfunctions(self, gfus: dict[str, ngs.GridFunction]) -> None:
        self.gfus = {space: self.get_step_gridfunctions(gfu) for space, gfu in gfus.items()}

    def set_initial_conditions(self):
        for gfu in self.gfus.values():
            for old in list(gfu.values())[1:]:
                old.vec.data = gfu['n+1'].vec

    def set_stage_time(self, stage: int, t0: float) -> None:
        self.t.Set(t0 + self.dt.Get() * self.time_of_stages[stage])

    def update_step_gridfunctions(self):
        for gfu in self.gfus.values():

            for step in range(2-self.number_of_steps, 1, 1):
                old_step = self.get_step_key(step)
                new_step = self.get_step_key(step+1)
                gfu[old_step].vec.data = gfu[new_step].vec

    def update_final_stage_solution(self) -> None:
        """ Updates the final stage solution

            This method is needed in case the time scheme is not stiffly accurate.
        """
        pass


class TimeRoutine(Configuration, is_interface=True):
    r""" Base interface for a top-level solution routine driving a :class:`SolverConfiguration`.

        While a :class:`TimeSchemes` discretizes a single time level or stage, a :class:`TimeRoutine`
        orchestrates the full solution process, i.e. it drives the main solution loop, see
        :meth:`start_solution_routine`, manages I/O, and reports solver convergence, see
        :meth:`parse_routine_log`. Available routines are :class:`StationaryRoutine`,
        :class:`TransientRoutine`, :class:`PseudoTimeSteppingRoutine`, and the IMEX routines derived
        from :class:`IMEXTimeRoutine`.
    """

    root: SolverConfiguration

    @property
    def is_stationary(self) -> bool:
        return isinstance(self, StationaryRoutine)

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:
        raise NotImplementedError("Overload this method in derived class!")

    def parse_routine_log(self,
                          it: int | None = None,
                          error: float | None = None,
                          t: float | None = None,
                          stage: int | None = None,
                          cfg: SolverConfiguration = None, **kwargs):
        """ Parse the routine log and return a formatted string. """

        root = cfg
        if cfg is None:
            root = self.root

        msg = f"{self.name} | {root.fem.name} {root.fem.scheme.name}"
        if it is not None:
            msg += f" | it: {it}"
        if error is not None:
            msg += f" | error: {error:8e}"
        if stage is not None:
            msg += f" | stage: {stage}"
        if t is not None:
            msg += f" | t: {t:.6e}"

        return msg


class StationaryRoutine(TimeRoutine):
    r""" Solves a stationary (steady-state) problem.

        No time integration is performed: the routine simply assembles the discrete problem and solves
        it via the nonlinear solver of the underlying scheme, see :meth:`Scheme.solve_stationary`.
    """

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

                if "is_diverged" in log:
                    logger.error("Stationary routine diverged!")
                    break

            yield None
            # Solution routine ends here

            io.save_post_time_routine()
        io.redraw()


class TransientRoutine(TimeRoutine):
    r""" Marches a time-dependent problem forward over a fixed time interval.

        The routine repeatedly advances the solution from :math:`t_n` to :math:`t_{n+1} = t_n + \Delta t`
        over the interval held by its :attr:`timer`, see :meth:`start_solution_routine`. Each time level
        is solved by the scheme of the underlying :class:`SolverConfiguration`, see
        :meth:`TimeSchemes.solve_current_time_level`. In addition to the standard solution routine, this
        class provides :meth:`start_timing_solution_routine` for performance profiling, and
        :meth:`find_stable_time_step` to determine, by bisection, the largest stable time step for an
        explicit scheme.
    """

    name: str = "transient"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "timer": Timer(mesh, root)
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def timer(self) -> Timer:
        r""" Sets the timer holding the time interval, time step and current simulation time.

            :getter: Returns the timer
            :setter: Sets the timer, defaults to :class:`Timer`
        """
        return self._timer

    @timer.setter
    def timer(self, timer: Timer):
        OPTIONS = [Timer]
        self._timer = self._get_configuration_option(timer, OPTIONS, Timer)

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        scheme = self.root.fem.scheme
        timer = self.timer

        timer.reset()

        if reassemble:
            scheme.assemble()

        if hasattr(self.root, 'timestep_controller') and self.root.timestep_controller is not None:
            self.root.timestep_controller.initialize()

        with self.root.io as io:
            io.save_pre_time_routine(self.timer.t.Get())

            # Solution routine starts here
            for rate, tn, t1 in self.timer():

                for log in scheme.solve_current_time_level(tn):
                    logger.info(self.parse_routine_log(**log))

                    if "is_diverged" in log:
                        break

                if "is_diverged" in log:
                    logger.error(f"Transient routine diverged from {tn:.6e} to {t1:.6e}!")
                    break

                logger.debug(f'Solved interval {tn:.6e} to {t1:.6e} \n')
                scheme.update_step_gridfunctions()

                # TODO: Add timestep controller to streams
                if hasattr(self.root, 'timestep_controller') and self.root.timestep_controller is not None:
                    self.root.timestep_controller.process_iteration(iteration=rate)

                yield t1

                io.save_in_time_routine(t1, rate)
                io.redraw(rate)
            # Solution routine ends here

            io.save_post_time_routine(t1, rate)

    def start_timing_solution_routine(self, timings: dict[str, float],
                                      reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        scheme = self.root.fem.scheme
        timer = self.timer

        timer.reset()
        if reassemble:
            scheme.assemble()

        steps = timer.num_steps()

        # Setup timing containers
        timings['total_time'] = 0.0
        for stage in range(1, scheme.number_of_stages + 1):
            timings[f'total_{stage}stage_time'] = 0.0
            timings[f'{stage}_stage_times'] = np.zeros(steps)
            timings[f'{stage}_stage_iterations'] = np.zeros(steps)

        timings['total_final_stage_time'] = 0.0
        timings['final_stage_times'] = np.zeros(steps)

        self.root.fem.solver.solve_update_step = self.root.fem.solver.solve_update_step_timing
        timings['solver_apply'] = []
        timings['solver_assemble'] = []
        timings['solver_solve'] = []
        timings['solver_local'] = []

        # Solution routine starts here
        timings['total_time'] = time.perf_counter()
        for rate, tn, t1 in self.timer():

            for stage in range(1, scheme.number_of_stages + 1):

                start = time.perf_counter()
                for log in scheme.solve_stage(stage, tn):
                    logger.info(self.parse_routine_log(**log))
                    timings['solver_assemble'].append(log.get('solver_assemble', 0))
                    timings['solver_solve'].append(log.get('solver_solve', 0))
                    timings['solver_apply'].append(log.get('solver_apply', 0))
                    timings['solver_local'].append(log.get('solver_local', 0))
                end = time.perf_counter()
                timings[f'total_{stage}stage_time'] += end - start
                timings[f'{stage}_stage_times'][rate-1] = end - start

                if 'it' in log:
                    timings[f'{stage}_stage_iterations'][rate-1] = log['it']

                if "is_diverged" in log:
                    break

            start = time.perf_counter()
            scheme.update_final_stage_solution()
            end = time.perf_counter()
            timings['total_final_stage_time'] += end - start
            timings['final_stage_times'][rate-1] = end - start

            if np.isnan(self.root.fem.gfu.vec).any():
                print("Time routine diverged!")
                break

            yield t1

            scheme.update_step_gridfunctions()

        timings['total_time'] = time.perf_counter() - timings['total_time']

    def find_stable_time_step(
            self, tol: float = 1e-8, process_routine: typing.Callable = None) -> typing.Generator[
            float | None, None, None]:

        scheme = self.root.fem.scheme
        timer = self.timer
        scheme.assemble()

        dt0 = timer.step.Get()
        dts = (0.0, timer.step.Get(), timer.step.Get())

        while True:
            logger.info(f"Starting find stable time step with initial 𝚫t = {timer.step.Get()}")
            timer.reset()

            yield timer.step.Get()

            # Solution routine starts here
            for rate, tn, _ in self.timer():

                for log in scheme.solve_current_time_level(tn):
                    if "is_diverged" in log:
                        break

                if "is_diverged" in log:
                    logger.error("Transient routine diverged!")
                    break

                scheme.update_step_gridfunctions()

            if "is_diverged" in log:

                if process_routine is not None:
                    process_routine(dt=timer.step.Get(), is_stable=False)

                dts = (dts[0], 0.5 * (dts[0] + dts[1]), dts[1])
                timer.step = dts[1]
                logger.info(f"Reducing time step to 𝚫t = {timer.step.Get()}")
            else:

                if process_routine is not None:
                    process_routine(dt=timer.step.Get(), is_stable=True)

                logger.info(f"Stable 𝚫t = {timer.step.Get()}")
                # TODO: Add timestep controller to streams
                # if self.root.timestep_controller is not None:
                #     self.root.timestep_controller.process_iteration(iteration=rate)

                dts = (dts[1], 0.5 * (dts[1] + dts[2]), dts[2])

                if abs(dts[1] - timer.step.Get()) < tol:
                    logger.info(f"Converged to stable time step 𝚫t = {timer.step.Get()}")
                    break

                timer.step = dts[1]

        timer.step = dt0


class PseudoTimeSteppingRoutine(TimeRoutine):
    r""" Marches a stationary problem to steady state using pseudo-time continuation.

        Rather than solving the nonlinear stationary problem directly, an artificial (pseudo) time
        derivative is added and advanced with a time step that is progressively increased, every
        :attr:`increment_at` iterations, by :attr:`increment_factor`, up to :attr:`max_time_step`.
        This continuation strategy improves the robustness of the nonlinear solver, especially when
        starting far from the steady-state solution, see :meth:`solver_iteration_update`.
    """

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
        r""" Sets the timer holding the pseudo-time step and current pseudo-time.

            :getter: Returns the timer
            :setter: Sets the timer, defaults to :class:`Timer`
        """
        return self._timer

    @timer.setter
    def timer(self, timer: Timer):
        OPTIONS = [Timer]
        self._timer = self._get_configuration_option(timer, OPTIONS, Timer)

    @dream_configuration
    def max_time_step(self) -> float:
        r""" Sets the upper bound the pseudo-time step is allowed to grow to.

            :getter: Returns the maximum pseudo-time step
            :setter: Sets the maximum pseudo-time step, defaults to 1.0
        """
        return self._max_time_step

    @max_time_step.setter
    def max_time_step(self, max_time_step: float):
        self._max_time_step = float(max_time_step)

    @dream_configuration
    def increment_at(self) -> int:
        r""" Sets the number of iterations after which the pseudo-time step is increased.

            :getter: Returns the increment interval
            :setter: Sets the increment interval, defaults to 10
        """
        return self._increment_at

    @increment_at.setter
    def increment_at(self, increment_at: int):
        self._increment_at = int(increment_at)

    @dream_configuration
    def increment_factor(self) -> int:
        r""" Sets the multiplicative factor applied to the pseudo-time step at each increment.

            :getter: Returns the increment factor
            :setter: Sets the increment factor, defaults to 10
        """
        return self._increment_factor

    @increment_factor.setter
    def increment_factor(self, increment_factor: int):
        self._increment_factor = int(increment_factor)

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        self.timer.reset()
        dt = self.timer.step.Get()

        scheme = self.root.fem.scheme

        if reassemble:
            scheme.assemble()

        with self.root.io as io:
            io.save_pre_time_routine()

            # Solution routine starts here
            for log in scheme.solve_current_time_level(0.0):
                it = log['it']
                logger.info(self.parse_routine_log(**log))

                if "is_diverged" in log:
                    logger.error("Pseudo time stepping routine diverged!")
                    break

                scheme.update_step_gridfunctions()
                self.solver_iteration_update(it)
                io.save_in_time_routine(it * dt, it)
                io.redraw()

                yield it

            # Solution routine ends here
            io.save_post_time_routine()

        self.timer.step = dt

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


class IMEXTimeRoutine(TimeRoutine, is_interface=True):
    r""" Base interface for a geometry-split implicit-explicit (IMEX) time routine.

        An :class:`IMEXTimeRoutine` couples two independently configured solvers, :attr:`cfg_implicit`
        and :attr:`cfg_explicit`, each holding its own :class:`~dream.solver.SolverConfiguration` and
        mesh. The routine itself does not partition a mesh into implicit/explicit regions; rather, it
        assumes that the two complementary meshes :math:`\mesh^{im}` and :math:`\mesh^{ex}`, with
        interface :math:`\Gamma_i = \mesh^{im} \cap \mesh^{ex}`, have already been constructed and
        assigned to `cfg_implicit` and `cfg_explicit`, respectively, e.g. with the stiff (typically
        small) region assigned to `cfg_implicit` and the remaining, non-stiff region to `cfg_explicit`.

        Stiff regions are typically treated implicitly using a hybridizable discontinuous Galerkin
        (HDG) scheme, while non-stiff regions are treated explicitly using a (standard) discontinuous
        Galerkin (DG) scheme; however, the routine itself only assumes that `cfg_implicit` is solved
        implicitly and `cfg_explicit` explicitly in time, and not any specific spatial discretization.
        In particular, both regions may equally well use a DG discretization, with only the time
        treatment (implicit vs. explicit) differing between them. The two solutions are weakly and
        conservatively coupled across :math:`\Gamma_i` by appropriate interface conditions, while
        temporal synchronization between the implicit and explicit schemes is achieved through additive
        Runge-Kutta (ARK) methods, in which the implicit part is a singly diagonally implicit
        Runge-Kutta (SDIRK) method and the explicit part a standard explicit Runge-Kutta (ERK) method.

        At each global time step, every stage is solved by first advancing the explicit region, see
        :meth:`solve_explicit_stage`, and then the implicit region, see :meth:`solve_implicit_stage`,
        such that the explicit stage uses the implicit solution at the previous stage, and the implicit
        stage uses the just-updated explicit solution, see :meth:`solve_stages`.

        Restricting the (typically more expensive) implicit scheme to a small region, while advancing
        the bulk of the domain with the cheaper explicit scheme, allows the explicit scheme's stable
        time step to increase compared to a fully explicit discretization of the whole domain; this
        increase yields the overall speedup of the IMEX approach, at the cost of the additional implicit
        solve. Two synchronization strategies are provided: :class:`SynchronizedIMEXTimeRoutine`, in
        which the implicit and explicit schemes share the same time step and stage times, and the
        predictor-corrector routines :class:`PCIMEXTimeRoutine`/:class:`LinearPCIMEXTimeRoutine`, which
        allow the explicit scheme to sub-cycle with a smaller, locally defined time step.
    """

    def __init__(self, cfg_implicit: SolverConfiguration, cfg_explicit: SolverConfiguration, **default):
        super().__init__(None, None, **default)

        self.cfg_implicit = cfg_implicit
        self.cfg_explicit = cfg_explicit

    @property
    def gtimer(self) -> Timer:
        """ Global timer for the implicit scheme. """
        return self.cfg_implicit.time.timer

    @property
    def ltimer(self) -> Timer:
        """ Local timer for the explicit scheme. """
        return self.cfg_explicit.time.timer

    @property
    def gscheme(self) -> TimeSchemes:
        """ Global implicit scheme. """
        return self.cfg_implicit.fem.scheme

    @property
    def lscheme(self) -> TimeSchemes:
        """ Local explicit scheme. """
        return self.cfg_explicit.fem.scheme

    def initialize(self, reassemble: bool):

        if not isinstance(self.cfg_implicit.time, TransientRoutine):
            raise TypeError("Implicit configuration must use a transient time routine!")

        if not isinstance(self.cfg_explicit.time, TransientRoutine):
            raise TypeError("Explicit configuration must use a transient time routine!")

        if reassemble:
            self.cfg_implicit.fem.scheme.assemble()
            self.cfg_explicit.fem.scheme.assemble()

        self.gtimer.reset()
        self.ltimer.reset()

        self.gdt = self.gtimer.step.Get()
        self.ldt = self.ltimer.step.Get()

        if self.gdt < self.ldt:
            raise ValueError(f"Global time step {self.gdt} must be greater than local time step {self.ldt}.")

        if self.gscheme.number_of_stages != self.lscheme.number_of_stages:
            raise ValueError(
                f"""Global scheme stages {self.gscheme.number_of_stages}  must be equal to local scheme stages
                {self.lscheme.number_of_stages}.""")

        if self.cfg_implicit.timestep_controller is not None:
            self.cfg_implicit.timestep_controller.initialize()

        if self.cfg_explicit.timestep_controller is not None:
            self.cfg_explicit.timestep_controller.initialize()
        
    def finalize(self):
        self.gtimer.step = self.gdt
        self.ltimer.step = self.ldt

    def solve_explicit_stage(self, stage: int, t0: float):
        raise NotImplementedError("Overload this method in derived class!")

    def solve_implicit_stage(self, stage: int, t0: float):
        raise NotImplementedError("Overload this method in derived class!")

    def start_solution_routine(self, reassemble=True):

        # Initialize predictor corrector routines.
        self.initialize(reassemble)

        # Open IO streams.
        with self.cfg_implicit.io as io_imp, self.cfg_explicit.io as io_exp:

            io_imp.save_pre_time_routine(self.gtimer.t.Get())
            io_exp.save_pre_time_routine(self.ltimer.t.Get())

            # Start the global time-stepping loop.
            for grate, gt0, gt1 in self.gtimer():

                # TODO: Add timestep controller to streams
                if self.cfg_implicit.timestep_controller is not None:
                    self.cfg_implicit.timestep_controller.process_iteration(iteration=grate)

                if self.cfg_explicit.timestep_controller is not None:
                    self.cfg_explicit.timestep_controller.process_iteration(iteration=grate)

                # Sync timers
                self.ltimer.t = self.gtimer.t.Get()
                self.ltimer.interval = (gt0, gt1)

                # Solve all stages.
                for log in self.solve_stages(gt0):
                    if "is_diverged" in log:
                        break

                if "is_diverged" in log:
                    logger.error("IMEX Time routine diverged!")
                    break

                logger.info("Completed all stages for current global time step.\n")

                yield gt1

                self.update_imex_step_gridfunctions()

                io_exp.save_in_time_routine(gt1, grate)
                io_imp.save_in_time_routine(gt1, grate)

            io_exp.save_post_time_routine(gt1, grate)
            io_imp.save_post_time_routine(gt1, grate)

        self.finalize()

    def solve_stages(self, t0: float) -> typing.Generator[Log, None, None]:

        # We loop over each stage and solve explicit regions first, then implicit ones.
        for stage in range(1, self.gscheme.number_of_stages + 1):

            # Step 1: Solve explicit stage.
            logger.info("Solving explicit stage...")
            yield self.solve_explicit_stage(stage, t0)

            # Step 2: Solve the implicit stage.
            logger.info("Solving implicit stage...")
            yield self.solve_implicit_stage(stage, t0)

        self.update_final_stage_solution()

    def solve(self, reassemble: bool = True):
        for t in self.start_solution_routine(reassemble):
            pass

    def update_imex_step_gridfunctions(self):
        self.gscheme.update_step_gridfunctions()

    def update_final_stage_solution(self) -> None:
        """ Updates the final stage solution

            This method is needed in case the time scheme is not stiffly accurate.
        """
        pass


class SynchronizedIMEXTimeRoutine(IMEXTimeRoutine):
    r""" IMEX routine in which the implicit and explicit schemes share the same time step and stage times.

        Both the implicit and explicit schemes advance with the same (global) time step
        :math:`\Delta t` and the additive Runge-Kutta tableaux are synchronized via padding, i.e. the
        stage times coincide, :math:`\overline{c}_{i+1} = c_i`. This is the classical structure of an
        ARK method and is the synchronization strategy used for the geometry-split IMEX schemes
        (ARS-type) developed for compressible flows in :class:`IMEXTimeRoutine`.

        At every stage :math:`i`, the explicit stage solution is computed first, see
        :meth:`solve_explicit_stage`, followed by the implicit stage solution, see
        :meth:`solve_implicit_stage`, both using the previously computed stage solution of the other
        scheme. The final solution update accounts for the first-same-as-last (FSAL) property of the
        explicit scheme and the stiff accuracy of the implicit scheme, see
        :meth:`update_final_stage_solution`.
    """

    def initialize(self, reassemble):
        super().initialize(reassemble)

        if not np.isclose(self.gdt, self.ldt, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Global time step {self.gdt} and local time step {self.ldt} must be equal.")

        if not np.isclose(self.gscheme.time_of_stages, self.lscheme.time_of_stages, rtol=1e-10, atol=1e-10).all():
            raise ValueError(f"Global scheme stage times and local scheme stage times must be equal.")

    def find_stable_time_step(
            self, tol: float = 1e-8, process_routine: typing.Callable = None) -> typing.Generator[
            float | None, None, None]:

        self.initialize(reassemble=True)

        dts = (0.0, self.gtimer.step.Get(), self.gtimer.step.Get())
        while True:
            logger.info(f"Starting find stable time step with initial 𝚫t = {self.gtimer.step.Get()}")
            self.gtimer.reset()
            self.ltimer.reset()

            yield self.gtimer.step.Get()

            # Start the global time-stepping loop.
            for _, gt0, gt1 in self.gtimer():

                # Sync timers
                self.ltimer.t = self.gtimer.t.Get()
                self.ltimer.interval = (gt0, gt1)

                # Solve all stages.
                for log in self.solve_stages(gt0):
                    if "is_diverged" in log:
                        break

                if "is_diverged" in log:
                    break

                self.update_imex_step_gridfunctions()

            if "is_diverged" in log:

                if process_routine is not None:
                    process_routine(dt=self.gtimer.step.Get(), is_stable=False)

                dts = (dts[0], 0.5 * (dts[0] + dts[1]), dts[1])
                self.gtimer.step = dts[1]
                logger.info(f"Reducing time step to 𝚫t = {self.gtimer.step.Get()}")

            else:

                if process_routine is not None:
                    process_routine(dt=self.gtimer.step.Get(), is_stable=True)

                logger.info(f"Stable 𝚫t = {self.gtimer.step.Get()}")

                dts = (dts[1], 0.5 * (dts[1] + dts[2]), dts[2])

                if abs(dts[1] - self.gtimer.step.Get()) < tol:
                    logger.info(f"Converged to stable time step 𝚫t = {self.gtimer.step.Get()}")
                    break

                self.gtimer.step = dts[1]

            self.ltimer.step = self.gtimer.step.Get()

        self.finalize()

    def solve_explicit_stage(self, stage: int, t0: float):

        # Solve explicit stage.
        for log in self.lscheme.solve_stage(stage, t0):
            logger.info(self.parse_routine_log(**log, cfg=self.cfg_explicit))

        return log

    def solve_implicit_stage(self, stage: int, t0: float):

        for log in self.gscheme.solve_stage(stage, t0):
            logger.info(self.parse_routine_log(**log, cfg=self.cfg_implicit))

        return log

    def start_timing_solution_routine(self, timings: dict[str, float], reassemble=True):

        # Initialize predictor corrector routines.
        self.initialize(reassemble)

        steps = self.gtimer.num_steps()

        # Setup timing containers
        timings['total_time'] = 0.0
        for stage in range(1, self.gscheme.number_of_stages + 1):
            timings[f'total_{stage}stage_time_implicit'] = 0.0
            timings[f'{stage}_stage_times_implicit'] = np.zeros(steps)
            timings[f'{stage}_stage_iterations_implicit'] = np.zeros(steps)

            timings[f'total_{stage}stage_time_explicit'] = 0.0
            timings[f'{stage}_stage_times_explicit'] = np.zeros(steps)

        timings['total_final_stage_time_implicit'] = 0.0
        timings['final_stage_times_implicit'] = np.zeros(steps)

        timings['total_final_stage_time_explicit'] = 0.0
        timings['final_stage_times_explicit'] = np.zeros(steps)

        self.cfg_implicit.fem.solver.solve_update_step = self.cfg_implicit.fem.solver.solve_update_step_timing
        timings['solver_apply'] = []
        timings['solver_assemble'] = []
        timings['solver_solve'] = []
        timings['solver_local'] = []

        # Start the global time-stepping loop.
        timings['total_time'] = time.perf_counter()
        for rate, gt0, gt1 in self.gtimer():

            # Sync timers
            self.ltimer.t = self.gtimer.t.Get()
            self.ltimer.interval = (gt0, gt1)

            # Solve all stages.
            # We loop over each stage and solve explicit regions first, then implicit ones.
            for stage in range(1, self.gscheme.number_of_stages + 1):

                # Step 1: Solve explicit stage.
                start = time.perf_counter()
                for log in self.lscheme.solve_stage(stage, gt0):
                    ...
                end = time.perf_counter()
                timings[f'total_{stage}stage_time_explicit'] += end - start
                timings[f'{stage}_stage_times_explicit'][rate-1] = end - start

                # Step 2: Solve the implicit stage.
                start = time.perf_counter()
                for log in self.gscheme.solve_stage(stage, gt0):
                    timings['solver_assemble'].append(log['solver_assemble'])
                    timings['solver_apply'].append(log['solver_apply'])
                    timings['solver_solve'].append(log['solver_solve'])
                    timings['solver_local'].append(log['solver_local'])
                end = time.perf_counter()
                timings[f'total_{stage}stage_time_implicit'] += end - start
                timings[f'{stage}_stage_times_implicit'][rate-1] = end - start
                timings[f'{stage}_stage_iterations_implicit'][rate-1] = log['it']

                if "is_diverged" in log:
                    break

            if "is_diverged" in log:
                print("IMEX Time routine diverged!")
                break

            start = time.perf_counter()
            self.lscheme.update_final_stage_solution()
            end = time.perf_counter()
            timings[f'total_final_stage_time_explicit'] += end - start
            timings['final_stage_times_explicit'][rate-1] = end - start

            start = time.perf_counter()
            self.gscheme.update_final_stage_solution()
            end = time.perf_counter()
            timings[f'total_final_stage_time_implicit'] += end - start
            timings['final_stage_times_implicit'][rate-1] = end - start

            yield gt1

            self.update_imex_step_gridfunctions()

        self.finalize()
        timings['total_time'] = time.perf_counter() - timings['total_time']

    def update_imex_step_gridfunctions(self):
        self.lscheme.update_step_gridfunctions()
        self.gscheme.update_step_gridfunctions()

    def update_final_stage_solution(self):
        self.lscheme.update_final_stage_solution()
        self.gscheme.update_final_stage_solution()


class PCIMEXTimeRoutine(IMEXTimeRoutine):
    r""" IMEX predictor-corrector routine with frozen interface values during the predictor stage.

        Unlike :class:`SynchronizedIMEXTimeRoutine`, the global (implicit) time step :math:`\Delta t`
        need not equal the local (explicit) time step, but must be an integer multiple of it, allowing
        the explicit scheme to sub-cycle with several smaller steps per implicit stage.

        For each implicit stage, a predictor step first advances the implicit solution, see
        :meth:`solve_predictor_stage`, and the explicit scheme is then sub-cycled across the stage
        interval using the predicted interface value frozen at :math:`\vec{U}_n^{im}` for all explicit
        sub-steps, see :meth:`set_predictor_solution`. The implicit (corrector) solution is finally
        recomputed using the now up-to-date explicit interface state, see :meth:`solve_implicit_stage`.
    """

    def initialize(self, reassemble):
        super().initialize(reassemble)

        if not np.isclose(self.gdt / self.ldt, round(self.gdt / self.ldt), rtol=1e-10, atol=1e-10):
            raise ValueError(f"Global time step {self.gdt} must be an integer multiple of local time step {self.ldt}.")

        # Compute local stage time steps based on global ones.
        gstage_dt = self.gscheme.get_stage_time_steps()
        self.lstage_dt = (0.0,) + tuple(stage_dt/np.ceil(stage_dt/self.ldt) for stage_dt in gstage_dt[1:])
        self.gstage_t = self.gscheme.get_stage_times()

        logger.info(f"Global time step: {self.gdt}, Local time step: {self.ldt}")
        logger.info(f"Global stage time steps: {gstage_dt}, Global stage times: {self.gstage_t}")
        logger.info(f"Local stage time steps: {self.lstage_dt}")

    def set_predictor_solution(self):
        pass

    def solve_predictor_stage(self, stage: int, t0: float):
        pass

    def solve_explicit_stage(self, stage: int, t0: float):

        # Solve predictor stage.
        logger.info("Solving predictor stage...")
        self.solve_predictor_stage(stage, t0)

        # Reset stage time in the global scheme.
        self.gscheme.set_stage_time(stage-1, t0)

        # Set local time step for the current stage.
        self.ltimer.step = self.lstage_dt[stage]
        self.ltimer.interval = (t0 + self.gstage_t[stage-1], t0 + self.gstage_t[stage])

        for _, lt0, _ in self.ltimer():

            for lstage in range(1, self.lscheme.number_of_stages + 1):
                for log in self.lscheme.solve_stage(lstage, lt0):
                    logger.info(self.parse_routine_log(**log, cfg=self.cfg_explicit))
                self.set_predictor_solution()

            self.lscheme.update_final_stage_solution()
            self.lscheme.update_step_gridfunctions()

        return log

    def solve_implicit_stage(self, stage: int, t0: float):

        for log in self.gscheme.solve_stage(stage, t0):
            logger.info(self.parse_routine_log(**log, cfg=self.cfg_implicit))

        return log

    def update_final_stage_solution(self):
        self.gscheme.update_final_stage_solution()


class LinearPCIMEXTimeRoutine(PCIMEXTimeRoutine):
    r""" IMEX predictor-corrector routine with linear interpolation of interface values.

        Extends :class:`PCIMEXTimeRoutine` by replacing the frozen interface value during the
        predictor stage with a linear interpolation in time. The predictor stage first solves the
        implicit scheme to obtain a prediction of the interface state at the end of the stage, see
        :meth:`solve_predictor_stage`, and the explicit sub-steps then use values linearly interpolated,
        based on the local sub-step time, between the implicit solution at the start (:math:`\vec{y}_1`)
        and end (:math:`\vec{y}_2`) of the stage, see :meth:`set_predictor_solution`. The corrector step,
        see :meth:`solve_implicit_stage`, resets the implicit solution to :math:`\vec{y}_1` and resolves
        the implicit stage using the now updated explicit solution, improving accuracy compared to the
        frozen-interface predictor of :class:`PCIMEXTimeRoutine`.
    """

    def initialize(self, reassemble):
        super().initialize(reassemble)

        if reassemble:
            self.y1 = self.cfg_implicit.fem.gfu.vec.CreateVector()
            self.y2 = self.cfg_implicit.fem.gfu.vec.CreateVector()

    def set_predictor_solution(self):

        # Get normalized time within the predictor interval.
        t0, tn = self.ltimer.interval
        t_ = (self.ltimer.t.Get() - t0)/(tn - t0)

        # Set the interpolated value in the implicit gfu at the interface.
        self.cfg_implicit.fem.gfu.vec.data = (1.0 - t_) * self.y1 + t_ * self.y2

    def solve_predictor_stage(self, stage: int, t0: float):

        # First, we copy the solution at start of the predictor stage.
        self.y1.data = self.cfg_implicit.fem.gfu.vec

        # Second, we solve for the predicted solution.
        for log in self.gscheme.solve_stage(stage, t0):
            logger.info(self.parse_routine_log(**log, cfg=self.cfg_implicit))

        if stage == self.gscheme.number_of_stages:
            self.gscheme.update_final_stage_solution()

        # Third, we copy the solution at the end of the predictor stage.
        self.y2.data = self.cfg_implicit.fem.gfu.vec
        self.cfg_implicit.fem.gfu.vec.data = self.y1

        return log

    def solve_implicit_stage(self, stage: int, t0: float):

        # Reset the implicit gfu to the initial stage value u^{s}.
        self.cfg_implicit.fem.gfu.vec.data = self.y1

        # Finally, solve for the corrected solution.
        for log in self.gscheme.solve_stage(stage, t0):
            logger.info(self.parse_routine_log(**log, cfg=self.cfg_implicit))

        return log
