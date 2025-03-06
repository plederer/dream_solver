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

    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:
        raise NotImplementedError()

    def assemble(self) -> None:
        raise NotImplementedError()

    def initialize(self):

        self.dx = self.cfg.fem.get_temporal_integrators()
        self.spaces = {}
        self.TnT = {}
        self.gfus = {}

        for variable in self.dx:
            self.spaces[variable] = self.cfg.spaces[variable]
            self.TnT[variable] = self.cfg.TnT[variable]
            self.gfus[variable] = self.initialize_level_gridfunctions(self.cfg.gfus[variable])

    def initialize_level_gridfunctions(self, gfu: ngs.GridFunction) -> dict[str, ngs.GridFunction]:
        gfus = [ngs.GridFunction(gfu.space) for _ in self.time_levels[:-1]] + [gfu]
        return {level: gfu for level, gfu in zip(self.time_levels, gfus)}

    def set_initial_conditions(self):
        for gfu in self.gfus.values():
            for old in list(gfu.values())[:-1]:
                old.vec.data = gfu['n+1'].vec

    def solve_current_time_level(self)-> typing.Generator[int | None, None, None]:
        raise NotImplementedError()

    def update_gridfunctions(self):
        for gfu in self.gfus.values():
            for old, new in zip(self.time_levels[:-1], self.time_levels[1:]):
                gfu[old].vec.data = gfu[new].vec



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # #
# PURELY EXPLICIT SCHEMES ARE DEFINED HERE.
# # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



class ExplicitSchemes(TimeSchemes, skip=True):

    def assemble(self) -> None:

        # Ensure that this is a standard DG formulation, otherwise issue an error.
        # TODO: check via instances, instead of string names.
        if self.cfg.fem.method.name != "dg":
            raise TypeError("Only standard DG schemes are compatible with explicit time-stepping schemes.")

        # Check that a mass matrix is indeed defined in the bilinear form dictionary.
        if "mass" not in self.cfg.blf:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")


        # Check whether we could precompile or do this in runtime.
        compile = self.cfg.optimizations.compile

        self.blf = ngs.BilinearForm(self.cfg.fes)
        self.lf = ngs.LinearForm(self.cfg.fes)
        self.rhs = self.cfg.gfu.vec.CreateVector()
        self.minv = ngs.BilinearForm(self.cfg.fes)

        # Step 1: precompute and store the inverse mass matrix. Note, this is scaled by dt.
        if compile.realcompile:
            self.minv += self.cfg.blf["mass"].Compile(**compile)
        else:
            self.minv += self.cfg.blf["mass"]

        # Invert the mass matrix.
        self.minv.Assemble()
        self.minv = self.cfg.linear_solver.inverse(self.minv, self.cfg.fes)

        # Remove the mass matrix item from the bilinear form dictionary, before proceeding.
        self.cfg.blf.pop("mass")


        # Process all items in the relevant bilinear and linear forms.
        for name, cf in self.cfg.blf.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blf += cf.Compile(**compile)
            else:
                self.blf += cf

        for name, cf in self.cfg.lf.items():
            logger.debug(f"Adding {name} to the LinearForm!")

            if compile.realcompile:
                self.lf += cf.Compile(**compile)
            else:
                self.lf += cf


    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        u, v = self.TnT[variable]
        gfus = self.gfus[variable].copy()
        gfus['n+1'] = u

        # Add the mass matrix.
        blf['mass'] = ngs.InnerProduct( u/self.dt, v ) * self.dx[variable]


    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def update_solution(self, t: float):
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None)-> typing.Generator[int | None, None, None]:
        self.update_solution(t)
        yield None


class ExplicitEuler(ExplicitSchemes):

    name: str = "explicit_euler"
    time_levels = ('n', 'n+1')

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
        

    def update_solution(self, t: float):
    
        # Extract the current solution.
        Un = self.cfg.gfu
        
        self.blf.Apply( Un.vec, self.rhs )
        Un.vec.data += self.minv * self.rhs


class SSPRK3(ExplicitSchemes):

    name: str = "ssprk3"
    time_levels = ('n', 'n+1')

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
 
        # Number of stages.
        self.RKnStage = 3
       
        # Reserve space for the old time step (at t^n).
        self.U0 = self.cfg.gfu.vec.CreateVector()

        # Define the SSP-RK3 coefficients.
        self.RKa = [1.0, 0.75, 1.0/3.0]
        self.RKb = [1.0, 0.25, 2.0/3.0]
        self.RKc = [0.0, 1.00, 0.5]

    def update_solution(self, t: float):
    
        # Extract the current solution.
        self.U0.data = self.cfg.gfu.vec
        
        # Abbreviate the current solution.
        Un = self.cfg.gfu
        
        # The first stage is handled separately, for efficiency.
        self.blf.Apply( Un.vec, self.rhs )
        Un.vec.data = self.U0 + self.minv * self.rhs

        # Second stage.
        self.blf.Apply( Un.vec, self.rhs )
        Un.vec.data = 0.25 * Un.vec  \
                    + 0.75 * self.U0 \
                    + 0.25 * self.minv * self.rhs

        # Abbreviations.
        tov3 = 2.0/3.0
        oov3 = 1.0/3.0

        # Third stage.
        self.blf.Apply( Un.vec, self.rhs )
        Un.vec.data = tov3 * Un.vec  \
                    + oov3 * self.U0 \
                    + tov3 * self.minv * self.rhs


class CRK4(ExplicitSchemes):

    name: str = "crk4"
    time_levels = ('n', 'n+1')

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Number of stages.
        self.RKnStage = 4

        # Define the CRK4 coefficients.
        self.RKa = [0.0,     0.5,     0.5,     1.0]
        self.RKb = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]
        self.RKc = [0.0,     0.5,     0.5,     1.0]

        # Reserve space for the tentative solution.
        self.K1 = self.cfg.gfu.vec.CreateVector()
        self.K2 = self.cfg.gfu.vec.CreateVector()
        self.K3 = self.cfg.gfu.vec.CreateVector()
        self.K4 = self.cfg.gfu.vec.CreateVector()
        self.Us = self.cfg.gfu.vec.CreateVector()

    def update_solution(self, t: float):
   
        # First stage.
        self.blf.Apply( self.cfg.gfu.vec, self.rhs )
        self.K1.data = self.minv * self.rhs

        # Second stage.
        self.Us.data = self.cfg.gfu.vec + 0.5 * self.K1
        self.blf.Apply( self.Us, self.rhs )
        self.K2.data = self.minv * self.rhs
       
        # Third stage.
        self.Us.data = self.cfg.gfu.vec + 0.5 * self.K2
        self.blf.Apply( self.Us, self.rhs )
        self.K3.data = self.minv * self.rhs
      
        # Fourth stage.
        self.Us.data = self.cfg.gfu.vec + self.K3
        self.blf.Apply( self.Us, self.rhs )
        self.K4.data = self.minv * self.rhs
    
        # Abbreviations.
        ov6 = 1.0/6.0
        ov3 = 1.0/3.0

        # Reconstruct the solution at t^{n+1}.
        self.cfg.gfu.vec.data += ( ov6*self.K1 + ov3*self.K2 + ov3*self.K3 + ov6*self.K4 )



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # #
# PURELY IMPLICIT SCHEMES ARE DEFINED HERE.
# # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



class ImplicitSchemes(TimeSchemes, skip=True):

    def assemble(self) -> None:

        condense = self.cfg.optimizations.static_condensation
        compile = self.cfg.optimizations.compile

        self.blf = ngs.BilinearForm(self.cfg.fes, condense=condense)
        self.lf = ngs.LinearForm(self.cfg.fes)

        for name, cf in self.cfg.blf.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blf += cf.Compile(**compile)
            else:
                self.blf += cf

        for name, cf in self.cfg.lf.items():
            logger.debug(f"Adding {name} to the LinearForm!")

            if compile.realcompile:
                self.lf += cf.Compile(**compile)
            else:
                self.lf += cf

        self.cfg.nonlinear_solver.initialize(self.blf, self.lf, self.cfg.gfu)

    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        u, v = self.TnT[variable]
        gfus = self.gfus[variable].copy()
        gfus['n+1'] = u

        blf[f'time'] = ngs.InnerProduct(self.get_time_derivative(gfus), v) * self.dx[variable]

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        raise NotImplementedError()

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None)-> typing.Generator[int | None, None, None]:
        for it in self.cfg.nonlinear_solver.solve(t):
            yield it


class ImplicitEuler(ImplicitSchemes):

    name: str = "implicit_euler"
    aliases = ("ie", )
    time_levels = ('n', 'n+1')

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return (gfus['n+1'] - gfus['n'])/self.dt

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt


class BDF2(ImplicitSchemes):

    name: str = "bdf2"

    time_levels = ('n-1', 'n', 'n+1')

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return (3*gfus['n+1'] - 4*gfus['n'] + gfus['n-1'])/(2*self.dt)

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        if normalized:
            return 4/3*gfus['n'] - 1/3*gfus['n-1']
        return 4 * gfus['n'] - gfus['n-1']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        if normalized:
            return 2/3*self.dt
        return 2*self.dt



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # #
# HYBRID IMEX SCHEMES ARE DEFINED HERE.
# # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # #
# TODO: FINISH ME!
# # # # # # # # # #





class TimeConfig(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def is_stationary(self) -> bool:
        return isinstance(self, StationaryConfig)

    def assemble(self) -> None:
        raise NotImplementedError("Symbolic Forms not implemented!")

    def add_symbolic_temporal_forms(
            self, blf: dict[str, ngs.comp.SumOfIntegrals],
            lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:
        pass

    def initialize(self):
        pass

    def set_initial_conditions(self):
        self.cfg.fem.set_initial_conditions()

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:
        raise NotImplementedError("Solution Routine not implemented!")


class StationaryConfig(TimeConfig):

    name: str = "stationary"

    def assemble(self) -> None:

        condense = self.cfg.optimizations.static_condensation
        compile = self.cfg.optimizations.compile

        self.blf = ngs.BilinearForm(self.cfg.fes, condense=condense)
        self.lf = ngs.LinearForm(self.cfg.fes)

        for name, cf in self.cfg.blf.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blf += cf.Compile(**compile)
            else:
                self.blf += cf

        for name, cf in self.cfg.lf.items():
            logger.debug(f"Adding {name} to the LinearForm!")

            if compile.realcompile:
                self.lf += cf.Compile(**compile)
            else:
                self.lf += cf

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        if reassemble:
            self.assemble()

        with self.cfg.io as io:
            io.save_pre_time_routine()

            # Solution routine starts here
            self.solve()
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

    def assemble(self):
        self.scheme.assemble()

    def add_symbolic_temporal_forms(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]):
        self.cfg.fem.add_symbolic_temporal_forms(blf, lf)

    def initialize(self):
        super().initialize()
        self.scheme.initialize()

    def set_initial_conditions(self):
        super().set_initial_conditions()
        self.scheme.set_initial_conditions()

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        if reassemble:
            self.assemble()

        with self.cfg.io as io:
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

    def add_symbolic_temporal_forms(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]):
        self.cfg.fem.add_symbolic_temporal_forms(blf, lf)

    def assemble(self):
        self.scheme.assemble()

    def initialize(self):
        super().initialize()
        self.scheme.initialize()

    def start_solution_routine(self, reassemble: bool = True) -> typing.Generator[float | None, None, None]:

        if reassemble:
            self.scheme.assemble()

        with self.cfg.io as io:
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

    scheme: ImplicitEuler | BDF2
    timer: Timer
    max_time_step: float
    increment_at: int
    increment_factor: int
