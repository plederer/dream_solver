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
       
        # Reserve space for the solution at the old time step (at t^n).
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
        

        for name, cf in self.cfg.blf.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blf += cf.Compile(**compile)
            else:
                self.blf += cf

        if self.cfg.lf:
            self.lf = ngs.LinearForm(self.cfg.fes)
            for name, cf in self.cfg.lf.items():
                logger.debug(f"Adding {name} to the LinearForm!")

                if compile.realcompile:
                    self.lf += cf.Compile(**compile)
                else:
                    self.lf += cf
        self.bsvec = None
        self.cfg.nonlinear_solver.initialize(self.blf, self.bsvec, self.cfg.gfu)

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
        return (3.0*gfus['n+1'] - 4.0*gfus['n'] + gfus['n-1'])/(2.0*self.dt)

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        if normalized:
            return (4.0/3.0)*gfus['n'] - (1.0/3.0)*gfus['n-1']
        return 4.0*gfus['n'] - gfus['n-1']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        if normalized:
            return (2.0/3.0)*self.dt
        return 2.0*self.dt


class DIRKSchemes(TimeSchemes, skip=True):
    r""" All DIRK-type schemes are solving the following HDG problem:
          PDE: M * u_t + f(u,uhat) = 0,
           AE:           g(u,uhat) = 0.

         RK update is: 
            u^{n+1} = u^{n} - dt * sum_{i=1}^{s} b_{i}  * M^{-1} * f(z_i).
         where,
          PDE:  y_i = u^{n} - dt * sum_{j=1}^{i} a_{ij} * M^{-1} * f(z_j),
           AE:    0 = g(z_i).
        Note, 
           z = (y,yhat), are the stage values. Also, we do not explicitly need uhat^{n+1}.

        The residual is defined as R_i = (r_i, rhat_i), as such:
           r_i = M_i * y_i - M_i * u^{n} + (1/a_{ii}) * sum_{j=1}^{i-1} * f(z_j) + f(z_i),
        rhat_i = g(z_i).

        where, 
          M_i = ( 1/(dt*a_{ii}) ) * M.

        Thus, the linearized SOE is based on: 
          N_{i}^{k} * dz_{i}^{k} = -R_{i}( z_{i}^{k} ),
          where the iteration matrix, 
            N_{i}^{k} = dR/dz_i ( z_{i}^{k} ),
                      = { M_i + df/dy_i, df/dyhat_i }
                        {       dg/dy_i, dg/dyhat_i }.

        Implementation is based on the two bilinear forms:

         blf:  { M_i * y_i + f(y_i,yhat_i) }
               {             g(y_i,yhat_i) }  ... needed for iteration matrix + rhs.

         blfs: {             f(y_i,yhat_i) }
               {             g(y_i,yhat_i) }  ... needed for rhs only.

         ... and the (weighted) mass matrix: M_i. 

         This way, 
          1) the iteration matrix N_{i}^{k} is based on blf.
          2) blfs, which depends on the known data from previous stages, is needed for the rhs only.
    """

    def assemble(self) -> None:

        condense = self.cfg.optimizations.static_condensation
        compile = self.cfg.optimizations.compile

        self.blf = ngs.BilinearForm(self.cfg.fes, condense=condense)
        self.blfs = ngs.BilinearForm(self.cfg.fes, condense=condense)
        self.mass = ngs.BilinearForm(self.cfg.fes, symmetric=True)
        self.lf = ngs.LinearForm(self.cfg.fes)

        # Check that a mass matrix is defined in the bilinear form dictionary.
        if "mass" not in self.cfg.blf:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        # Precompute the weighted mass matrix. Note, this is potentially weighted by 1/(dt*aii).
        if compile.realcompile:
            self.mass += self.cfg.blf["mass"].Compile(**compile)
        else:
            self.mass += self.cfg.blf["mass"]

        self.mass.Assemble()

        for name, cf in self.cfg.blf.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blf += cf.Compile(**compile)
            else:
                self.blf += cf

            # Not elegant, but let's add other bilinear form, which does not need a mass item.
            # TODO: change the algebraic transmissibility contraits to separate items in blf,
            # such that we can isolate them here (instead of redundantly computing g(u,uhat) = 0.
            if name != "mass":
                if compile.realcompile:
                    self.blfs += cf.Compile(**compile)
                else:
                    self.blfs += cf
        
        if self.cfg.lf:
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
        raise NotImplementedError()

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        raise NotImplementedError()

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        raise NotImplementedError()

    def update_solution(self, t: float):
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None)-> typing.Generator[int | None, None, None]:
        self.update_solution(t)
        yield None


class SDIRK22(DIRKSchemes):
    r""" Updates the solution via a 2-stage 2nd-order (L-stable) 
         singly diagonal implicit Runge-Kutta (SDIRK).
         Taken from Section 2.6 in [1]. 
    
    [1] Ascher, Uri M., Steven J. Ruuth, and Raymond J. Spiteri. 
        "Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations." 
        Applied Numerical Mathematics 25.2-3 (1997): 151-167. 
    """
    
    name: str = "sdirk22"
    time_levels = ('n', 'n+1')

    def initialize_butcher_tableau(self):
        
        alpha = ngs.sqrt(2.0)/2.0

        self.aii = 1.0 - alpha 
        self.a21 = alpha 
       
        # Time stamps for the stage values between t = [n,n+1].
        self.c1  = 1.0 - alpha
        self.c2  = 1.0 

        # This is possible, because the method is L-stable.
        self.b1  = self.a21
        self.b2  = self.aii


    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
 
        # Reserve space for additional vectors.
        self.mu0 = self.cfg.gfu.vec.CreateVector()
        self.ubar = self.cfg.gfu.vec.CreateVector()
        self.x1 = self.cfg.gfu.vec.CreateVector()

        self.cfg.nonlinear_solver.initialize(self.blf, self.ubar, self.cfg.gfu)
      
    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        u, v = self.TnT[variable]
        gfus = self.gfus[variable].copy()
        gfus['n+1'] = u
       
        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['mass'] = ngs.InnerProduct( ovadt*u, v ) * self.dx[variable]

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def solve_stage(self, t):
        for it in self.cfg.nonlinear_solver.solve(t):
            pass

    def update_solution(self, t: float):
 
        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.cfg.gfu.vec

        # Abbreviations.
        a21 = -self.a21 / self.aii
        
        # Stage: 1.
        self.ubar.data = self.mu0
        self.solve_stage(t) 

        # Stage: 2.
        self.blfs.Apply( self.cfg.gfu.vec, self.x1 )
        self.ubar.data = self.mu0      \
                       + a21 * self.x1
        self.solve_stage(t)

        # NOTE,
        # No need to explicitly update the gfu, since the last stage 
        # corresponds to the value at time: t^{n+1}.




class SDIRK33(DIRKSchemes):
    r""" Updates the solution via a 3-stage 3rd-order (L-stable) 
         singly diagonal implicit Runge-Kutta (SDIRK).
         Taken from Section 2.7 in [1]. 
    
    [1] Ascher, Uri M., Steven J. Ruuth, and Raymond J. Spiteri. 
        "Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations." 
        Applied Numerical Mathematics 25.2-3 (1997): 151-167. 
    """

    name: str = "sdirk33"
    time_levels = ('n', 'n+1')

    def initialize_butcher_tableau(self):

        self.aii =  0.4358665215
        self.a21 =  0.2820667392
        self.a31 =  1.2084966490
        self.a32 = -0.6443631710

        # Time stamps for the stage values between t = [n,n+1].
        self.c1  =  0.4358665215 
        self.c2  =  0.7179332608
        self.c3  =  1.0

        # This is possible, because the method is L-stable.
        self.b1  = self.a31
        self.b2  = self.a32
        self.b3  = self.aii

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
 
        # Reserve space for additional vectors.
        self.mu0 = self.cfg.gfu.vec.CreateVector()
        self.ubar = self.cfg.gfu.vec.CreateVector()
        self.x1 = self.cfg.gfu.vec.CreateVector()
        self.x2 = self.cfg.gfu.vec.CreateVector()

        self.cfg.nonlinear_solver.initialize(self.blf, self.ubar, self.cfg.gfu)
      
    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        u, v = self.TnT[variable]
        gfus = self.gfus[variable].copy()
        gfus['n+1'] = u
 
        # This initializes the coefficients for this scheme.      
        self.initialize_butcher_tableau()
       
        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['mass'] = ngs.InnerProduct( ovadt*u, v ) * self.dx[variable]

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def solve_stage(self, t):
        for it in self.cfg.nonlinear_solver.solve(t):
            pass

    def update_solution(self, t: float):
 
        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.cfg.gfu.vec

        # Abbreviations.
        a21 = -self.a21 / self.aii
        a31 = -self.a31 / self.aii
        a32 = -self.a32 / self.aii

        # Stage: 1.
        self.ubar.data = self.mu0
        self.solve_stage(t) 

        ## Stage: 2.
        self.blfs.Apply( self.cfg.gfu.vec, self.x1 )
        self.ubar.data = self.mu0 + a21 * self.x1
        self.solve_stage(t)

        # Stage: 3.
        self.blfs.Apply( self.cfg.gfu.vec, self.x2 )
        self.ubar.data = self.mu0 + a31 * self.x1 + a32 * self.x2
        self.solve_stage(t)

        # NOTE,
        # No need to explicitly update the gfu, since the last stage 
        # corresponds to the value at time: t^{n+1}.



class SDIRK54(DIRKSchemes):
    r""" Updates the solution via a 5-stage 4th-order (L-stable) 
         singly diagonal implicit Runge-Kutta (SDIRK).
         Taken from Table 6.5 in [1]. 
    
    [1] Wanner, Gerhard, and Ernst Hairer. 
        "Solving ordinary differential equations II."
        Vol. 375. New York: Springer Berlin Heidelberg, 1996. 
    """
    
    name: str = "sdirk54"
    time_levels = ('n', 'n+1')

    def initialize_butcher_tableau(self):

        self.aii =    1.0/4.0
        
        self.a21 =    1.0/2.0 
        
        self.a31 =   17.0/50.0 
        self.a32 =   -1.0/25.0
        
        self.a41 =  371.0/1360.0
        self.a42 = -137.0/2720.0
        self.a43 =   15.0/544.0
        
        self.a51 =   25.0/24.0
        self.a52 =  -49.0/48.0
        self.a53 =  125.0/16.0
        self.a54 =  -85.0/12.0

        # Time stamps for the stage values between t = [n,n+1].
        self.c1  =  1.0/4.0
        self.c2  =  3.0/4.0
        self.c3  = 11.0/20.0
        self.c4  =  1.0/2.0
        self.c5  =  1.0

        # This is possible, because the method is L-stable.
        self.b1  = self.a51
        self.b2  = self.a52
        self.b3  = self.a53
        self.b4  = self.a54
        self.b5  = self.aii


    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
 
        # Reserve space for additional vectors.
        self.mu0 = self.cfg.gfu.vec.CreateVector()
        self.ubar = self.cfg.gfu.vec.CreateVector()
        self.x1 = self.cfg.gfu.vec.CreateVector()
        self.x2 = self.cfg.gfu.vec.CreateVector()
        self.x3 = self.cfg.gfu.vec.CreateVector()
        self.x4 = self.cfg.gfu.vec.CreateVector()

        self.cfg.nonlinear_solver.initialize(self.blf, self.ubar, self.cfg.gfu)
      
    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        u, v = self.TnT[variable]
        gfus = self.gfus[variable].copy()
        gfus['n+1'] = u
        
        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['mass'] = ngs.InnerProduct( ovadt*u, v ) * self.dx[variable]

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def solve_stage(self, t):
        for it in self.cfg.nonlinear_solver.solve(t):
            pass

    def update_solution(self, t: float):
 
        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.cfg.gfu.vec

        # Abbreviations.
        a21 = -self.a21 / self.aii
        
        a31 = -self.a31 / self.aii
        a32 = -self.a32 / self.aii

        a41 = -self.a41 / self.aii
        a42 = -self.a42 / self.aii
        a43 = -self.a43 / self.aii

        a51 = -self.a51 / self.aii
        a52 = -self.a52 / self.aii
        a53 = -self.a53 / self.aii
        a54 = -self.a54 / self.aii

        # Stage: 1.
        self.ubar.data = self.mu0
        self.solve_stage(t) 

        ## Stage: 2.
        self.blfs.Apply( self.cfg.gfu.vec, self.x1 )
        self.ubar.data = self.mu0      \
                       + a21 * self.x1
        self.solve_stage(t)

        # Stage: 3.
        self.blfs.Apply( self.cfg.gfu.vec, self.x2 )
        self.ubar.data = self.mu0      \
                       + a31 * self.x1 \
                       + a32 * self.x2
        self.solve_stage(t)

        # Stage: 4.
        self.blfs.Apply( self.cfg.gfu.vec, self.x3 )
        self.ubar.data = self.mu0      \
                       + a41 * self.x1 \
                       + a42 * self.x2 \
                       + a43 * self.x3
        self.solve_stage(t)

        # Stage: 5.
        self.blfs.Apply( self.cfg.gfu.vec, self.x4 )
        self.ubar.data = self.mu0      \
                       + a51 * self.x1 \
                       + a52 * self.x2 \
                       + a53 * self.x3 \
                       + a54 * self.x4
        self.solve_stage(t)

        # NOTE,
        # No need to explicitly update the gfu, since the last stage 
        # corresponds to the value at time: t^{n+1}.



class dirk43_wso2(DIRKSchemes):
    r""" Updates the solution via a 4-stage 3rd-order (L-stable) 
         diagonal implicit Runge-Kutta (DIRK) with a weak stage order (WSO) of 3.
         Taken from Section 3 in [1]. 
   
    [1] Ketcheson, David I., et al. 
        "DIRK schemes with high weak stage order." 
        Spectral and High Order Methods for Partial Differential Equations (2020): 453.
    """
    name: str = "dirk43_wso2"
    time_levels = ('n', 'n+1')

    def initialize_butcher_tableau(self):

        self.a11 =  0.01900072890
        
        self.a21 =  0.40434605601
        self.a22 =  0.38435717512
        
        self.a31 =  0.06487908412
        self.a32 = -0.16389640295
        self.a33 =  0.51545231222

        self.a41 =  0.02343549374
        self.a42 = -0.41207877888
        self.a43 =  0.96661161281
        self.a44 =  0.42203167233

        # Time stamps for the stage values between t = [n,n+1].
        self.c1  = self.a11 
        self.c2  = self.a21 + self.a22 
        self.c3  = self.a31 + self.a32 + self.a33 
        self.c4  = 1.0

        # This is possible, because the method is L-stable.
        self.b1  = self.a41
        self.b2  = self.a42
        self.b3  = self.a43
        self.b4  = self.a44

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
 
        # Reserve space for additional vectors.
        self.mu0 = self.cfg.gfu.vec.CreateVector()
        self.ubar = self.cfg.gfu.vec.CreateVector()
        self.x1 = self.cfg.gfu.vec.CreateVector()
        self.x2 = self.cfg.gfu.vec.CreateVector()
        self.x3 = self.cfg.gfu.vec.CreateVector()
        
        self.cfg.nonlinear_solver.initialize(self.blf, self.ubar, self.cfg.gfu)
      
    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        u, v = self.TnT[variable]
        gfus = self.gfus[variable].copy()
        gfus['n+1'] = u
       
        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Create a variable parameter, for the diagonal coefficients a_{ii}.
        self.aii = ngs.Parameter(1.0)

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['mass'] = ngs.InnerProduct( ovadt*u, v ) * self.dx[variable]

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def solve_stage(self, t):
        for it in self.cfg.nonlinear_solver.solve(t):
            pass

    def update_solution(self, t: float):
 
        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.cfg.gfu.vec

        
        # Stage: 1.
        self.aii.Set( self.a11 )
        ovaii = 1.0 / self.aii.Get()
        
        self.ubar.data = ovaii * self.mu0
        self.solve_stage(t) 

        # Stage: 2.
        self.aii.Set( self.a22 )
        ovaii =  1.0 / self.aii.Get()
        a21 = -ovaii * self.a21
        
        self.blfs.Apply( self.cfg.gfu.vec, self.x1 )
        self.ubar.data = ovaii * self.mu0 \
                       +   a21 * self.x1
        self.solve_stage(t)

        # Stage: 3.
        self.aii.Set( self.a33 )
        ovaii =  1.0 / self.aii.Get()
        a31 = -self.a31 * ovaii
        a32 = -self.a32 * ovaii
        
        self.blfs.Apply( self.cfg.gfu.vec, self.x2 )
        self.ubar.data = ovaii * self.mu0 \
                       +   a31 * self.x1  \
                       +   a32 * self.x2
        self.solve_stage(t)

        # Stage: 4.
        self.aii.Set( self.a44 )
        ovaii =  1.0 / self.aii.Get()
        a41 = -ovaii * self.a41 
        a42 = -ovaii * self.a42 
        a43 = -ovaii * self.a43 
        
        self.blfs.Apply( self.cfg.gfu.vec, self.x3 )
        self.ubar.data = ovaii * self.mu0 \
                       +   a41 * self.x1  \
                       +   a42 * self.x2  \
                       +   a43 * self.x3
        self.solve_stage(t)

        # NOTE,
        # No need to explicitly update the gfu, since the last stage 
        # corresponds to the value at time: t^{n+1}.



class dirk34_ldd(DIRKSchemes):
    r""" Updates the solution via a 3-stage 4th-order (A-stable) 
         diagonal implicit Runge-Kutta (DIRK) with low-dispersion and dissipation.
         Taken from Table 1 in [1]. 
   
    [1] Nazari, Farshid, Abdolmajid Mohammadian, and Martin Charron. 
        "High-order low-dissipation low-dispersion diagonally implicit Runge–Kutta schemes."
        Journal of Computational Physics 286 (2015): 38-48. 
    """
   
    name: str = "dirk34_ldd"
    time_levels = ('n', 'n+1')

    def initialize_butcher_tableau(self):

        #self.a11 =  0.675592332328701
        #
        #self.a21 =  1.351242940337120
        #self.a22 = -0.851207182169909 
        #
        #self.a31 =  1.351467284887694
        #self.a32 = -1.702697002012658
        #self.a33 =  0.675614858562624

        ## Time stamps for the stage values between t = [n,n+1].
        #self.c1  = self.a11 
        #self.c2  = self.a21 + self.a22 
        #self.c3  = self.a31 + self.a32 + self.a33 

        ## NOTE, this is not L-stable.
        #self.b1  =  1.351467260320785
        #self.b2  = -1.702414526538024
        #self.b3  =  1.350947266217122



        # DEBUGGING
        # uncomment the above and delete this, once debugging is done.
        aii =  0.4358665215
        
        self.a11 =  aii
        self.a21 =  0.2820667392 
        self.a22 =  aii
        self.a31 =  1.2084966490
        self.a32 = -0.6443631710
        self.a33 =  aii
        # Time stamps for the stage values between t = [n,n+1].
        self.c1  =  0.4358665215 
        self.c2  =  0.7179332608
        self.c3  =  1.0
        # This is possible, because the method is L-stable.
        self.b1  = self.a31
        self.b2  = self.a32
        self.b3  = aii




    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
 
        # Reserve space for additional vectors.
        self.u0  = self.cfg.gfu.vec.CreateVector()
        self.mu0 = self.cfg.gfu.vec.CreateVector()
        self.ubar = self.cfg.gfu.vec.CreateVector()
        self.x1 = self.cfg.gfu.vec.CreateVector()
        self.x2 = self.cfg.gfu.vec.CreateVector()
        self.x3 = self.cfg.gfu.vec.CreateVector()
        
        self.minv = ngs.BilinearForm(self.cfg.fes)
        self.minv = self.cfg.linear_solver.inverse(self.mass, self.cfg.fes)

        self.cfg.nonlinear_solver.initialize(self.blf, self.ubar, self.cfg.gfu)
      
    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        u, v = self.TnT[variable]
        gfus = self.gfus[variable].copy()
        gfus['n+1'] = u
       
        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Create a variable parameter, for the diagonal coefficients a_{ii}.
        self.aii = ngs.Parameter(1.0)

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['mass'] = ngs.InnerProduct( ovadt*u, v ) * self.dx[variable]

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def solve_stage(self, t):
        for it in self.cfg.nonlinear_solver.solve(t):
            pass

    def update_solution(self, t: float):
 
        # Book-keep the initial solution at U^n.
        self.u0.data  = self.cfg.gfu.vec

        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.cfg.gfu.vec

        
        # Stage: 1.
        self.aii.Set( self.a11 )
        ovaii = 1.0 / self.aii.Get()
        
        self.ubar.data = ovaii * self.mu0
        self.solve_stage(t) 
        
        # Stage: 2.
        self.aii.Set( self.a22 )
        ovaii =  1.0 / self.aii.Get()
        a21 = -ovaii * self.a21
        
        self.blfs.Apply( self.cfg.gfu.vec, self.x1 )
        self.ubar.data = ovaii * self.mu0 \
                       +   a21 * self.x1
        self.solve_stage(t)

        # Stage: 3.
        self.aii.Set( self.a33 )
        ovaii =  1.0 / self.aii.Get()
        a31 = -self.a31 * ovaii
        a32 = -self.a32 * ovaii
        
        self.blfs.Apply( self.cfg.gfu.vec, self.x2 )
        self.ubar.data = ovaii * self.mu0 \
                       +   a31 * self.x1  \
                       +   a32 * self.x2
        self.solve_stage(t)

        # Spatial term evaluated at stage 3.
        self.blfs.Apply( self.cfg.gfu.vec, self.x3 )

        # Need to explicitly update the solution. 
        # NOTE, the facet variables are not updated (not needed for now)!
        #self.cfg.gfu.vec.data = self.u0           \
        #                      - self.minv *       \
        #                      ( self.b1 * self.x1 \
        #                      + self.b2 * self.x2 \
        #                      + self.b3 * self.x3 )
        
        self.cfg.gfu.vec.data = self.u0 - self.minv*( self.b1*self.x1 + self.b2*self.x2 + self.b3*self.x3 )
        print( "------------------------------------------------------------" )
        #print( self.x2 )
        print( "unew" )
        print( self.cfg.gfu.vec )
        print("u0") 
        print( self.u0 )
        print( self.minv )
        

        # NOTE, we are in debug mode. Here's what we are doing or where we are:
        # 1) we noticed that the IC, u0, always has nan on some of its facets (regardless of the time scheme, even bdf2)
        # 2) we are testing an sdirk33, but using Set/Get ngs.Parameter in this class. Results should be identical to 
        #    the standard SDIRK33 we have running.
        # 3) pardiso print(minv) does not work. We suspect that minv is doing something weird, because if we do not 
        #    explicitly update the solution (line 1219), all works exactly the same as sdirk33.
        # 4) something weird happens, because the first few steps it works, then suddenly it blows up due to nan!?
        # 5) another possible source for this error is that we may have "weird" uhat^{n+1} values, which renders
        #    the inverse of the linearized matrix diverge!?
        # 
        # ... possible fix: 
        # implement a bilinear form with u being the grid function (known), taken as u^{n+1}. And solve for 
        # the uhat^{n+1} variable by linearizing and solving this (possibly) nonlinear system. Example:
        #  blfuhat.Apply( self.cfg.gfu.vec, rhs )
        #  blfuhat.AssembleLinearization( self.cfg.gfu.vec )
        #  duhat = blfuhat.mat.Inverse() * rhs
        #  somehow update only the facet variables using duhat...

        # FIXME




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
