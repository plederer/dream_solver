from __future__ import annotations

import numpy as np
import ngsolve as ngs
import logging
import typing

from dream.config import UniqueConfiguration, InterfaceConfiguration, parameter, configuration, interface, unique, Integrals

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)

from .time import CompressibleTimeSchemes




class ExplicitSchemes(CompressibleTimeSchemes, skip=True):

    def assemble(self) -> None:

        # Ensure that this is a standard DG formulation, otherwise issue an error.
        if self.cfg.fem.method.name != "dg":
            raise TypeError("Only standard DG schemes are compatible with explicit time-stepping schemes.")

        # Check that a mass matrix is indeed defined in the bilinear form dictionary.
        if "mass" not in self.cfg.blf['U']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        compile = self.cfg.optimizations.compile

        # NOTE, we assume that self.lf is not needed here (for efficiency).
        self.blf = ngs.BilinearForm(self.cfg.fes)
        self.rhs = self.cfg.gfu.vec.CreateVector()
        self.minv = ngs.BilinearForm(self.cfg.fes, symmetric=True)

        # Step 1: precompute and store the inverse mass matrix. Note, this is scaled by dt.
        if compile.realcompile:
            self.minv += self.cfg.blf['U']['mass'].Compile(**compile)
        else:
            self.minv += self.cfg.blf['U']['mass']

        # Invert the mass matrix.
        self.minv.Assemble()
        self.minv = self.cfg.linear_solver.inverse(self.minv, self.cfg.fes)

        # Remove the mass matrix item from the bilinear form dictionary, before proceeding.
        self.cfg.blf['U'].pop('mass')

        # Process all items in the relevant bilinear and linear forms.
        self.add_sum_of_integrals(self.blf, self.cfg.blf)

    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:

        u, v = self.TnT[variable]
        gfus = self.gfus[variable].copy()
        gfus['n+1'] = u

        # Add the mass matrix.
        blf[variable]['mass'] = ngs.InnerProduct( u/self.dt, v ) * self.dx[variable]

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
        logger.info(f"time: {t:6e}")
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
    r"""Strong-Stability-Preserving 3rd-order Runge-Kutta.
        This is taken from Section 4.1, Equation 4.2 in [1]. 

    [1] Gottlieb, Sigal, Chi-Wang Shu, and Eitan Tadmor. 
        "Strong stability-preserving high-order time discretization methods." 
        SIAM review 43.1 (2001): 89-112.
    """
    name: str = "ssprk3"
    time_levels = ('n', 'n+1')

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
 
        # Number of stages.
        self.RKnStage = 3
       
        # Reserve space for the solution at the old time step (at t^n).
        self.U0 = self.cfg.gfu.vec.CreateVector()

        # Time stamps for the stage values between t = [n,n+1].
        self.c1  = 0.0
        self.c2  = 1.0
        self.c3  = 0.5
                
        # Define the SSP-RK3 coefficients using alpha and beta (see reference [1]).
        self.alpha20 = 0.75
        self.alpha21 = 0.25
        self.beta21  = 0.25

        self.alpha30 = 1.0/3.0
        self.alpha32 = 2.0/3.0
        self.beta32  = 2.0/3.0

    def update_solution(self, t: float):
    
        # Extract the current solution.
        self.U0.data = self.cfg.gfu.vec
        
        # First stage.
        self.blf.Apply( self.cfg.gfu.vec, self.rhs )
        self.cfg.gfu.vec.data = self.U0 + self.minv * self.rhs

        # Second stage.
        self.blf.Apply( self.cfg.gfu.vec, self.rhs )
        
        # NOTE, avoid 1-liners with dependency on the same read/write data. Can be bugged in NGSolve.
        self.cfg.gfu.vec.data *= self.alpha21
        self.cfg.gfu.vec.data += self.alpha20 * self.U0          \
                              +  self.beta21  * self.minv * self.rhs

        # Third stage.
        self.blf.Apply( self.cfg.gfu.vec, self.rhs )
        # NOTE, avoid 1-liners with dependency on the same read/write data. Can be bugged in NGSolve.
        self.cfg.gfu.vec.data *= self.alpha32 
        self.cfg.gfu.vec.data += self.alpha30 * self.U0          \
                              +  self.beta32  * self.minv * self.rhs


class CRK4(ExplicitSchemes):

    name: str = "crk4"
    time_levels = ('n', 'n+1')

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Number of stages.
        self.RKnStage = 4

        # Define the CRK4 coefficients.
        self.a21 = 0.5
        self.a32 = 0.5
        self.a43 = 1.0

        self.b1 = 1.0/6.0
        self.b2 = 1.0/3.0
        self.b3 = 1.0/3.0
        self.b4 = 1.0/6.0

        self.c1 = 0.0
        self.c2 = 0.5
        self.c3 = 0.5
        self.c4 = 1.0

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
        self.Us.data = self.cfg.gfu.vec + self.a21 * self.K1
        self.blf.Apply( self.Us, self.rhs )
        self.K2.data = self.minv * self.rhs
       
        # Third stage.
        self.Us.data = self.cfg.gfu.vec + self.a32 * self.K2
        self.blf.Apply( self.Us, self.rhs )
        self.K3.data = self.minv * self.rhs
      
        # Fourth stage.
        self.Us.data = self.cfg.gfu.vec + self.K3
        self.blf.Apply( self.Us, self.rhs )
        self.K4.data = self.minv * self.rhs

        # Reconstruct the solution at t^{n+1}.
        self.cfg.gfu.vec.data += self.b1 * self.K1 \
                               + self.b2 * self.K2 \
                               + self.b3 * self.K3 \
                               + self.b4 * self.K4




