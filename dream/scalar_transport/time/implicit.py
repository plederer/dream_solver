from __future__ import annotations

import numpy as np
import ngsolve as ngs
import logging
import typing

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)

from dream.time import TimeSchemes



class ImplicitSchemes(TimeSchemes):

    def assemble(self) -> None:

        condense = self.root.optimizations.static_condensation
        compile = self.root.optimizations.compile

        self.blf = ngs.BilinearForm(self.root.fes, condense=condense)
     
        # Check that a mass matrix is defined in the bilinear form dictionary.
        if "mass" not in self.root.blf['U']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        # Precompute the (scaled, with 1/dt) mass matrix.
        self.mass = ngs.BilinearForm(self.root.fes, symmetric=True)

        if compile.realcompile:
            self.mass += self.root.blf['U']['mass'].Compile(**compile)
        else:
            self.mass += self.root.blf['U']['mass']
        
        # And assemble it.
        self.mass.Assemble()

        # Remove the mass matrix item from the bilinear form dictionary, before proceeding.
        #self.root.blf['U'].pop('mass')

        # Process all items in the relevant bilinear and linear forms.
        self.add_sum_of_integrals(self.blf, self.root.blf)
      
        # Recall, this is a linear equation, so we can precompute the matrix
        # associated with this bilinear form.
        self.blf.Assemble()
        
        # Next, let's compute the inverse of this bilinear form matrix.
        binv = self.root.linear_solver.inverse(self.blf, self.root.fes)
        
        # Finally, let's premultiply this by the (scaled) mass matrix.
        self.binv = binv @ self.mass.mat


    def add_symbolic_temporal_forms(self, 
                                    space: str, 
                                    blf: Integrals, 
                                    lf: Integrals) -> None:

        u, v = self.TnT[space]
        gfus = self.gfus[space].copy()
        gfus['n+1'] = u

        blf[space][f'mass'] = ngs.InnerProduct(u/self.dt, v) * self.dx[space]

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        raise NotImplementedError()

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        raise NotImplementedError()

    def update_solution(self, t: float):
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        logger.info(f"time: {t:6e}")
        self.update_solution(t)
        yield None


class ImplicitEuler(ImplicitSchemes):

    name: str = "implicit_euler"
    aliases = ("ie", )
    time_levels = ('n', 'n+1')


    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()
       

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def update_solution(self, t: float):
        self.root.gfu.vec.data = self.binv * self.root.gfu.vec







