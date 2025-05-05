from __future__ import annotations

import numpy as np
import ngsolve as ngs
import logging
import typing
from dream.time import TimeSchemes

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)


class IMEXRKSchemes(TimeSchemes):

    def assemble(self) -> None:

        condense = self.root.optimizations.static_condensation
        compile = self.root.optimizations.compile

        # NOTE, we assume that self.lf is not needed here (for efficiency).
        self.blf = ngs.BilinearForm(self.root.fes, condense=condense)
        self.blfs = ngs.BilinearForm(self.root.fes) 
        self.blfe = ngs.BilinearForm(self.root.fes)
        self.mass = ngs.BilinearForm(self.root.fes, symmetric=True)
        self.rhs = self.root.gfu.vec.CreateVector()
        self.mu0 = self.root.gfu.vec.CreateVector()
        
        # Check that a mass matrix is defined in the bilinear form dictionary.
        if "mass" not in self.root.blf['U']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        # Precompute the weighted mass matrix, with weights: 1/(dt*aii).
        if compile.realcompile:
            self.mass += self.root.blf['U']['mass'].Compile(**compile)
        else:
            self.mass += self.root.blf['U']['mass']
        
        # Assemble the mass matrix once.
        self.mass.Assemble()


        # FIXME, these have to be carefully done, as they are based on the operator splitting assumed.
        
        print( "blf: " )
        # Add the mass matrix and spatial terms (excluding convection) in blf.
        #self.add_sum_of_integrals(self.blf, self.root.blf, 'convection')

        #print( self.root.fem.method.name )
        from dream.compressible.formulations.conservative import HDG, DG, DG_HDG
        print( isinstance( self.root.fem, HDG ) )

        #NOTE, if this is a pure HDG-IMEX, then we only skip the volume convection term in 
        #      the volume equations (tested by V), while we retain the inviscid terms in the
        #      facet equations (tested by Vhat) which are computed implicitly.
         
        # Determine which spaces to iterate over.
        integrals = self.root.blf
        form = self.blf
        pass_terms = 'convection'
        spaces = integrals.keys()
        
        for space in spaces:
            if space not in integrals:
                logger.warning(f"Space '{space}' not found in integrals. Skipping.")
                continue 

            for term, cf in integrals[space].items():
                if term in pass_terms and space == "U":
                    logger.debug(f"Skipping {term}!")
                    # DEBUGGING
                    print( "  skipped: ", term, "[", space, "]" )
                    continue

                logger.debug(f"Adding {term}!")
                # DEBUGGING
                print( "    added: ", term, "[", space, "]" )

                if compile.realcompile:
                    form += cf.Compile(**compile)
                else:
                    form += cf
        # 


        print( "-------------------------------------" )
        print( "blfs: " )
        # Skip the mass matrix and convection contribution in blfs and only use the space for "U".
        self.add_sum_of_integrals(self.blfs, self.root.blf, 'mass', 'convection', fespace='U')
        
        print( "-------------------------------------" )
        print( "blfe: " )
        # Add only the convection part in blfe, as this is handled explicitly in time.
        self.add_sum_of_integrals(self.blfe, self.root.blf, 'mass', 'diffusion', fespace='U')

        # Initialize the nonlinear solver here. Notice, it uses a reference to blf, rhs and gfu.
        self.root.nonlinear_solver.initialize(self.blf, self.rhs, self.root.gfu)



    def add_symbolic_temporal_forms(self,
                                    variable: str,
                                    blf: dict[str, ngs.comp.SumOfIntegrals],
                                    lf: dict[str, ngs.comp.SumOfIntegrals]) -> None:
        raise NotImplementedError()

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def update_solution(self, t: float):
        raise NotImplementedError()

    def get_current_level(self, variable: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[variable]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def solve_stage(self, t, s):
        for it in self.root.nonlinear_solver.solve(t, s):
            pass

    def solve_current_time_level(self, t: float | None = None)-> typing.Generator[int | None, None, None]:
        self.update_solution(t)
        yield None



class IMEXRK_ARS443(IMEXRKSchemes):
 
    name: str = "imex_rk_ars443"
    time_levels = ('n', 'n+1')

    def initialize_butcher_tableau(self):
        
        # Implicit RK coefficients.
        self.aii =  1.0/2.0
        self.a21 =  1.0/6.0
        self.a31 = -1.0/2.0
        self.a32 =  1.0/2.0
        self.a41 =  3.0/2.0
        self.a42 = -3.0/2.0
        self.a43 =  1.0/2.0

        self.b1  = self.a41
        self.b2  = self.a42
        self.b3  = self.a43
        self.b4  = self.aii

        self.c1  = 1.0/2.0
        self.c2  = 2.0/3.0
        self.c3  = 1.0/2.0
        self.c4  = 1.0

        # Explicit RK coefficients.
        self.ae21 =   1.0/2.0
        self.ae31 =  11.0/18.0
        self.ae32 =   1.0/18.0
        self.ae41 =   5.0/6.0
        self.ae42 =  -5.0/6.0
        self.ae43 =   1.0/2.0
        self.ae51 =   1.0/4.0
        self.ae52 =   7.0/4.0
        self.ae53 =   3.0/4.0
        self.ae54 =  -7.0/4.0

        self.be1  = self.ae51
        self.be2  = self.ae52
        self.be3  = self.ae53
        self.be4  = self.ae54

        self.ce2  = self.c1
        self.ce3  = self.c2
        self.ce4  = self.c3
        self.ce5  = self.c4        

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.gfu.vec.CreateVector()
        self.x2 = self.root.gfu.vec.CreateVector()
        self.x3 = self.root.gfu.vec.CreateVector()
        
        self.f1 = self.root.gfu.vec.CreateVector()
        self.f2 = self.root.gfu.vec.CreateVector()
        self.f3 = self.root.gfu.vec.CreateVector()
        self.f4 = self.root.gfu.vec.CreateVector()

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
        blf[variable]['mass'] = ngs.InnerProduct( ovadt*u, v ) * self.dx[variable]

    def update_solution(self, t: float):
 
        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.root.gfu.vec
  
        # Abbreviation.
        ovaii = 1.0/self.aii

        # Stage: 1.
        self.blfe.Apply( self.root.gfu.vec, self.f1 )

        ae21 = ovaii*self.ae21 

        self.rhs.data = self.mu0       \
                      - ae21 * self.f1
        self.solve_stage(t, 1)

        # Stage: 2.
        self.blfe.Apply( self.root.gfu.vec, self.f2 )
        self.blfs.Apply( self.root.gfu.vec, self.x1 )
        
        ae31 = ovaii*self.ae31
        ae32 = ovaii*self.ae32
        a21  = ovaii*self.a21

        self.rhs.data = self.mu0       \
                      - ae31 * self.f1 \
                      - ae32 * self.f2 \
                      -  a21 * self.x1
        self.solve_stage(t, 2)

        # Stage: 3.
        self.blfe.Apply( self.root.gfu.vec, self.f3 )
        self.blfs.Apply( self.root.gfu.vec, self.x2 )
        
        ae41 = ovaii*self.ae41
        ae42 = ovaii*self.ae42
        ae43 = ovaii*self.ae43
        a31  = ovaii*self.a31
        a32  = ovaii*self.a32

        self.rhs.data = self.mu0       \
                      - ae41 * self.f1 \
                      - ae42 * self.f2 \
                      - ae43 * self.f3 \
                      -  a31 * self.x1 \
                      -  a32 * self.x2
        self.solve_stage(t, 3)

        # Stage: 4.
        self.blfe.Apply( self.root.gfu.vec, self.f4 )
        self.blfs.Apply( self.root.gfu.vec, self.x3 )
        
        ae51 = ovaii*self.ae51
        ae52 = ovaii*self.ae52
        ae53 = ovaii*self.ae53
        ae54 = ovaii*self.ae54
        a41  = ovaii*self.a41
        a42  = ovaii*self.a42
        a43  = ovaii*self.a43

        self.rhs.data = self.mu0       \
                      - ae51 * self.f1 \
                      - ae52 * self.f2 \
                      - ae53 * self.f3 \
                      - ae54 * self.f4 \
                      -  a41 * self.x1 \
                      -  a42 * self.x2 \
                      -  a43 * self.x3
        self.solve_stage(t, 4)

        # NOTE,
        # No need to explicitly update the gfu, since the last stage 
        # corresponds to the value at time: t^{n+1}.





