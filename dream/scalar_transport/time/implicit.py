""" Definitions of implicit time marching schemes for a scalar transport equation """
from __future__ import annotations
from dream.config import Integrals
from dream.time import TimeSchemes

import ngsolve as ngs
import logging
import typing

logger = logging.getLogger(__name__)


class ImplicitSchemes(TimeSchemes):

    def assemble_bilinear_form(self, blf) -> None:
        
        # Import locally, to avoid a circular dependency error.
        from dream.scalar_transport.spatial import DG, HDG
        
        # If this is an HDG method, we need to linearize to avoid a bug (with condensation).
        # ... even though we are (redundantly) linearizing a linear problem.
        if isinstance(self.root.fem, HDG):
            blf.AssembleLinearization(self.root.fem.gfu.vec)
        elif isinstance(self.root.fem, DG):
            blf.Assemble()
        else:
            raise ValueError("Can only support (pure) DG or HDG scheme for now.")

    def assemble(self) -> None:

        condense = self.root.optimizations.static_condensation
        compile = self.root.optimizations.compile

        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
     
        # Check that a mass matrix is defined in the bilinear form dictionary.
        if "mass" not in self.root.fem.blf['U']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        # Process all items in the relevant bilinear and linear forms.
        self.add_sum_of_integrals(self.blf, self.root.fem.blf)
     
        # We do this as a wrapper, which decides how to assemble self.blf, which avoids a bug(?) in the assembly.
        self.assemble_bilinear_form(self.blf)

        # Add the integrals to the linear functional, if they exist (all symbolic at this point).
        lf = ngs.LinearForm(self.root.fem.fes)
        self.add_sum_of_integrals(lf, self.root.fem.lf)
        
        # Before proceeding, check whether we require a linear functional or not. 
        if not lf.integrators:
            self.lf = None
        else:
            self.lf = lf
            self.lf.Assemble()

        # Precompute the inverse of the bilinear matrix (if static condensation is false) 
        # or the factorization of the inverse of the Schur complement (if static condensation is True).
        self.binv = self.blf.mat.Inverse(freedofs=self.root.fem.fes.FreeDofs(self.blf.condense), inverse=self.root.linear_solver.name)

        # Precompute the (scaled, with c/dt, where c is some constant) mass matrix.
        self.mass = ngs.BilinearForm(self.root.fem.fes, symmetric=True)

        if self.root.optimizations.compile.realcompile:
            self.mass += self.root.fem.blf['U']['mass'].Compile(**compile)
        else:
            self.mass += self.root.fem.blf['U']['mass']
        
        # And assemble it.
        self.mass.Assemble()

        # Can be avoided, but for readability and given the simplicity of the PDE, we allocate a rhs anyway.
        self.rhs = self.root.fem.gfu.vec.CreateVector()

    def update_solution(self, t: float):
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        logger.info(f"time: {t:6e}")
        self.update_solution(t)
        yield None


class ImplicitEuler(ImplicitSchemes):
    r""" Class responsible for implementing an implicit (backwards-)Euler time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). Namely,

    .. math::
        \widetilde{\bm{M}} \bm{u}^{n+1} + \bm{B} \bm{u}^{n+1} = \widetilde{\bm{M}} \bm{u}^{n},

    where :math:`\widetilde{\bm{M}} = \frac{1}{\delta t} \int_{D} u v\, d\bm{x}` is the modified mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "implicit_euler"
    aliases = ("ie", )
    time_levels = ('n+1',)

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        blf['U'][f'mass'] = ngs.InnerProduct(u/self.dt, v) * ngs.dx
 
    def update_solution(self, t: float):

        # Compute the right-hand side, first.
        self.rhs.data = self.mass.mat * self.root.fem.gfu.vec
        if self.lf is not None:
            self.rhs.data += self.lf.vec

        # Then, solve, depending on whether we static condense or not.
        if self.root.optimizations.static_condensation is True:
            self.rhs.data += self.blf.harmonic_extension_trans * self.rhs
            self.root.fem.gfu.vec.data = self.binv * self.rhs
            self.root.fem.gfu.vec.data += self.blf.harmonic_extension * self.root.fem.gfu.vec
            self.root.fem.gfu.vec.data += self.blf.inner_solve * self.rhs
        else:
            self.root.fem.gfu.vec.data = self.binv * self.rhs


class BDF2(ImplicitSchemes):
    r""" Class responsible for implementing an implicit second-order backward differentiation formula that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), using also the previous solution (:math:`t = t^{n-1}`). Namely,

    .. math::
        \widetilde{\bm{M}} \bm{u}^{n+1} + \bm{B} \bm{u}^{n+1} = \widetilde{\bm{M}} \Big( \frac{4}{3} \bm{u}^{n} - \frac{1}{3} \bm{u}^{n-1}\Big),

    where :math:`\widetilde{\bm{M}} = \frac{3}{2\delta t} \int_{D} u v\, d\bm{x}` is the weighted mass matrix and :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """
    name: str = "bdf2"
    time_levels = ('n-1', 'n', 'n+1')

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        c = 3.0/2.0 
        blf['U'][f'mass'] = ngs.InnerProduct( c*u/self.dt, v) * ngs.dx
     
    def update_solution(self, t: float):
        
        # Scaling factors for the BDF2 scheme.
        f1 = 4.0/3.0
        f2 = 1.0/3.0
        
        # Abbreviation for the three solution steps we book-keep.
        Unp1 = self.gfus['U']['n+1']
        Un   = self.gfus['U']['n']
        Unm1 = self.gfus['U']['n-1']

        # We are computing (and storing U^{n+1}): f1*Un - f2 * U^{n-1}.
        Unp1.vec.data *= f1
        Unp1.vec.data -= f2 * Unm1.vec

        # Compute the right-hand side, first.
        self.rhs.data = self.mass.mat * self.root.fem.gfu.vec
        if self.lf is not None:
            self.rhs.data += self.lf.vec

        # Then, solve, depending on whether we static condense or not.
        if self.root.optimizations.static_condensation is True:
            self.rhs.data += self.blf.harmonic_extension_trans * self.rhs
            self.root.fem.gfu.vec.data = self.binv * self.rhs
            self.root.fem.gfu.vec.data += self.blf.harmonic_extension * self.root.fem.gfu.vec
            self.root.fem.gfu.vec.data += self.blf.inner_solve * self.rhs
        else:
            self.root.fem.gfu.vec.data = self.binv * self.rhs


class DIRKSchemes(TimeSchemes):
    r""" Interface class responsible for configuring a generic diagonally-implicit Runge-Kutta scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), using s-stages. Namely,

    .. math::
        \widetilde{\bm{M}} \bm{u}^{n+1}                        &= \widetilde{\bm{M}} \bm{u}^{n}     - \frac{1}{a_{ii}} \sum_{i=1}^{s}   b_i    \bm{B} \bm{y}_{i},\\
        \widetilde{\bm{M}}_{i} \bm{y}_{i} + \bm{B} \bm{y}_{i}  &= \widetilde{\bm{M}}_{i} \bm{u}^{n} - \frac{1}{a_{ii}} \sum_{j=1}^{i-1} a_{ij} \bm{B} \bm{y}_{j},

    where 

    - :math:`\widetilde{\bm{M}}_{i} = \frac{1}{a_{ii}\delta t} \int_{D} u v\, d\bm{x}` is the *ith* stage-weighted mass matrix.
    - :math:`\widetilde{\bm{M}}` is based on :math:`a_{ii}=1`.
    - :math:`\bm{y}_{i}` is the solution at the *ith* stage.
    - :math:`a_{ij}` and :math:`b_{i}` are taken from the Butcher of a specific scheme.
    
    Finally, :math:`\bm{B}` is the matrix associated with the spatial bilinear form, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_symbolic_spatial_forms` for the implementation.
    """

    def assemble_bilinear_form(self, blf) -> None:
        
        # Import locally, to avoid a circular dependency error.
        from dream.scalar_transport.spatial import DG, HDG
        
        # Book-keep whether this is a DG class (needed for imposing BCs in linear functional).
        self.is_dgfem = False

        # If this is an HDG method, we need to linearize to avoid a bug (with condensation).
        # ... even though we are (redundantly) linearizing a linear problem.
        if isinstance(self.root.fem, HDG):
            blf.AssembleLinearization(self.root.fem.gfu.vec)
        elif isinstance(self.root.fem, DG):
            blf.Assemble()
            self.is_dgfem = True
        else:
            raise ValueError("Can only support (pure) DG or HDG scheme for now.")

    def assemble(self) -> None:

        condense = self.root.optimizations.static_condensation
        compile = self.root.optimizations.compile

        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.blfs = ngs.BilinearForm(self.root.fem.fes, condense=condense)

        # Check that a mass matrix is defined in the bilinear form dictionary.
        self.mass = ngs.BilinearForm(self.root.fem.fes, symmetric=True)
        if "mass" not in self.root.fem.blf['U']:
            raise ValueError("Could not find a mass matrix definition in the bilinear form.")

        # Precompute the weighted mass matrix, with weights: 1/(dt*aii).
        if compile.realcompile:
            self.mass += self.root.fem.blf['U']['mass'].Compile(**compile)
        else:
            self.mass += self.root.fem.blf['U']['mass']

        # Assemble the mass matrix once.
        self.mass.Assemble()
        
        # Add both spatial and mass-matrix terms in blf.
        self.add_sum_of_integrals(self.blf, self.root.fem.blf)
        # Skip the mass matrix contribution in blfs and only use the space for "U".
        self.add_sum_of_integrals(self.blfs, self.root.fem.blf, 'mass', fespace='U')

        # We do this as a wrapper, which decides how to assemble self.blf, which avoids a bug(?) in the assembly.
        self.assemble_bilinear_form(self.blf)

        # Add the integrals to the linear functional, if they exist (all symbolic at this point).
        lf = ngs.LinearForm(self.root.fem.fes)
        self.add_sum_of_integrals(lf, self.root.fem.lf)
        
        # Before proceeding, check whether we require a linear functional or not. 
        if not lf.integrators:
            self.lf = None
        else:
            self.lf = lf
            self.lf.Assemble()

        # Precompute the inverse of the bilinear matrix (if static condensation is false) 
        # or the factorization of the inverse of the Schur complement (if static condensation is True).
        self.binv = self.blf.mat.Inverse(freedofs=self.root.fem.fes.FreeDofs(self.blf.condense), inverse=self.root.linear_solver.name)

    def compute_previous_stage(self, U: ngs.GridFunction, rhs: ngs.BaseVector):
        
        # First, we apply our grid function to get its spatial residual.
        self.blfs.Apply(U.vec, rhs)
        
        # Then, we check whether this is a DG formulation that also 
        # requires a non-periodic BC. If it is, then we need to explicitly
        # account for the BCs conveyed in the linear functional (with -ve sign).
        # Otherwise, in the case of an HDG, we skip this, as the BCs are already
        # imposed on the facets and blfs uses only for the volume solution ['U'].
        if self.is_dgfem and self.lf is not None:
            rhs.data -= self.lf.vec 

    def solve_stage(self, t: float, U: ngs.GridFunction, rhs: ngs.BaseVector):
        if self.root.optimizations.static_condensation is True:
            rhs.data += self.blf.harmonic_extension_trans * rhs
            U.vec.data = self.binv * rhs
            U.vec.data += self.blf.harmonic_extension * U.vec
            U.vec.data += self.blf.inner_solve * rhs
        else:
            U.vec.data = self.binv * rhs

    def update_solution(self, t: float):
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        logger.info(f"time: {t:6e}")
        self.update_solution(t)
        yield None


class SDIRK22(DIRKSchemes):
    r""" Updates the solution via a 2-stage 2nd-order (stiffly-accurate) 
         singly diagonally-implicit Runge-Kutta (SDIRK).
         Taken from Section 2.6 in :cite:`ascher1997implicit`. Its corresponding Butcher tableau is:

    .. math::
        \begin{array}{c|cc}
	        \alpha & \alpha     & 0      \\
	        1    &   1 - \alpha & \alpha \\
            \hline
	             & 1 - \alpha    & \alpha
        \end{array}

    where :math:`\alpha = (2 - \sqrt{2})/2`.
    
    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{u}^{n+1} = \bm{y}_{2}`.
    """
    name: str = "sdirk22"
    time_levels = ('n+1',)

    def assemble(self) -> None:
        super().assemble()
        
        # Reserve space for additional vectors.
        self.mu0 = self.root.fem.gfu.vec.CreateVector()
        self.rhs = self.root.fem.gfu.vec.CreateVector()
        
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

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
       
        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['U']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx

    def update_solution(self, t: float):

        # Initial rhs vector: M*U^n + lf.vec.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec
        if self.lf is not None:
            self.mu0.data += self.lf.vec

        # Abbreviations.
        a21 = -self.a21 / self.aii

        # Stage: 1.
        self.rhs.data = self.mu0 
        self.solve_stage(t, self.root.fem.gfu, self.rhs)

        # Stage: 2.
        self.compute_previous_stage(self.root.fem.gfu, self.rhs)
        self.rhs.data *= a21
        self.rhs.data += self.mu0
        self.solve_stage(t, self.root.fem.gfu, self.rhs)


class SDIRK33(DIRKSchemes):
    r""" Updates the solution via a 3-stage 3rd-order (stiffly-accurate) 
         singly diagonally-implicit Runge-Kutta (SDIRK).
         Taken from Section 2.7 in :cite:`ascher1997implicit`. Its corresponding Butcher tableau is: 

    .. math::
        \begin{array}{c|ccc}
	        0.4358665215 & 0.4358665215 &  0            & 0            \\
	        0.7179332608 & 0.2820667392 &  \phantom{-}0.4358665215 & 0 \\
            1            & 1.2084966490 & -0.6443631710 & 0.4358665215 \\
            \hline
	                     & 1.2084966490 & -0.6443631710 & 0.4358665215
        \end{array}

    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{u}^{n+1} = \bm{y}_{3}`.
    """
    name: str = "sdirk33"
    time_levels = ('n+1',)

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
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.mu0 = self.root.fem.gfu.vec.CreateVector()
        self.rhs = self.root.fem.gfu.vec.CreateVector()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        gfus['n+1'] = u
 
        # This initializes the coefficients for this scheme.      
        self.initialize_butcher_tableau()
       
        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['U']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx

    def update_solution(self, t: float):
 
        # Initial rhs vector: M*U^n + lf.vec.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec
        if self.lf is not None:
            self.mu0.data += self.lf.vec
        
        # Abbreviations.
        a21 = -self.a21 / self.aii
        a31 = -self.a31 / self.aii
        a32 = -self.a32 / self.aii

        # Stage: 1.
        self.rhs.data = self.mu0
        self.solve_stage(t, self.root.fem.gfu, self.rhs)
        
        # Stage: 2.
        self.compute_previous_stage(self.root.fem.gfu, self.x1)
        self.rhs.data = self.mu0 + a21 * self.x1
        self.solve_stage(t, self.root.fem.gfu, self.rhs)

        # Stage: 3.
        self.compute_previous_stage(self.root.fem.gfu, self.x2)
        self.rhs.data = self.mu0 + a31 * self.x1 + a32 * self.x2
        self.solve_stage(t, self.root.fem.gfu, self.rhs)




