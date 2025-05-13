from __future__ import annotations
from dream.config import Integrals
from dream.time import TimeSchemes

import ngsolve as ngs
import logging
import typing

logger = logging.getLogger(__name__)


class ImplicitSchemes(TimeSchemes):

    def assemble(self) -> None:

        condense = self.root.optimizations.static_condensation

        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.lf = ngs.LinearForm(self.root.fem.fes)

        self.add_sum_of_integrals(self.blf, self.root.fem.blf)
        self.add_sum_of_integrals(self.lf, self.root.fem.lf)

        self.root.nonlinear_solver.initialize(self.blf, self.lf.vec, self.root.fem.gfu)

        # NOTE
        # Pehaps its better to avoid lf, since it is empty, and specify the 2nd.
        # argument in nonlinear_solver.initialize() as "None". That way, we
        # guarantee avoiding additional unecessary memory. For example:
        # self.root.nonlinear_solver.initialize(self.blf, None, self.root.gfu)

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        gfus['n+1'] = u

        blf['U'][f'time'] = ngs.InnerProduct(self.get_time_derivative(gfus), v) * ngs.dx

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def get_current_level(self, space: str, normalized: bool = False) -> ngs.CF:
        raise NotImplementedError()

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        raise NotImplementedError()

    def solve_current_time_level(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
        for it in self.root.nonlinear_solver.solve(t):
            yield it


class ImplicitEuler(ImplicitSchemes):

    name: str = "implicit_euler"
    time_levels = ('n', 'n+1')

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return (gfus['n+1'] - gfus['n'])/self.dt

    def get_current_level(self, space: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[space]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt


class BDF2(ImplicitSchemes):

    name: str = "bdf2"
    time_levels = ('n-1', 'n', 'n+1')

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return (3.0*gfus['n+1'] - 4.0*gfus['n'] + gfus['n-1'])/(2.0*self.dt)

    def get_current_level(self, space: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[space]
        if normalized:
            return (4.0/3.0)*gfus['n'] - (1.0/3.0)*gfus['n-1']
        return 4.0*gfus['n'] - gfus['n-1']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        if normalized:
            return (2.0/3.0)*self.dt
        return 2.0*self.dt


class DIRKSchemes(TimeSchemes):
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
               {                        0  }  ... needed for rhs only.

         ... and the (weighted) mass matrix: M_i. 

         This way, 
          1) the iteration matrix N_{i}^{k} is based on blf.
          2) blfs, which depends on the known data from previous stages, is needed for the rhs only.
    """

    def assemble(self) -> None:

        condense = self.root.optimizations.static_condensation
        compile = self.root.optimizations.compile

        # NOTE, we assume that self.lf is not needed here (for efficiency).
        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.blfs = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.mass = ngs.BilinearForm(self.root.fem.fes, symmetric=True)
        self.rhs = self.root.fem.gfu.vec.CreateVector()
        self.mu0 = self.root.fem.gfu.vec.CreateVector()

        # Check that a mass matrix is defined in the bilinear form dictionary.
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

        # Initialize the nonlinear solver here. Notice, it uses a reference to blf, rhs and gfu.
        self.root.nonlinear_solver.initialize(self.blf, self.rhs, self.root.fem.gfu)

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        raise NotImplementedError()

    def update_solution(self, t: float):
        raise NotImplementedError()

    def get_current_level(self, space: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[space]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt

    def solve_stage(self, t, s):
        for it in self.root.nonlinear_solver.solve(t, s):
            pass

    def solve_current_time_level(self, t: float | None = None) -> typing.Generator[int | None, None, None]:
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
        self.c1 = 1.0 - alpha
        self.c2 = 1.0

        # This is possible, because the method is L-stable.
        self.b1 = self.a21
        self.b2 = self.aii

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()

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

        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec

        # Abbreviations.
        a21 = -self.a21 / self.aii

        # Stage: 1.
        self.rhs.data = self.mu0
        self.solve_stage(t, 1)

        # Stage: 2.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x1)
        self.rhs.data = self.mu0 + a21 * self.x1
        self.solve_stage(t, 2)

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

        self.aii = 0.4358665215
        self.a21 = 0.2820667392
        self.a31 = 1.2084966490
        self.a32 = -0.6443631710

        # Time stamps for the stage values between t = [n,n+1].
        self.c1 = 0.4358665215
        self.c2 = 0.7179332608
        self.c3 = 1.0

        # This is possible, because the method is L-stable.
        self.b1 = self.a31
        self.b2 = self.a32
        self.b3 = self.aii

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()

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

        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec

        # Abbreviations.
        a21 = -self.a21 / self.aii
        a31 = -self.a31 / self.aii
        a32 = -self.a32 / self.aii

        # Stage: 1.
        self.rhs.data = self.mu0
        self.solve_stage(t, 1)

        # Stage: 2.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x1)
        self.rhs.data = self.mu0 + a21 * self.x1
        self.solve_stage(t, 2)

        # Stage: 3.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x2)
        self.rhs.data = self.mu0 + a31 * self.x1 + a32 * self.x2
        self.solve_stage(t, 3)

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

        self.aii = 1.0/4.0

        self.a21 = 1.0/2.0

        self.a31 = 17.0/50.0
        self.a32 = -1.0/25.0

        self.a41 = 371.0/1360.0
        self.a42 = -137.0/2720.0
        self.a43 = 15.0/544.0

        self.a51 = 25.0/24.0
        self.a52 = -49.0/48.0
        self.a53 = 125.0/16.0
        self.a54 = -85.0/12.0

        # Time stamps for the stage values between t = [n,n+1].
        self.c1 = 1.0/4.0
        self.c2 = 3.0/4.0
        self.c3 = 11.0/20.0
        self.c4 = 1.0/2.0
        self.c5 = 1.0

        # This is possible, because the method is L-stable.
        self.b1 = self.a51
        self.b2 = self.a52
        self.b3 = self.a53
        self.b4 = self.a54
        self.b5 = self.aii

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.x3 = self.root.fem.gfu.vec.CreateVector()
        self.x4 = self.root.fem.gfu.vec.CreateVector()

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

        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec

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
        self.rhs.data = self.mu0
        self.solve_stage(t, 1)

        # Stage: 2.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x1)
        self.rhs.data = self.mu0      \
            + a21 * self.x1
        self.solve_stage(t, 2)

        # Stage: 3.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x2)
        self.rhs.data = self.mu0      \
            + a31 * self.x1 \
            + a32 * self.x2
        self.solve_stage(t, 3)

        # Stage: 4.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x3)
        self.rhs.data = self.mu0      \
            + a41 * self.x1 \
            + a42 * self.x2 \
            + a43 * self.x3
        self.solve_stage(t, 4)

        # Stage: 5.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x4)
        self.rhs.data = self.mu0      \
            + a51 * self.x1 \
            + a52 * self.x2 \
            + a53 * self.x3 \
            + a54 * self.x4
        self.solve_stage(t, 5)

        # NOTE,
        # No need to explicitly update the gfu, since the last stage
        # corresponds to the value at time: t^{n+1}.


class DIRK43_WSO2(DIRKSchemes):
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

        self.a11 = 0.01900072890

        self.a21 = 0.40434605601
        self.a22 = 0.38435717512

        self.a31 = 0.06487908412
        self.a32 = -0.16389640295
        self.a33 = 0.51545231222

        self.a41 = 0.02343549374
        self.a42 = -0.41207877888
        self.a43 = 0.96661161281
        self.a44 = 0.42203167233

        # Time stamps for the stage values between t = [n,n+1].
        self.c1 = self.a11
        self.c2 = self.a21 + self.a22
        self.c3 = self.a31 + self.a32 + self.a33
        self.c4 = 1.0

        # This is possible, because the method is L-stable.
        self.b1 = self.a41
        self.b2 = self.a42
        self.b3 = self.a43
        self.b4 = self.a44

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.x3 = self.root.fem.gfu.vec.CreateVector()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        gfus['n+1'] = u

        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Create a variable parameter, for the diagonal coefficients a_{ii}.
        self.aii = ngs.Parameter(1.0)

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['U']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx

    def update_solution(self, t: float):

        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec

        # Stage: 1.
        self.aii.Set(self.a11)
        ovaii = 1.0 / self.aii.Get()

        self.rhs.data = ovaii * self.mu0
        self.solve_stage(t, 1)

        # Stage: 2.
        self.aii.Set(self.a22)
        ovaii = 1.0 / self.aii.Get()
        a21 = -ovaii * self.a21

        self.blfs.Apply(self.root.fem.gfu.vec, self.x1)
        self.rhs.data = ovaii * self.mu0 \
            + a21 * self.x1
        self.solve_stage(t, 2)

        # Stage: 3.
        self.aii.Set(self.a33)
        ovaii = 1.0 / self.aii.Get()
        a31 = -self.a31 * ovaii
        a32 = -self.a32 * ovaii

        self.blfs.Apply(self.root.fem.gfu.vec, self.x2)
        self.rhs.data = ovaii * self.mu0 \
            + a31 * self.x1  \
            + a32 * self.x2
        self.solve_stage(t, 3)

        # Stage: 4.
        self.aii.Set(self.a44)
        ovaii = 1.0 / self.aii.Get()
        a41 = -ovaii * self.a41
        a42 = -ovaii * self.a42
        a43 = -ovaii * self.a43

        self.blfs.Apply(self.root.fem.gfu.vec, self.x3)
        self.rhs.data = ovaii * self.mu0 \
            + a41 * self.x1  \
            + a42 * self.x2  \
            + a43 * self.x3
        self.solve_stage(t, 4)

        # NOTE,
        # No need to explicitly update the gfu, since the last stage
        # corresponds to the value at time: t^{n+1}.


class DIRK34_LDD(DIRKSchemes):
    r""" Updates the solution via a 3-stage 4th-order (A-stable) 
         diagonal implicit Runge-Kutta (DIRK) with low-dispersion and dissipation.
         Taken from Table A.1 in [1]. 

    [1] Najafi-Yazdi, Alireza, and Luc Mongeau. 
        "A low-dispersion and low-dissipation implicit Runge–Kutta scheme." 
        Journal of computational physics 233 (2013): 315-323.
    """
    name: str = "dirk34_ldd"
    time_levels = ('n', 'n+1')

    def initialize_butcher_tableau(self):

        self.a11 = 0.377847764031163

        self.a21 = 0.385232756462588
        self.a22 = 0.461548399939329

        self.a31 = 0.675724855841358
        self.a32 = -0.061710969841169
        self.a33 = 0.241480233100410

        # Time stamps for the stage values between t = [n,n+1].
        self.c1 = 0.257820901066211
        self.c2 = 0.434296446908075
        self.c3 = 0.758519768667167

        # NOTE, this is not L-stable.
        self.b1 = 0.750869573741408
        self.b2 = -0.362218781852651
        self.b3 = 0.611349208111243

    def assemble(self) -> None:

        # Call the parent's assemble, in case additional checks need be done first.
        super().assemble()

        # Reserve space for additional vectors.
        self.u0 = self.root.fem.gfu.vec.CreateVector()
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.x3 = self.root.fem.gfu.vec.CreateVector()

        # Precompute the mass matrix of the volume elements.
        self.minv = self.root.linear_solver.inverse(self.mass, self.root.fes)

        # Compute the inverse mass matrix for the facets only. Needed to update uhat^{n+1}.
        # NOTE, this assumes uhat^{n+1} = 0.5*( U_L^{n+1} + U_R^{n+1} ).

        # Step 1: extract the relevant space for the facets.
        gfu = self.root.fem.gfus['Uhat']
        fes = self.root.fem.gfus['Uhat'].space
        uhat, vhat = fes.TnT()

        # Step 2: define the facet "mass" matrix term.
        blfh = ngs.BilinearForm(fes)
        blfh += uhat*vhat*ngs.dx(element_boundary=True)

        # Step 3: define the rhs for the facet solution, needed to approximate uhat^{n+1}.
        self.f_uhat = ngs.LinearForm(fes)
        self.f_uhat += self.root.fem.gfus['U'] * vhat * ngs.dx(element_boundary=True)

        # Step 4: compute the inverse of the mass matrix on the facets.
        blfh.Assemble()
        self.minv_uhat = blfh.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky")

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        gfus['n+1'] = u

        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Create a variable parameter, for the diagonal coefficients a_{ii}.
        self.aii = ngs.Parameter(1.0)

        # Abbreviation.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['U']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx

    def update_solution(self, t: float):

        # Book-keep the initial solution at U^n.
        self.u0.data = self.root.fem.gfu.vec

        # Initial vector: M*U^n.
        self.mu0.data = self.mass.mat * self.root.fem.gfu.vec

        # Stage: 1.
        self.aii.Set(self.a11)
        ovaii = 1.0 / self.aii.Get()

        self.rhs.data = ovaii * self.mu0
        self.solve_stage(t, 1)

        # Stage: 2.
        self.aii.Set(self.a22)
        ovaii = 1.0 / self.aii.Get()
        a21 = -ovaii * self.a21

        self.blfs.Apply(self.root.fem.gfu.vec, self.x1)
        self.rhs.data = ovaii * self.mu0 \
            + a21 * self.x1
        self.solve_stage(t, 2)

        # Stage: 3.
        self.aii.Set(self.a33)
        ovaii = 1.0 / self.aii.Get()
        a31 = -self.a31 * ovaii
        a32 = -self.a32 * ovaii

        self.blfs.Apply(self.root.fem.gfu.vec, self.x2)
        self.rhs.data = ovaii * self.mu0 \
            + a31 * self.x1  \
            + a32 * self.x2
        self.solve_stage(t, 3)

        # Spatial term evaluated at stage 3.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x3)

        # Need to explicitly update the solution.
        self.root.fem.gfu.vec.data = self.u0           \
            - self.minv *       \
            (self.b1 * self.x1
             + self.b2 * self.x2
             + self.b3 * self.x3)

        # We assumbe f, because it uses the (volume) solution at u^{n+1}.
        self.f_uhat.Assemble()
        self.root.fem.gfus['Uhat'].vec.data = self.minv_uhat * self.f_uhat.vec
