""" Definitions of implicit time integration schemes for conservative methods. """
from __future__ import annotations
from dream.config import Integrals, Log
from dream.time import TimeSchemes

import ngsolve as ngs
import typing


class ImplicitSchemes(TimeSchemes):

    def assemble(self) -> None:

        condense = self.root.fem.static_condensation

        self.blf = ngs.BilinearForm(self.root.fem.fes, condense=condense)
        self.add_sum_of_integrals(self.blf, self.root.fem.blf, 'implicit bilinear form')

        rhs = None
        if self.root.fem.lf:
            self.lf = ngs.LinearForm(self.root.fem.fes)
            self.add_sum_of_integrals(self.lf, self.root.fem.lf, 'linear form')
            self.lf.Assemble()
            rhs = self.lf.vec

        self.root.fem.solver.initialize_nonlinear_routine(self.blf, self.root.fem.gfu, rhs)

        # NOTE
        # Pehaps its better to avoid lf, since it is empty, and specify the 3nd. 
        # argument in fem.solver.initialize_nonlinear_routine as "None". That way, we 
        # guarantee avoiding additional unecessary memory. For example:
        # self.root.fem.solver.initialize_nonlinear_routine(self.blf, self.root.gfu, None)

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

    def get_num_stages(self) -> int:
        raise NotImplementedError()
    def get_stage_dt(self) -> list[float]:
        raise NotImplementedError()
    def update_solution(self) -> None:
        pass

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        for log in self.root.fem.solver.solve_nonlinear_system():
            yield log


class ImplicitEuler(ImplicitSchemes):
    r""" Class responsible for implementing an implicit (backwards-)Euler time-marching scheme that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`). Assuming an HDG formulation,

    .. math::
        \widetilde{\bm{M}} \bm{U}^{n+1} + \bm{f}(\bm{U}^{n+1}, \hat{\bm{U}}) &= \widetilde{\bm{M}} \bm{U}^{n},\\
                                          \bm{g}(\bm{U}^{n+1}, \hat{\bm{U}}) &= \bm{0},

    where :math:`\widetilde{\bm{M}} = \bm{M} / \delta t` is the weighted mass matrix, :math:`\bm{M}` is the mass matrix and :math:`\bm{f}` and :math:`\bm{g}` arise from the spatial discretization of the PDE on the physical elements and the AE on the facets, respectively.
    """
    name: str = "implicit_euler"
    aliases = ("ie", )
    time_levels = ('n', 'n+1')

    def get_num_stages(self) -> int:
        return 1
    def get_stage_dt(self) -> float:
        return [x * self.dt.Get() for x in [0.0, 1.0]] 

    def solve_stage(self, iStage) -> typing.Generator[Log, None, None]:
        
        if iStage != 1:
            raise TypeError(f"Stage {iStage} does not exist.")

        for log in self.root.fem.solver.solve_nonlinear_system():
            yield log

    def get_time_derivative(self, gfus: dict[str, ngs.GridFunction]) -> ngs.CF:
        return (gfus['n+1'] - gfus['n'])/self.dt

    def get_current_level(self, space: str, normalized: bool = False) -> ngs.CF:
        gfus = self.gfus[space]
        return gfus['n']

    def get_time_step(self, normalized: bool = False) -> ngs.CF:
        return self.dt


class BDF2(ImplicitSchemes):
    r""" Class responsible for implementing an implicit 2nd-order backward differentiation formula that updates the current solution (:math:`t = t^{n}`) to the next time step (:math:`t = t^{n+1}`), using also the previous solution (:math:`t = t^{n-1}`). Assuming an HDG formulation,

    .. math::
        \widetilde{\bm{M}} \bm{U}^{n+1} + \bm{f}(\bm{U}^{n+1}, \hat{\bm{U}}) &= \widetilde{\bm{M}} \Big( \frac{4}{3} \bm{U}^{n} - \frac{1}{3} \bm{U}^{n-1}\Big),\\
                                          \bm{g}(\bm{U}^{n+1}, \hat{\bm{U}}) &= \bm{0},

    where :math:`\widetilde{\bm{M}} = 3 \bm{M} / (2\delta t)` is the weighted mass matrix and :math:`\bm{M}` is the mass matrix and :math:`\bm{f}` and :math:`\bm{g}` arise from the spatial discretization of the PDE on the physical elements and the AE on the facets, respectively.
    """

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
    r"""Assuming an HDG formulation, we are solving
    
    .. math::
        \bm{M} \partial_t \bm{U} + \bm{f}\big(\bm{U}, \hat{\bm{U}} \big) &= \bm{0},\\
                                   \bm{g}\big(\bm{U}, \hat{\bm{U}} \big) &= \bm{0},

    where :math:`\bm{M}` is the mass matrix, and :math:`\bm{f}` and :math:`\bm{g}` arise from the spatial discretization of the PDE on the physical elements and the AE on the facets, respectively.

    To obtain the solution at :math:`t = t^{n+1}`, an *s-stage* DIRK update formula is

    .. math::
        \bm{U}^{n+1} = \bm{U}^{n} - \delta t \sum_{i=1}^{s} b_{i} \bm{M}^{-1} \bm{f}(\bm{z}_{i}),

    where :math:`\bm{z}_i = \big( \bm{y}_i, \hat{\bm{y}}_i \big)^T` denotes the physical :math:`\bm{y}` and facet :math:`\hat{\bm{y}}` solution at the *ith* stage, obtained from

    .. math::
        \bm{y}_{i}       &= \bm{U}^{n} - \delta t \sum_{j=1}^{s} a_{ij} \bm{M}^{-1} \bm{f}(\bm{z}_j),\\
        \bm{g}(\bm{z}_i) &= \bm{0}.
  
    The above can be used to define a residual for solving the nonlinear system iteratively

    .. math::
        \bm{r}_i       &= \widetilde{\bm{M}}_i \bm{y}_i - \widetilde{\bm{M}}_i \bm{U}^{n} + \frac{1}{a_{ii}} \sum_{j=1}^{i-1} a_{ij} \bm{f}(\bm{z}_j) + \bm{f}(\bm{z}_i),\\ 
        \hat{\bm{r}}_i &= \bm{g}(\bm{z}_i),

    where :math:`\widetilde{\bm{M}}_i = \bm{M}/(a_{ii} \delta t)` is the weighted mass matrix associated with the *ith* stage.

    Therefore, a Newton-Rhapson update formula for the *kth* iteration can be written as

    .. math::
        \bm{N}_{i}^{k} \delta \bm{z}_{i}^{k} = - \bm{R}_{i}^{k},

    where the overall residual is :math:`\bm{R}_{i}^{k} = \Big( \bm{r}(\bm{z}_{i}^{k}), \hat{\bm{r}}(\bm{z}_{i}^{k}) \Big)^T` and the iteration matrix is defined as

    .. math::
        \bm{N}_{i}^{k} &= \partial \bm{R} (\bm{z}_{i}^{k} ) / \partial \bm{z}_i, \\[2ex]
        &= 
        \begin{pmatrix}
            \widetilde{\bm{M}}_{i} + \partial \bm{f}^{k} / \partial \bm{y}_i  &  \partial \bm{f}^{k} / \partial \hat{\bm{y}}_i\\
                                     \partial \bm{g}^{k} / \partial \bm{y}_i  &  \partial \bm{g}^{k} / \partial \hat{\bm{y}}_i
        \end{pmatrix}.

    :note: In terms of implementation, this is based on two bilinear forms: **blf** and **blfs**

    .. math:: 
        \bf{blf} = 
        \begin{pmatrix}
            \widetilde{\bm{M}}_i \bm{y}_i + \bm{f}(\bm{z}_i)\\
                                            \bm{g}(\bm{z}_i)
        \end{pmatrix},
        \qquad 
        \bf{blfs} = 
        \begin{pmatrix}
                                            \bm{f}(\bm{z}_i)\\
                                                      \bm{0}
        \end{pmatrix}.
    """
    time_levels = ('n+1',)
    
    def assemble(self) -> None:

        condense = self.root.fem.static_condensation

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
        self.mass += self.root.fem.blf['U']['mass']

        # Assemble the mass matrix once.
        self.mass.Assemble()
        
        # Add both spatial and mass-matrix terms in blf.
        self.add_sum_of_integrals(self.blf, self.root.fem.blf, 'implicit bilinear form')

        # Skip the mass matrix contribution in blfs and only use the space for "U".
        integrals = self.parse_sum_of_integrals(self.root.fem.blf, include_spaces=['U'], exclude_terms=('mass',))
        self.add_sum_of_integrals(self.blfs, integrals, "implicit bilinear form for splitting")

        # Initialize the nonlinear solver here. Notice, it uses a reference to blf, rhs and gfu.
        self.root.fem.solver.initialize_nonlinear_routine(self.blf, self.root.fem.gfu, self.rhs)

    def get_num_stages(self) -> int:
        raise NotImplementedError()
    def get_stage_dt(self) -> list[float]:
        raise NotImplementedError()
    def update_solution(self) -> None:
        pass

    # Generic function to store (S-)DIRK coefficient, vector pairs.
    def setup_stage_definitions(self, 
                                stage_data: list[tuple[typing.Any, list[tuple[float, typing.Any]]]]
                                ) -> None:
        r"""
        stage_data:
            list of (apply_target, coeffs) for each stage, starting at stage 1.
            apply_target: vector to pass to Apply() or None.
            coeffs: list of (coeff, vector) pairs to sum into rhs.
        """
        self.stage_definitions = \
        {
            i + 1: {"apply_target": apply_target, "coeffs": coeffs}
            for i, (apply_target, coeffs) in enumerate(stage_data)
        }

    # Generic function that solves a (S-)DIRK scheme.
    def solve_stage(self, iStage) -> typing.Generator[Log, None, None]:

        # Extract the stage information, if possible.
        try:
            stage_info = self.stage_definitions[iStage]
        except KeyError:
            raise TypeError(f"Stage {iStage} does not exist.")
    
        # Stage 1 is a special case, we handle it separately.
        if iStage == 1:
            self.mu0.data = self.mass.mat * self.root.fem.gfu.vec
 
        # Distinguish between singly-diagonal (SDIRK) and variable (DIRK) coefficients.
        if self.variable_aii is not None:
            self.aii.Set( self.variable_aii[iStage] ) # should be set before the solving routine.
            ovaii = 1.0/self.variable_aii[iStage]
            self.rhs.data = ovaii * self.mu0
        else:
            self.rhs.data = self.mu0

        # Compute previous-stage residuals, if the apply_target exists.
        if stage_info["apply_target"] is not None:
            self.blfs.Apply(self.root.fem.gfu.vec, stage_info["apply_target"])
           
        # Build the right-hand side, scaled with its respective coefficients.
        for aij, xj in stage_info["coeffs"]:
            self.rhs.data += aij * xj
    
        # Solve the resulting nonlinear system.
        for log in self.root.fem.solver.solve_nonlinear_system():
            yield {"stage": iStage, **log}


class SDIRK22(DIRKSchemes):
    r""" Updates the solution via a 2-stage 2nd-order (stiffly-accurate) singly diagonally-implicit Runge-Kutta (SDIRK). Taken from Section 2.6 in :cite:`ascher1997implicit`. Its corresponding Butcher tableau is:

    .. math::
        \begin{array}{c|cc}
	        \alpha & \alpha     & 0      \\
	        1    &   1 - \alpha & \alpha \\
            \hline
	             & 1 - \alpha    & \alpha
        \end{array}

    where :math:`\alpha = (2 - \sqrt{2})/2`.
    
    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{U}^{n+1} = \bm{y}_{2}`.
    """
    name: str = "sdirk22" 

    def initialize_butcher_tableau(self):
        
        alpha = ngs.sqrt(2.0)/2.0

        self.aii = 1.0 - alpha 
        self.a21 = alpha 
       
        # Time stamps for the stage values between t = [n,n+1].
        self.c = [1.0 - alpha, 1.0] 

        # This is possible, because the method is stiffly-accurate.
        self.b1 = self.a21
        self.b2 = self.aii

    def get_num_stages(self) -> int:
        return 2
    def get_stage_dt(self) -> list[float]:
        return [x * self.dt.Get() for x in ([0.0] + self.c)] # 1st stage is padded. 

    def configure_scheme(self) -> None:
        
        # This is an SDIRK scheme, aii is constant.
        self.variable_aii = None
        
        # Scale the coefficients by their (-ve) diagonal counterpart.
        a21 = -self.a21 / self.aii
        self.setup_stage_definitions([
            (None, []),                 # stage 1
            (self.x1, [(a21, self.x1)]) # stage 2
        ])

    def assemble(self) -> None:
        super().assemble()
 
        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()

        # Initialize the multistage configuration parameters for the scheme.
        self.configure_scheme()

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

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        yield from self.solve_stage(1)
        yield from self.solve_stage(2) # last stage corresponds to u^{n+1}.


class SDIRK33(DIRKSchemes):
    r""" Updates the solution via a 3-stage 3rd-order (stiffly-accurate) singly diagonally-implicit Runge-Kutta (SDIRK). Taken from Section 2.7 in :cite:`ascher1997implicit`. Its corresponding Butcher tableau is: 

    .. math::
        \begin{array}{c|ccc}
	        0.4358665215 & 0.4358665215 &  0            & 0            \\
	        0.7179332608 & 0.2820667392 &  \phantom{-}0.4358665215 & 0 \\
            1            & 1.2084966490 & -0.6443631710 & 0.4358665215 \\
            \hline
	                     & 1.2084966490 & -0.6443631710 & 0.4358665215
        \end{array}

    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{U}^{n+1} = \bm{y}_{3}`.
    """
    name: str = "sdirk33"

    def initialize_butcher_tableau(self):

        self.aii =  0.4358665215
        self.a21 =  0.2820667392
        self.a31 =  1.2084966490
        self.a32 = -0.6443631710

        # Time stamps for the stage values between t = [n,n+1].
        self.c = [0.4358665215, 0.7179332608, 1.0]

        # This is possible, because the method is stiffly-accurate.
        self.b1 = self.a31
        self.b2 = self.a32
        self.b3 = self.aii
        
    def get_num_stages(self) -> int:
        return 3
    def get_stage_dt(self) -> list[float]:
        return [x * self.dt.Get() for x in ([0.0] + self.c)] # 1st stage is padded. 

    def configure_scheme(self) -> None:
        
        # This is an SDIRK scheme, aii is constant.
        self.variable_aii = None
        
        # Scale the coefficients by their (-ve) diagonal counterpart.
        a21 = -self.a21 / self.aii
        a31 = -self.a31 / self.aii 
        a32 = -self.a32 / self.aii
        self.setup_stage_definitions([
            (None, []),                                 # stage 1
            (self.x1, [(a21, self.x1)]),                # stage 2
            (self.x2, [(a31, self.x1), (a32, self.x2)]) # stage 3
        ])

    def assemble(self) -> None:
        super().assemble()
 
        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()

        # Invert the (weighted-)mass matrix.
        self.minv = self.root.fem.solver.get_inverse(self.mass, self.root.fem.fes)

        # Initialize the multistage configuration parameters for the scheme.
        self.configure_scheme()

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

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        yield from self.solve_stage(1)
        yield from self.solve_stage(2)
        yield from self.solve_stage(3) # last stage corresponds to u^{n+1}.


class SDIRK43(DIRKSchemes):
    r""" Updates the solution via a 4-stage 3rd-order (stiffly-accurate) singly diagonally-implicit Runge-Kutta (SDIRK). Taken from Section 2.8 in :cite:`ascher1997implicit`. Its corresponding Butcher tableau is:

    .. math::
        \begin{array}{c|cccc}
	        \frac{1}{2} & \phantom{-}\frac{1}{2} & \phantom{-}0           & 0           & 0 \\
	        \frac{2}{3} & \phantom{-}\frac{1}{6} & \phantom{-}\frac{1}{2} & 0           & 0 \\
            \frac{1}{2} & -\frac{1}{2}           & \phantom{-}\frac{1}{2} & \frac{1}{2} & 0 \\
            1           & \phantom{-}\frac{3}{2} & -\frac{3}{2}           & \frac{1}{2} & \frac{1}{2} \\
            \hline
	                    & \phantom{-}\frac{3}{2} & -\frac{3}{2}           & \frac{1}{2} & \frac{1}{2}
        \end{array}
    
    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{U}^{n+1} = \bm{y}_{2}`.
    """
    name: str = "sdirk43"

    def initialize_butcher_tableau(self):

        self.aii =  1.0/2.0
        self.a21 =  1.0/6.0
        self.a31 = -1.0/2.0
        self.a32 =  1.0/2.0
        self.a41 =  3.0/2.0
        self.a42 = -3.0/2.0
        self.a43 =  1.0/2.0

        # Time stamps for the stage values between t = [n,n+1].
        self.c = [1.0/2.0, 2.0/3.0, 1.0/2.0, 1.0]

        # This is possible, because the method is stiffly-accurate.
        self.b1 = self.a41
        self.b2 = self.a42
        self.b3 = self.a43
        self.b4 = self.aii

    def get_num_stages(self) -> int:
        return 4
    def get_stage_dt(self) -> list[float]:
        return [x * self.dt.Get() for x in ([0.0] + self.c)] # 1st stage is padded. 

    def configure_scheme(self) -> None:
        
        # This is an SDIRK scheme, aii is constant.
        self.variable_aii = None
        
        # Scale the coefficients by their (-ve) diagonal counterpart.
        a21 = -self.a21 / self.aii
        a31 = -self.a31 / self.aii 
        a32 = -self.a32 / self.aii
        a41 = -self.a41 / self.aii 
        a42 = -self.a42 / self.aii
        a43 = -self.a43 / self.aii
        self.setup_stage_definitions([
            (None, []),                                                 # stage 1
            (self.x1, [(a21, self.x1)]),                                # stage 2
            (self.x2, [(a31, self.x1), (a32, self.x2)]),                # stage 3
            (self.x3, [(a41, self.x1), (a42, self.x2), (a43, self.x3)]) # stage 4
        ])

    def assemble(self) -> None:
        super().assemble()
 
        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.x3 = self.root.fem.gfu.vec.CreateVector()

        # Initialize the multistage configuration parameters for the scheme.
        self.configure_scheme()

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

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        yield from self.solve_stage(1)
        yield from self.solve_stage(2)
        yield from self.solve_stage(3)
        yield from self.solve_stage(4) # last stage corresponds to u^{n+1}.


class SDIRK54(DIRKSchemes):
    r""" Updates the solution via a 5-stage 4th-order (stiffly-accurate) singly diagonally-implicit Runge-Kutta (SDIRK). Taken from Table 6.5 in :cite:`wanner1996solving`. Its corresponding Butcher tableau is: 

    .. math::
        \begin{array}{c|ccccc}
	        \frac{1}{4}   & \frac{1}{4}      &  \phantom{-}0           & 0              &  \phantom{-}0           & 0  \\
	        \frac{3}{4}   & \frac{1}{2}      &  \phantom{-}\frac{1}{4} & 0              &  \phantom{-}0           & 0  \\
            \frac{11}{20} & \frac{17}{50}    &  -\frac{1}{25}          & \frac{1}{4}    &  \phantom{-}0           & 0  \\
            \frac{1}{2}   & \frac{371}{1360} &  -\frac{137}{2720}      & \frac{15}{544} &  \phantom{-}\frac{1}{4} & 0  \\
            1             & \frac{25}{24}    &  -\frac{49}{48}         & \frac{125}{16} & -\frac{85}{12} & \frac{1}{4}\\
            \hline
	                      & \frac{25}{24}    &  -\frac{49}{48}         & \frac{125}{16} & -\frac{85}{12} & \frac{1}{4}
        \end{array}

    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{U}^{n+1} = \bm{y}_{5}`.
    """
    name: str = "sdirk54"

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
        self.c = [1.0/4.0, 3.0/4.0, 11.0/20.0, 1.0/2.0, 1.0]

        # This is possible, because the method is stiffly-accurate.
        self.b1 = self.a51
        self.b2 = self.a52
        self.b3 = self.a53
        self.b4 = self.a54
        self.b5 = self.aii

    def get_num_stages(self) -> int:
        return 5
    def get_stage_dt(self) -> list[float]:
        return [x * self.dt.Get() for x in ([0.0] + self.c)] # 1st stage is padded. 

    def configure_scheme(self) -> None:
        
        # This is an SDIRK scheme, aii is constant.
        self.variable_aii = None
        
        # Scale the coefficients by their (-ve) diagonal counterpart.
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
        self.setup_stage_definitions([
            (None, []),                                                                 # stage 1
            (self.x1, [(a21, self.x1)]),                                                # stage 2
            (self.x2, [(a31, self.x1), (a32, self.x2)]),                                # stage 3
            (self.x3, [(a41, self.x1), (a42, self.x2), (a43, self.x3)]),                # stage 4
            (self.x4, [(a51, self.x1), (a52, self.x2), (a53, self.x3), (a54, self.x4)]) # stage 5
        ])

    def assemble(self) -> None:
        super().assemble()
 
        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.x3 = self.root.fem.gfu.vec.CreateVector()
        self.x4 = self.root.fem.gfu.vec.CreateVector()

        # Initialize the multistage configuration parameters for the scheme.
        self.configure_scheme()

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

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        yield from self.solve_stage(1)
        yield from self.solve_stage(2)
        yield from self.solve_stage(3)
        yield from self.solve_stage(4)
        yield from self.solve_stage(5) # last stage corresponds to u^{n+1}.


class DIRK43_WSO2(DIRKSchemes):
    r""" Updates the solution via a 4-stage 3rd-order (stiffly-accurate) diagonally-implicit Runge-Kutta (DIRK) with a weak stage order (WSO) of 3. Taken from Section 3 in :cite:`ketcheson2020dirk`. Its corresponding Butcher tableau is: 

    .. math::
        \begin{array}{c|cccc}
	        0.01900072890 & 0.01900072890 & \phantom{-}0             & 0             & 0            \\
	        0.78870323114 & 0.40434605601 & \phantom{-}0.38435717512 & 0             & 0            \\
            0.41643499339 & 0.06487908412 &           -0.16389640295 & 0.51545231222 & 0            \\
            1             & 0.02343549374 &           -0.41207877888 & 0.96661161281 & 0.42203167233\\
            \hline
	                      & 0.02343549374 &           -0.41207877888 & 0.96661161281 & 0.42203167233
        \end{array}

    :note: No need to explicitly form the solution at the next time step, since this is a stiffly-accurate method, i.e. :math:`\bm{U}^{n+1} = \bm{y}_{4}`.
    """
    name: str = "dirk43_wso2"

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
        self.c  = [self.a11, 
                   self.a21 + self.a22,
                   self.a31 + self.a32 + self.a33,
                   1.0]

        # This is possible, because the method is stiffly-accurate.
        self.b1 = self.a41
        self.b2 = self.a42
        self.b3 = self.a43
        self.b4 = self.a44

    def get_num_stages(self) -> int:
        return 4
    def get_stage_dt(self) -> list[float]:
        return [x * self.dt.Get() for x in ([0.0] + self.c)] # 1st stage is padded. 

    def configure_scheme(self) -> None:
        
        # This is a non-single DIRK scheme, aii is different between stages.
        self.variable_aii = [None, self.a11, self.a22, self.a33, self.a44] # 1st stage is padded.

        # Scale the coefficients by their (-ve) diagonal counterpart.
        a21 = -self.a21 / self.a22
        a31 = -self.a31 / self.a33 
        a32 = -self.a32 / self.a33
        a41 = -self.a41 / self.a44 
        a42 = -self.a42 / self.a44
        a43 = -self.a43 / self.a44
        self.setup_stage_definitions([
            (None, []),                                                 # stage 1
            (self.x1, [(a21, self.x1)]),                                # stage 2
            (self.x2, [(a31, self.x1), (a32, self.x2)]),                # stage 3
            (self.x3, [(a41, self.x1), (a42, self.x2), (a43, self.x3)]) # stage 4
        ]) 

    def assemble(self) -> None:
        super().assemble()
 
        # Reserve space for additional vectors.
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.x3 = self.root.fem.gfu.vec.CreateVector()

        # Initialize the multistage configuration parameters for the scheme.
        self.configure_scheme()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:

        u, v = self.root.fem.TnT['U']
        gfus = self.gfus['U'].copy()
        gfus['n+1'] = u
       
        # This initializes the coefficients for this scheme.
        self.initialize_butcher_tableau()

        # Create a variable parameter, for the diagonal coefficients a_{ii}.
        self.aii = ngs.Parameter(1.0)

        # NOTE, aii needs to be kept here as a stage-variable, since it shows 
        # up in the residual iterations via the nonlinear solver.
        ovadt = 1.0/(self.aii*self.dt) 

        # Add the scaled mass matrix.
        blf['U']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        yield from self.solve_stage(1)
        yield from self.solve_stage(2)
        yield from self.solve_stage(3)
        yield from self.solve_stage(4) # last stage corresponds to u^{n+1}.
 

class DIRK34_LDD(DIRKSchemes):
    r""" Updates the solution via a 3-stage 4th-order diagonally-implicit Runge-Kutta (DIRK) with a low-dispersion and dissipation. Taken from Table A.1 in :cite:`najafi2013low`. Its corresponding Butcher tableau is: 

    .. math::
        \begin{array}{c|ccc}
	        0.257820901066211 & 0.377847764031163 &  \phantom{-}0                 & 0                 \\
	        0.434296446908075 & 0.385232756462588 &  \phantom{-}0.461548399939329 & 0                 \\
            0.758519768667167 & 0.675724855841358 & -0.061710969841169            & 0.241480233100410 \\
            \hline
	                          & 0.750869573741408 & -0.362218781852651            & 0.611349208111243  
        \end{array}
    """

    name: str = "dirk34_ldd"

    def initialize_butcher_tableau(self):

        self.a11 =  0.377847764031163
        self.a21 =  0.385232756462588
        self.a22 =  0.461548399939329
        self.a31 =  0.675724855841358
        self.a32 = -0.061710969841169
        self.a33 =  0.241480233100410
        
        # Time stamps for the stage values between t = [n,n+1].
        self.c = [0.257820901066211, 0.434296446908075, 0.758519768667167]

        # NOTE, this is not L-stable.
        self.b1 =  0.750869573741408
        self.b2 = -0.362218781852651
        self.b3 =  0.611349208111243

    def get_num_stages(self) -> int:
        return 3
    def get_stage_dt(self) -> list[float]:
        return [x * self.dt.Get() for x in ([0.0] + self.c)] # 1st stage is padded. 

    def configure_scheme(self) -> None:
        
        # This is a non-single DIRK scheme, aii is different between stages.
        self.variable_aii = [None, self.a11, self.a22, self.a33] # 1st stage is padded.

        # Scale the coefficients by their (-ve) diagonal counterpart.
        a21 = -self.a21 / self.a22
        a31 = -self.a31 / self.a33 
        a32 = -self.a32 / self.a33
        self.setup_stage_definitions([
            (None, []),                                 # stage 1
            (self.x1, [(a21, self.x1)]),                # stage 2
            (self.x2, [(a31, self.x1), (a32, self.x2)]) # stage 3
        ]) 

    def assemble(self) -> None:
        super().assemble()
 
        # Reserve space for additional vectors.
        self.u0 = self.root.fem.gfu.vec.CreateVector()
        self.x1 = self.root.fem.gfu.vec.CreateVector()
        self.x2 = self.root.fem.gfu.vec.CreateVector()
        self.x3 = self.root.fem.gfu.vec.CreateVector()

        # Precompute the mass matrix of the volume elements.
        self.minv = self.root.fem.solver.get_inverse(self.mass, self.root.fem.fes)

        # Initialize the multistage configuration parameters for the scheme.
        self.configure_scheme()
        
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

        # NOTE, aii needs to be kept here as a stage-variable, since it shows 
        # up in the residual iterations via the nonlinear solver.
        ovadt = 1.0/(self.aii*self.dt)

        # Add the scaled mass matrix.
        blf['U']['mass'] = ngs.InnerProduct(ovadt*u, v) * ngs.dx

    def update_solution(self) -> None:
        
        # Spatial term evaluated at the final stage: 3.
        self.blfs.Apply(self.root.fem.gfu.vec, self.x3)

        # Need to explicitly update the solution.
        self.root.fem.gfu.vec.data = self.u0 \
            - self.minv * ( self.b1 * self.x1 + self.b2 * self.x2 + self.b3 * self.x3)

        # We assume f, because it uses the (volume) solution at u^{n+1}.
        self.f_uhat.Assemble()
        self.root.fem.gfus['Uhat'].vec.data = self.minv_uhat * self.f_uhat.vec

    def solve_current_time_level(self) -> typing.Generator[Log, None, None]:

        # Book-keep the solution at u^{n}.
        self.u0.data = self.root.fem.gfu.vec # NOTE, this needs changes to work as an IMEX-pair.

        yield from self.solve_stage(1)
        yield from self.solve_stage(2)
        yield from self.solve_stage(3)

        # NOTE, must explicitly reconstruct the solution at u^{n+1}.
        self.update_solution()


