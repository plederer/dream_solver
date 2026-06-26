## Solver Architecture

{py:class}`~dream.compressible_flow.solver.CompressibleFlowSolver` is configured by composing
interchangeable sub-objects for the finite element method, equation of state, viscosity model,
non-dimensionalisation scaling, and Riemann solver. As with all dream solvers, `solver.time`
selects the outer solution loop and `solver.fem.scheme` selects the numerical time integration
scheme.

## Compressible flow equations

The fundamental equations describing the motion of an unsteady, viscous and compressible flow in a space-time cylinder
$\Omega \times (0, t_{end}] \in \mathbb{R}^{d+1}$ with non-empty bounded $d$-dimensional spatial domain $\Omega$, with
boundary $\partial \Omega$, and final time $t_{end}$, are specified by the Navier-Stokes equations. In terms of
conservative variables, $\vec{U} = \begin{pmatrix} \rho, & \rho \vec{u}, & \rho E \end{pmatrix}^\T$, with density $\rho$, velocity $\vec{u}$, and total specific energy $E$,
this system can be expressed in dimensionless form as

\begin{align*}
    \frac{\partial \vec{U}}{\partial t} + \div(\vec{F}(\vec{U}) - \vec{G}(\vec{U}, \nabla \vec{U} )) = \vec{0}.
\end{align*}

In a general form the convective $\vec{F}$ and the viscous fluxes $\vec{G}$ are given by
\begin{align*}
    \vec{F}(\vec{U}) & = \begin{pmatrix}
                             \rho \vec{u}^\T                     \\
                             \rho \vec{u} \otimes \vec{u} + p \I \\
                             \rho H \vec{u}^\T
                         \end{pmatrix}, &
    \vec{G}(\vec{U}, \nabla \vec{U}) = \begin{pmatrix}\vec{0}^\T \\\mat{\tau}    \\ (\mat{\tau} \vec{u} - \vec{q})^\T \end{pmatrix},
\end{align*}
where $p$ denotes the pressure, $H = E + p/\rho$ the specific enthalpy, $\mat{\tau}$ the deviatoric stress tensor, and $\vec{q}$ the heat flux vector.

To close the system of equations we need to specify the equation of state (see {py:class}`~dream.compressible_flow.eos`) and the constitutive relations for the deviatoric stress tensor $\mat{\tau}$ and the heat flux vector $\vec{q}$ (see {py:class}`~dream.compressible_flow.viscosity`).

## Quasi-linear Euler equations

The hyperbolic nature of the Navier-Stokes equations with respect to time lies in the Euler
equations {cite}`hirschNumericalComputationInternal2002`
\begin{align*}
    \frac{\partial \vec{U}}{\partial t} + \div(\vec{F}(\vec{U}))  = \vec{0},
\end{align*}
which are derived by neglecting the viscous contributions. From a characteristic point of view,
it is essential to express these equations in quasi-linear form
\begin{align*}
    \frac{\partial \vec{U}}{\partial t} + \sum_{i=1}^d \mat{A}_i \frac{\partial \vec{U}}{\partial x_i}  &= \vec{0}, \\
\end{align*}
where the $\mat{A}_i$ are the directional convective Jacobians.

## Discretisation

The solver discretises the compressible Navier-Stokes equations on a mesh $\mesh$ using either
the **Hybridised Discontinuous Galerkin (HDG)** method ({py:class}`~dream.compressible_flow.conservative.hdg.ConservativeHDG`)
or the standard **Discontinuous Galerkin (DG)** method
({py:class}`~dream.compressible_flow.conservative.dg.ConservativeDG`).

**HDG** introduces an additional *skeleton* unknown $\widehat{\vec{U}}_h$ living on the mesh
facets $\facets$. The local element problems are solved independently for each element $K \in
\mesh$ given $\widehat{\vec{U}}_h$, and then a global system is assembled solely in terms of the
skeleton degrees of freedom. This *static condensation* makes HDG particularly efficient for
high-order discretisations since the global system is much smaller than the full DG system. The
viscous terms are handled by the `viscous_treatment` sub-object of
{py:class}`~dream.compressible_flow.conservative.hdg.ConservativeHDG`: the default
{py:class}`~dream.compressible_flow.conservative.diffusion.StrainHeat` formulation uses the
physical strain-rate and heat-flux form; {py:class}`~dream.compressible_flow.conservative.diffusion.Gradient`
and {py:class}`~dream.compressible_flow.conservative.diffusion.InteriorPenaltyHDG` are
available as alternatives.

**DG** keeps all degrees of freedom element-local with no additional skeleton unknowns. It is
straightforward to use with explicit time integrators, at the cost of a
larger globally coupled system for implicit schemes. The viscous treatment for DG is
{py:class}`~dream.compressible_flow.conservative.diffusion.InteriorPenaltySDG`.

In both formulations the convective numerical flux at element interfaces is determined by the
selected {py:class}`~dream.compressible_flow.riemann_solver.RiemannSolver`. The available
options range from the simple {py:class}`~dream.compressible_flow.riemann_solver.LaxFriedrich`
 solver to the more accurate
 {py:class}`~dream.compressible_flow.riemann_solver.Upwind`,
{py:class}`~dream.compressible_flow.riemann_solver.Roe`,
{py:class}`~dream.compressible_flow.riemann_solver.HLL`, and
{py:class}`~dream.compressible_flow.riemann_solver.HLLEM` solvers.

## Boundary Conditions

Boundary conditions are assigned to named mesh boundaries via the solver's `bcs`
({py:class}`~dream.mesh.BoundaryConditions`) attribute. The following condition types are
available:

| Class | Description |
|---|---|
| {py:class}`~dream.compressible_flow.config.FarField` | Characteristic inflow/outflow;<br>farfield state $(\rho, \mathbf{u}, T)_\infty$ |
| {py:class}`~dream.compressible_flow.config.Outflow` | Pressure outflow: prescribe static pressure $p$ |
| {py:class}`~dream.compressible_flow.config.Inflow` | Inflow: prescribe $(\rho, \mathbf{u})$<br>or total conditions |
| {py:class}`~dream.compressible_flow.config.InviscidWall` | Slip wall: zero normal velocity |
| {py:class}`~dream.compressible_flow.config.Symmetry` | Symmetry plane |
| {py:class}`~dream.compressible_flow.config.IsothermalWall` | No-slip wall with fixed temperature |
| {py:class}`~dream.compressible_flow.config.AdiabaticWall` | No-slip wall with zero heat flux |
| {py:class}`~dream.compressible_flow.config.CBC` | Non-reflecting BC;<br>subclasses: `GRCBC`, `NSCBC` |
| {py:class}`~dream.compressible_flow.config.InterfaceBC` | IMEX coupling: solution fields<br>from neighbouring submesh at $\Gamma_i$ |

## Initial and Domain Conditions

Domain conditions are assigned via the solver's `dcs`
({py:class}`~dream.mesh.DomainConditions`) attribute and apply over named mesh regions
rather than boundaries. For transient simulations the initial state is mandatory:

```python
Uinf = solver.get_farfield_fields((1, 0))
solver.dcs['domain'] = Initial(fields=Uinf)
```

The fields object passed to `Initial` must be a {py:class}`~dream.compressible_flow.config.flowfields`
instance, typically obtained via
{py:meth}`~dream.compressible_flow.solver.CompressibleFlowSolver.get_farfield_fields`
for a uniform free-stream state, or assembled manually from density, velocity, and pressure (or
temperature) using `flowfields(rho=..., u=..., p=...)`.

| Class | Description |
|---|---|
| {py:class}`~dream.mesh.Initial` | Initial state $\mathbf{U}(\cdot,0)$ projected onto the solution space;<br>required for `'transient'` and `'pseudo_time_stepping'` outer loops |
| {py:class}`~dream.compressible_flow.config.Force` | Body force term added to the momentum equation |
| {py:class}`~dream.compressible_flow.config.Perturbation` | Superimposed perturbation field<br>on the background state |

## Time Integration

The outer solution loop is selected via `solver.time` and the numerical time integration
scheme via `solver.fem.scheme`
(see the {doc}`time module </modules/time/index>` for the full time infrastructure).

**Implicit schemes** (unconditionally stable — recommended for viscous or diffusion-dominated
problems and large time steps):

| Scheme | Stages | Order | Key |
|--------|--------|-------|-----|
| {py:class}`~dream.compressible_flow.conservative.time.implicit.ImplicitEuler` | 1 | 1 | `'implicit_euler'` |
| {py:class}`~dream.compressible_flow.conservative.time.implicit.BDF2` | 1 | 2 | `'bdf2'` |
| {py:class}`~dream.compressible_flow.conservative.time.implicit.BDF3` | 1 | 3 | `'bdf3'` |
| {py:class}`~dream.compressible_flow.conservative.time.implicit.SDIRK22` | 2 | 2 | `'sdirk22'` |
| {py:class}`~dream.compressible_flow.conservative.time.implicit.SDIRK33` | 3 | 3 | `'sdirk33'` |
| {py:class}`~dream.compressible_flow.conservative.time.implicit.SDIRK43` | 4 | 3 | `'sdirk43'` |
| {py:class}`~dream.compressible_flow.conservative.time.implicit.SDIRK54` | 5 | 4 | `'sdirk54'` |
| {py:class}`~dream.compressible_flow.conservative.time.implicit.DIRK43_WSO2` | 4 | 3 (WSO 3) | `'dirk43_wso2'` |
| {py:class}`~dream.compressible_flow.conservative.time.implicit.DIRK34_LDD` | 3 | 4 | `'dirk34_ldd'` |

**Explicit schemes** (require CFL condition — suitable for inviscid or convection-dominated problems):

| Scheme | Stages | Order | Key |
|--------|--------|-------|-----|
| {py:class}`~dream.compressible_flow.conservative.time.explicit.ExplicitEuler` | 1 | 1 | `'explicit_euler'` |
| {py:class}`~dream.compressible_flow.conservative.time.explicit.RK_ARS22` | 2 | 2 | `'rk_ars22'` |
| {py:class}`~dream.compressible_flow.conservative.time.explicit.RK_ARS232` | 2 | 2 | `'rk_ars232'` |
| {py:class}`~dream.compressible_flow.conservative.time.explicit.RK_ARS33` | 3 | 3 | `'rk_ars33'` |
| {py:class}`~dream.compressible_flow.conservative.time.explicit.RK_ARS43` | 4 | 3 | `'rk_ars43'` |
| {py:class}`~dream.compressible_flow.conservative.time.explicit.SSPRK3` | 3 | 3 | `'ssprk3'` |
| {py:class}`~dream.compressible_flow.conservative.time.explicit.CRK4` | 4 | 4 | `'crk4'` |

**IMEX schemes** (geometry splitting — two submeshes $\mesh^{im} \cup \mesh^{ex}$):

The implicit HDG solver runs on $\mesh^{im}$ and the explicit DG solver on $\mesh^{ex}$.
Each solver uses its own scheme, but the stage times must be synchronised:
$c_i^{im} = \bar{c}_{i+1}^{ex}$. The ARS-family schemes are designed for exactly this pairing:

| Order | Implicit scheme | Explicit scheme |
|-------|----------------|-----------------|
| 2 | `'sdirk22'` | `'rk_ars22'` |
| 2 | `'sdirk22'` | `'rk_ars232'` |
| 3 | `'sdirk33'` | `'rk_ars33'` |
| 3 | `'sdirk43'` | `'rk_ars43'` |

