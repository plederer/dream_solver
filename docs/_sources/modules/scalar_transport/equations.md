## Solver Architecture

{py:class}`~dream.scalar_transport.solver.ScalarTransportSolver` is composed of three
interchangeable sub-objects selected via `solver.fem`, `solver.riemann_solver`, and `solver.time`.
The finite element method (`fem`) additionally owns the numerical time integration scheme via its
`scheme` property, while `solver.time` controls the outer solution loop (transient, stationary, or
pseudo-time-stepping).

## Scalar Transport Equation

A transport equation expresses a conservation principle by describing how a scalar quantity
$u$ evolves in space and time due to the combined effects of convection and diffusion.
The governing (linear, scalar) convection–diffusion equation reads

$$
\frac{\partial u}{\partial t} + \nabla \cdot (\vec{b}\,u) - \nabla \cdot (\kappa\,\nabla u) = 0
\quad \text{in } \Omega \times (0, t_{\mathrm{end}}],
$$

where $\vec{b}$ is the advecting velocity field (set via
{py:attr}`~dream.scalar_transport.solver.ScalarTransportSolver.convection_velocity`;
its spatial dimension is inferred from the mesh) and $\kappa \geq 0$ is the scalar diffusivity
(set via {py:attr}`~dream.scalar_transport.solver.ScalarTransportSolver.diffusion_coefficient`).

Setting {py:attr}`~dream.scalar_transport.solver.ScalarTransportSolver.is_inviscid` to `True`
enforces $\kappa = 0$, reducing the problem to a pure convection (hyperbolic) equation.

## Discretisation

### Hybridised Discontinuous Galerkin (HDG)

The {py:class}`~dream.scalar_transport.spatial.HDG` method partitions the domain into elements
and introduces an additional *skeleton* (or *trace*) unknown $\hat{u}$ living on the mesh
facets $\facets$.  The global system couples only skeleton unknowns; the element-interior
unknowns are eliminated locally via static condensation.  This leads to a significantly
smaller global system compared to standard DG, making HDG attractive for implicit time
integration at low polynomial orders.

The discrete problem is: find $(u_h, \hat{u}_h)$ such that for every element $K \in \mesh$

$$
\int_K \frac{\partial u_h}{\partial t} v\,\mathrm{d}x
+ \int_{\partial K} \hat{f}^*(u_h, \hat{u}_h;\vec{b})\,v\,\mathrm{d}s
+ \int_K \kappa\,\nabla u_h \cdot \nabla v\,\mathrm{d}x
- \int_{\partial K} \kappa\,\nabla u_h \cdot \vec{n}\,v\,\mathrm{d}s
= 0,
$$

where $\hat{f}^*$ is a single-valued numerical flux on $\partial K$ determined by the
{py:class}`~dream.scalar_transport.riemann_solver.RiemannSolver` and the stabilisation
parameter (see {py:class}`~dream.scalar_transport.spatial.HDG` for details).

### Discontinuous Galerkin (DG)

The {py:class}`~dream.scalar_transport.spatial.DG` method uses element-wise polynomial
spaces without skeleton unknowns.  Inter-element communication is handled entirely through
the numerical flux chosen by the Riemann solver.  DG is simpler to implement and sometimes
preferred for explicit time stepping, but the global system is larger because no static
condensation is possible.

### Riemann Solver

The {py:class}`~dream.scalar_transport.riemann_solver.LaxFriedrich` Riemann solver defines
the numerical flux at element interfaces as

$$
\hat{f}^*(u^-, u^+) = \tfrac{1}{2}(\vec{b}\cdot\vec{n})(u^- + u^+)
  - \tfrac{\lambda}{2}(u^+ - u^-),
$$

where $\lambda = |\vec{b}\cdot\vec{n}|$ is the maximum wave speed and $u^\pm$ are the
one-sided traces from the two neighbouring elements.

## Boundary and Initial Conditions

Boundary and initial conditions are attached to the solver via
{py:class}`~dream.mesh.BoundaryConditions` and {py:class}`~dream.mesh.DomainConditions`:

| Class | Description |
|-------|-------------|
| {py:class}`~dream.scalar_transport.config.FarField` | Inflow value $u_\infty$<br>via upwind criterion |
| {py:class}`~dream.scalar_transport.config.Periodic` | Periodic coupling across<br>matched boundary pairs |
| {py:class}`~dream.scalar_transport.config.Initial` | Initial condition $u(\cdot, 0) = u_0$<br>over domain regions |

## Time Integration

The outer solution loop is selected via `solver.time` (e.g. `'transient'`, `'stationary'`), and
the numerical time integration scheme is chosen via `solver.fem.scheme`
(see the {doc}`time module </modules/time/index>` for the full time infrastructure).

**Implicit schemes** (unconditionally stable — recommended for diffusion-dominated problems
or large time steps):

| Scheme | Order | Key |
|--------|-------|-----|
| {py:class}`~dream.scalar_transport.time.implicit.ImplicitEuler` | 1 | `'implicit_euler'` |
| {py:class}`~dream.scalar_transport.time.implicit.BDF2` | 2 | `'bdf2'` |
| {py:class}`~dream.scalar_transport.time.implicit.SDIRK22` | 2 | `'sdirk22'` |
| {py:class}`~dream.scalar_transport.time.implicit.SDIRK33` | 3 | `'sdirk33'` |

**Explicit schemes** (require CFL condition $\Delta t \leq C\,h / |\vec{b}|$):

| Scheme | Order | Key |
|--------|-------|-----|
| {py:class}`~dream.scalar_transport.time.explicit.ExplicitEuler` | 1 | `'explicit_euler'` |
| {py:class}`~dream.scalar_transport.time.explicit.SSPRK3` | 3 | `'ssprk3'` |
| {py:class}`~dream.scalar_transport.time.explicit.CRK4` | 4 | `'crk4'` |


