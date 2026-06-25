The {py:mod}`dream.time` module provides the time-stepping infrastructure that is shared across all
`dream` solvers. It separates two concerns:

- A {py:class}`~dream.time.TimeSchemes` discretizes the time derivative of the semi-discrete problem
  arising from the spatial discretization and advances the solution by a single time level or
  Runge-Kutta stage. Concrete schemes (explicit, implicit, IMEX) are implemented per solver, e.g. in
  {py:mod}`dream.scalar_transport.time` and {py:mod}`dream.compressible_flow.conservative.time`.
- A {py:class}`~dream.time.TimeRoutine` drives the overall solution process: it owns the main solution
  loop, manages I/O, and reports solver convergence. The available routines are
  {py:class}`~dream.time.StationaryRoutine`, {py:class}`~dream.time.TransientRoutine`,
  {py:class}`~dream.time.PseudoTimeSteppingRoutine`, and the IMEX routines derived from
  {py:class}`~dream.time.IMEXTimeRoutine`.

### Stationary and transient routines

For steady-state problems, {py:class}`~dream.time.StationaryRoutine` solves the discrete problem
directly, without time-stepping. For time-dependent problems, {py:class}`~dream.time.TransientRoutine`
marches the solution forward over a fixed time interval $(t_0, t_{\mathrm{end}})$ held by a
{py:class}`~dream.time.Timer`, advancing $t_n \to t_{n+1} = t_n + \Delta t$ until $t_{\mathrm{end}}$ is
reached. {py:class}`~dream.time.PseudoTimeSteppingRoutine` instead marches a stationary problem towards
steady state using pseudo-time continuation, in which an artificial time derivative is added and the
(pseudo) time step is progressively increased, improving the robustness of the nonlinear solver when
starting far from the solution.

### Geometry-split implicit-explicit (IMEX) time integration

In many flow problems, only a small part of the computational mesh is responsible for the
geometry-induced stiffness that limits the stable time step of an explicit scheme, for example highly
refined regions near walls or small geometric features. The IMEX routines derived from
{py:class}`~dream.time.IMEXTimeRoutine` exploit this by coupling two independently configured
{py:class}`~dream.solver.SolverConfiguration`\ s, held by an `IMEXTimeRoutine` as `cfg_implicit` and
`cfg_explicit`. The routine itself does not partition a mesh into implicit/explicit regions; rather, it
assumes that the two complementary meshes,
\begin{align}
    \mesh = \mesh^{im} \cup \mesh^{ex}, \qquad \Gamma_i = \mesh^{im} \cap \mesh^{ex},
\end{align}
with interface $\Gamma_i$, have already been constructed and assigned to `cfg_implicit` and
`cfg_explicit`, respectively, e.g. with the stiff (typically small) region assigned to `cfg_implicit`
and the remaining, non-stiff region to `cfg_explicit`. Stiff regions are typically discretized with an
implicit hybridizable discontinuous Galerkin (HDG) scheme, while non-stiff regions are discretized with
a (standard) discontinuous Galerkin (DG) scheme; however, the routine itself only assumes that
`cfg_implicit` is solved implicitly and `cfg_explicit` explicitly in time, not any specific spatial
discretization. In particular, both regions may equally well use a DG discretization, with only the
time treatment (implicit vs. explicit) differing between them.

The two solutions are coupled weakly and conservatively across $\Gamma_i$ by appropriate interface
conditions, while the temporal synchronization between the implicit and explicit schemes is achieved
through additive Runge-Kutta (ARK) methods: the implicit part is advanced with a singly diagonally
implicit Runge-Kutta (SDIRK) method, and the explicit part with a standard explicit Runge-Kutta (ERK)
method. At every stage, the explicit solution is updated first, using the implicit solution of the
previous stage, and the implicit solution is then updated using the just-computed explicit solution.

Two synchronization strategies are available:

- {py:class}`~dream.time.SynchronizedIMEXTimeRoutine`: the implicit and explicit schemes share the same
  (global) time step $\Delta t$ and their stage times coincide, $\overline{c}_{i+1} = c_i$, following the
  classical structure of an ARK method. This is the strategy used for the ARS-type IMEX schemes for
  compressible flows.
- {py:class}`~dream.time.PCIMEXTimeRoutine` and {py:class}`~dream.time.LinearPCIMEXTimeRoutine`: a
  predictor-corrector strategy in which the global (implicit) time step may be a larger integer multiple
  of the local (explicit) time step, allowing the explicit scheme to sub-cycle within an implicit stage.
  During the sub-cycling, the interface value provided to the explicit scheme is either held frozen at
  $\vec{U}_n^{im}$ (`PCIMEXTimeRoutine`) or linearly interpolated in time between the implicit solution at
  the start and end of the stage (`LinearPCIMEXTimeRoutine`), before the implicit (corrector) solution is
  recomputed using the updated explicit state.

By restricting the (more expensive) implicit solve to the stiff region $\mesh^{im}$, while advancing the
remainder of the domain $\mesh^{ex}$ with the cheaper explicit scheme, the IMEX approach increases the
overall stable time step compared to a fully explicit discretization, at the cost of the additional
implicit solve. The net benefit therefore depends both on an effective mesh partitioning and an
efficient implicit solver.
