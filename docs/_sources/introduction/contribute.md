# Contribute

Thank you for your interest in contributing! To get started, you can use the source files located in the `dream` directory.

1. **Clone the Repository**  
    Make sure you have a local copy of the project.

2. **Navigate to Source Files**  
    The main source code for the dream solver is in the `dream` folder relative to this documentation file.

3. **Make Your Changes**  
    - Add features, fix bugs, or improve documentation within the `dream` or `docs` folder.
    - Follow the project's coding standards and guidelines.

4. **Test Your Changes**  
    Ensure your modifications do not break existing functionality by writing appropriate `tests`. 

5. **Submit a Pull Request**  
    - Push your changes to your fork.
    - Open a pull request with a clear description of your contribution.

We appreciate your help in improving dream!

```{tip}
- Reference relevant issues in your pull request.
- Ask questions or request feedback if needed.
```

# Structure

Every `dream` solver is a `SolverConfiguration` that composes a set of interchangeable
sub-objects, each declared with the `@dream_configuration` decorator. This decorator
turns a property into a configurable slot: assigning a string key (e.g. `solver.fem = 'hdg'`)
selects the corresponding concrete class, while assigning an instance gives full control.

## Solvers

The two main solvers are
{py:class}`~dream.scalar_transport.solver.ScalarTransportSolver` and
{py:class}`~dream.compressible_flow.solver.CompressibleFlowSolver`. Both follow the same
pattern: a `fem` slot for the finite element method, a `riemann_solver` slot, and a `time`
slot for the outer solution loop. Physical parameters (Reynolds number, Mach number, etc.) are
likewise declared as `@dream_configuration` properties with validation in their setters.

```python
class SomeCFDSolver(SolverConfiguration):

    @dream_configuration
    def fem(self) -> FiniteElementMethod:
        return self._fem

    @fem.setter
    def fem(self, fem):
        OPTIONS = [HDG, DG]
        self._fem = self._get_configuration_option(fem, OPTIONS, FiniteElementMethod)

    @dream_configuration
    def time(self) -> TimeRoutine:
        return self._time

    @time.setter
    def time(self, time):
        OPTIONS = [StationaryRoutine, TransientRoutine, PseudoTimeSteppingRoutine]
        self._time = self._get_configuration_option(time, OPTIONS, TimeRoutine)
```

## Two-level time structure

`dream` separates time integration into two distinct layers:

- **`solver.time`** — selects the *outer solution loop*
  ({py:class}`~dream.time.TransientRoutine`, {py:class}`~dream.time.StationaryRoutine`, or
  {py:class}`~dream.time.PseudoTimeSteppingRoutine`). This controls *how* the solver is driven
  (marching in time, solving stationary, or using pseudo-time continuation) and owns the
  {py:class}`~dream.time.Timer` via `solver.time.timer`.

- **`solver.fem.scheme`** — selects the *numerical time integration scheme*
  ({py:class}`~dream.time.TimeSchemes`). This is a property on the finite element method
  (`HDG`, `DG`, `ConservativeHDG`, …) and determines *how* the time derivative is
  discretised (e.g. `'implicit_euler'`, `'bdf2'`, `'sdirk22'`, `'ssprk3'`).

A typical transient setup therefore looks like:

```python
solver.time               = 'transient'      # outer loop
solver.fem.scheme         = 'bdf2'           # numerical scheme
solver.time.timer.interval = (0.0, 1.0)
solver.time.timer.step     = 1e-3
```

## Finite element methods

Each FEM class (e.g. {py:class}`~dream.scalar_transport.spatial.HDG`) implements the
`initialize` chain that builds finite element spaces, trial/test functions, and symbolic
bilinear/linear forms:

```python
class FiniteElementMethod:

    def initialize(self) -> None:
        self.initialize_finite_element_spaces()
        self.initialize_trial_and_test_functions()
        self.initialize_gridfunctions()
        self.initialize_time_scheme_gridfunctions()
        self.set_boundary_conditions()
        self.set_initial_conditions()
        self.initialize_symbolic_forms()
```

## Solution routines

{py:class}`~dream.time.TransientRoutine` advances the solution by calling
`solver.fem.scheme.solve_current_time_level()` at each step and yields the current time for
optional post-processing. Calling `solver.solve()` is the simplest interface — it calls
`solver.time.start_solution_routine()` internally and blocks until the simulation is complete:

```python
solver.setup()
solver.solve()
```

For finer control (e.g. custom output at every step), iterate over
`solver.time.start_solution_routine()` directly:

```python
solver.setup()
for t in solver.time.start_solution_routine():
    print(f"t = {t:.4f}")
```

