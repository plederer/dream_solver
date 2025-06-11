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

```python
class SomeCFDSolver:

    @dream_configuration
    def fem(self) -> FiniteElementMethod:
        """ Your favourite finite element method """
        return FiniteElementMethod(self)

    @dream_configuration
    def time(self) -> TransientRoutine:
        """ Your favourite time routine """
        return TransientRoutine(self)

    @dream_configuration
    def reynolds_number(self) -> float:
        r""" Reynolds number of the flow.
        
            .. math::
                Re = \frac{\rho U L}{\mu}
        """
        return self._reynolds_number

    @reynolds_number.setter
    def reynolds_number(self, value):

        value = float(value)

        if value <= 0:
            raise ValueError("Reynolds number must be positive.")
            
        self._reynolds_number = value
```

See e.g. {class}`~dream.compressible.solver.CompressibleFlowSolver` for a complete example for a documentation of a CFD solver using the dream package.

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

class TransientRoutine:

    def start_solution_routine(self) -> Generator[float, None, None]:
        """ Starts the solution routine for the CFD simulation. """

        scheme = self.root.fem.scheme

        scheme.Assemble()

        with self.root.io as io:

            # Solution routine starts here
            for t in self.timer():

                scheme.solve_current_time_level()

                yield t

                io.save()
                io.redraw()
            # Solution routine ends here
```

