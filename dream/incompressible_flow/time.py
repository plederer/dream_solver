from __future__ import annotations
import typing
import ngsolve as ngs

from dream.time import Scheme, TimeSchemes, Scheme, StationaryRoutine, TransientRoutine
from dream.config import Log, Integrals


if typing.TYPE_CHECKING:
    from dream.incompressible_flow.solver import IncompressibleFlowSolver


# Define solving schemes
class StationaryScheme(Scheme):

    root: IncompressibleFlowSolver

    name: str = "direct"

    def assemble(self):

        fem = self.root.fem

        self.blf = ngs.BilinearForm(fem.fes)
        self.add_sum_of_integrals(self.blf, fem.blf)

        self.lf = ngs.LinearForm(fem.fes)
        self.add_sum_of_integrals(self.lf, fem.lf)

        self.lf.Assemble()

        if not self.root.dynamic_viscosity.is_linear_model or self.root.convection:
            # self.blf.Apply(fem.gfu.vec, self.lf.vec)
            fem.solver.initialize_nonlinear_routine(self.blf, fem.gfu, self.lf.vec)

        else:
            # Add dirichlet boundary conditions
            self.blf.Assemble()
            self.lf.vec.data -= self.blf.mat * fem.gfu.vec

    def solve_stationary(self) -> typing.Generator[Log, None, None]:

        fem = self.root.fem

        if not self.root.dynamic_viscosity.is_linear_model or self.root.convection:

            for log in fem.solver.solve_nonlinear_system():
                yield log
        else:
            self.inv = fem.solver.get_inverse(self.blf, fem.fes)
            fem.gfu.vec.data += self.inv * self.lf.vec

            yield {}


class IMEX(TimeSchemes):

    name: str = "imex"
    number_of_steps: int = 2
    number_of_stages: int = 1
    time_of_stages: tuple[float] = (0.0, 1.0)

    def assemble(self) -> None:

        if not self.root.convection:
            raise ValueError("The IMEX scheme only support convection terms!")

        fem = self.root.fem
        condense = fem.static_condensation

        self.blf = ngs.BilinearForm(fem.fes, condense=condense)
        implicit = self.parse_sum_of_integrals(fem.blf, exclude_terms=['convection'])
        self.add_sum_of_integrals(self.blf, implicit, 'implicit')

        self.stokes = ngs.BilinearForm(fem.fes)
        stokes = self.parse_sum_of_integrals(fem.blf, exclude_terms=['convection', 'mass'])
        self.add_sum_of_integrals(self.stokes, stokes, 'stokes')

        self.convection = ngs.BilinearForm(fem.fes, nonassemble=True)
        convection = self.parse_sum_of_integrals(fem.blf, include_terms=['convection'])
        self.add_sum_of_integrals(self.convection, convection, 'convection')

        self.lf = ngs.LinearForm(fem.fes)
        self.add_sum_of_integrals(self.lf, fem.lf)
        self.lf.Assemble()

        self.tmp = fem.gfu.vec.CreateVector()
        self.tmp[:] = 0.0

        if not self.root.dynamic_viscosity.is_linear_model:
            # self.blf.Apply(fem.gfu.vec, self.lf.vec)
            fem.solver.initialize_nonlinear_routine(self.blf, fem.gfu, self.lf.vec)

        else:
            # Add dirichlet boundary conditions
            self.blf.Assemble()
            self.stokes.Assemble()
            self.inv = fem.solver.get_inverse(self.blf, fem.fes)

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals) -> None:
        u, v = self.root.fem.TnT['u']

        blf['u']['mass'] = ngs.InnerProduct(u/self.dt, v) * ngs.dx

    def solve_current_time_level(self, t0: float) -> typing.Generator[Log, None, None]:

        fem = self.root.fem

        if not self.root.dynamic_viscosity.is_linear_model:
            for log in fem.solver.solve_nonlinear_system():
                yield log
        else:

            self.convection.Apply(fem.gfu.vec, self.tmp)

            self.t.Set(t0 + self.dt.Get())
            self.tmp.data -= self.stokes.mat * fem.gfu.vec
            
            fem.solver.solve_linear_system(self.blf, fem.gfu, self.tmp, operator="-=")
            
            # fem.gfu.vec.data -= self.inv * self.tmp



            yield {"t": self.t.Get()}