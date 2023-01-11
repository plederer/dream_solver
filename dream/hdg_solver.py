from __future__ import annotations
from ngsolve import *
from .configuration import Simulation, SolverConfiguration
from .formulations import formulation_factory, Formulation
from .viscosity import DynamicViscosity
from .io import SolverSaver, SolverLoader
from . import boundary_conditions as bc


class CompressibleHDGSolver():
    def __init__(self, mesh: Mesh, solver_configuration: SolverConfiguration):

        self.mesh = mesh
        self.solver_configuration = solver_configuration
        self._formulation = formulation_factory(mesh, solver_configuration)

    @property
    def formulation(self) -> Formulation:
        return self._formulation

    @property
    def boundary_conditions(self) -> bc.BoundaryConditions:
        return self.formulation.bcs

    @property
    def initial_condition(self) -> bc.InitialCondition:
        return self.formulation.ic

    def setup(self, force: CF = None):

        num_temporary_vectors = self.formulation.time_scheme.num_temporary_vectors

        self.gfu = GridFunction(self.formulation.fes)
        self.gfu_old = tuple(GridFunction(self.formulation.fes) for num in range(num_temporary_vectors))

        self.residual = self.gfu.vec.CreateVector()
        self.temporary = self.gfu.vec.CreateVector()

        self._set_linearform(force)
        self._set_bilinearform()

    def _set_linearform(self, force):

        fes = self.formulation.fes
        TnT = self.formulation.TnT

        bonus_int_order = self.solver_configuration.bonus_int_order_vol

        (_, _, _), (V, _, _) = TnT

        self.f = LinearForm(fes)
        if force is not None:
            self.f += InnerProduct(force, V) * dx(bonus_intorder=bonus_int_order)
        self.f.Assemble()

    def _set_bilinearform(self):

        fes = self.formulation.fes

        condense = self.solver_configuration.static_condensation
        viscosity = self.solver_configuration.dynamic_viscosity

        self.blf = BilinearForm(fes, condense=condense)

        self.formulation.add_time_bilinearform(self.blf, self.gfu_old)
        self.formulation.add_convective_bilinearform(self.blf)

        if viscosity is not DynamicViscosity.INVISCID:
            self.formulation.add_diffusive_bilinearform(self.blf)

        self.formulation.add_bcs_bilinearform(self.blf, self.gfu_old)

    def solve_initial(self):

        fes = self.formulation.fes

        blf = BilinearForm(fes)
        self.formulation.add_mass_bilinearform(blf)

        lf = LinearForm(fes)
        self.formulation.add_ic_linearform(lf)

        blf.Assemble()
        lf.Assemble()

        blf_inverse = blf.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")

        self.gfu.vec.data = blf_inverse * lf.vec
        self.formulation.time_scheme.set_initial_solution(self.gfu, *self.gfu_old)

    def solve_timestep(self, printing=True, stop=False, max_dt=1, stat_step=10):

        simulation = self.solver_configuration.simulation
        linear_solver = self.solver_configuration.linear_solver
        damping_factor = self.solver_configuration.damping_factor
        max_iterations = self.solver_configuration.max_iterations
        min_convergence = self.solver_configuration.convergence_criterion
        dt = self.solver_configuration.time_step
        fes = self.formulation.fes

        for it in range(max_iterations):

            if simulation is Simulation.STATIONARY:
                self.formulation.time_scheme.update_previous_solution(self.gfu, *self.gfu_old)
                if (it % stat_step == 0) and (it > 0) and (dt.Get() < max_dt):
                    c_dt = dt.Get() * 10
                    dt.Set(c_dt)
                    print("new dt = ", c_dt)

            self.blf.Apply(self.gfu.vec, self.residual)
            self.residual.data -= self.f.vec

            try:
                self.blf.AssembleLinearization(self.gfu.vec)
            except Exception as e:
                print("Assemble Linearization failed! Try smaller time step!")
                raise e

            inv = self.blf.mat.Inverse(fes.FreeDofs(self.blf.condense), inverse=linear_solver)
            if self.blf.condense:
                self.residual.data += self.blf.harmonic_extension_trans * self.residual
                self.temporary.data = inv * self.residual
                self.temporary.data += self.blf.harmonic_extension * self.temporary
                self.temporary.data += self.blf.inner_solve * self.residual
            else:
                self.temporary.data = inv * self.residual

            self.gfu.vec.data -= damping_factor * self.temporary

            err = sqrt(InnerProduct(self.temporary, self.residual)**2)

            if printing:
                print(f"Newton iteration: {it}, error = {err}")

            if err < min_convergence:
                break

            if simulation is Simulation.STATIONARY:
                Redraw()
            if stop:
                input()

        self.formulation.time_scheme.update_previous_solution(self.gfu, *self.gfu_old)
        Redraw()

    def draw_solutions(self,
                       density: bool = True,
                       velocity: bool = True,
                       pressure: bool = True,
                       vorticity: bool = False,
                       energy: bool = False,
                       temperature: bool = False,
                       mach: bool = False):

        U, _, Q = self.formulation.get_gridfunction_components(self.gfu)

        if density:
            Draw(self.formulation.density(U), self.mesh, "rho")

        if velocity:
            Draw(self.formulation.velocity(U), self.mesh, "u")

        if pressure:
            Draw(self.formulation.pressure(U), self.mesh, "p")

        if vorticity:
            Draw(self.formulation.vorticity(U, Q), self.mesh, "omega")

        if energy:
            Draw(self.formulation.energy(U), self.mesh, "E")

        if temperature:
            Draw(self.formulation.temperature(U), self.mesh, "T")

        if mach:
            Draw(self.formulation.speed_of_sound(U), self.mesh, "c")
            Draw(self.formulation.mach_number(U), self.mesh, "M")

    def calculate_forces(self, boundary, scale=1):

        region = self.mesh.Boundaries(boundary)
        n = self.formulation.normal

        U, _, Q = self.formulation.get_gridfunction_components(self.gfu)

        stress = -self.formulation.pressure(U) * Id(self.mesh.dim)
        if self.solver_configuration.dynamic_viscosity is not DynamicViscosity.INVISCID:
            stress += self.formulation.deviatoric_stress_tensor(U, Q)

        stress = BoundaryFromVolumeCF(stress)
        forces = Integrate(stress * n, self.mesh, definedon=region)

        return scale * forces

    def get_saver(self, directory_name: str = "results", base_path=None):
        saver = SolverSaver(directory_name, base_path)
        saver.assign_solver(self)
        return saver

    def get_loader(self, directory_name: str = "results", base_path=None):
        loader = SolverLoader(directory_name, base_path)
        loader.assign_solver(self)
        return loader

    def reset_mesh(self, mesh: Mesh):
        self.mesh = mesh
        self._formulation = formulation_factory(mesh, self.solver_configuration)

    def reset_configuration(self, solver_configuration: SolverConfiguration):
        self.solver_configuration = solver_configuration
        self._formulation = formulation_factory(self.mesh, solver_configuration)
