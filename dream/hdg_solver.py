from __future__ import annotations
from typing import Optional, NamedTuple

from ngsolve import *
from . import io
from .sensor import Sensor
from .formulations import formulation_factory, _Formulation
from .configuration import SolverConfiguration, DreAmLogger
from math import isnan

import logging
logger = logging.getLogger('DreAm.Solver')


class IterationError(NamedTuple):
    error: float
    iteration_number: Optional[int]
    time: Optional[float]

    def __repr__(self) -> str:

        error_digit = DreAmLogger._iteration_error_digit
        time_step_digit = DreAmLogger._time_step_digit

        string = f"Iteration Error: {self.error:{error_digit}e}"
        if self.iteration_number is not None:
            string += f" - Iteration Number: {self.iteration_number}"
        if self.time is not None:
            string += f" - Time Step: {self.time:.{time_step_digit}f}"

        return string


class SolverStatus:

    def __init__(self, solver_configuration: SolverConfiguration) -> None:
        self.solver_configuration = solver_configuration
        self.reset()

    @property
    def is_converged(self) -> bool:
        return self._is_converged

    @property
    def is_nan(self) -> bool:
        return self._is_nan

    @property
    def iteration_error(self) -> IterationError:
        return self._iteration_error

    @iteration_error.setter
    def iteration_error(self, value: IterationError):
        min_convergence = self.solver_configuration.convergence_criterion

        self._iteration_error = value
        self._is_converged = value.error < min_convergence

        if isnan(value.error):
            self._is_nan = True
            logger.error("Solution process diverged!")

    def reset_convergence_status(self):
        self._is_converged = False
        self._is_nan = False

    def reset(self):
        self.reset_convergence_status()
        self.iteration_error_list = []

    def check_convergence(self,
                          step_direction,
                          residual,
                          time: Optional[float] = None,
                          iteration_number: Optional[int] = None) -> None:

        error = sqrt(InnerProduct(step_direction, residual)**2)
        self.iteration_error = IterationError(error, iteration_number, time)

        if logger.hasHandlers():
            logger.info(self.iteration_error)


class CompressibleHDGSolver:

    def __init__(self, mesh: Mesh,
                 solver_configuration: SolverConfiguration,
                 directory_tree: Optional[io.ResultsDirectoryTree] = None):

        if directory_tree is None:
            directory_tree = io.ResultsDirectoryTree()

        self._formulation = formulation_factory(mesh, solver_configuration)
        self._status = SolverStatus(self.solver_configuration)
        self._sensors = []
        self._drawer = io.Drawer(self.formulation)
        self._directory_tree = directory_tree

    @property
    def formulation(self) -> _Formulation:
        return self._formulation

    @property
    def mesh(self):
        return self.formulation.mesh

    @property
    def solver_configuration(self):
        return self.formulation.cfg

    @property
    def boundary_conditions(self):
        return self.formulation.dmesh.bcs

    @property
    def domain_conditions(self):
        return self.formulation.dmesh.dcs

    @property
    def sensors(self) -> list[Sensor]:
        return self._sensors

    @property
    def status(self) -> SolverStatus:
        return self._status

    @property
    def drawer(self) -> io.Drawer:
        return self._drawer

    @property
    def directory_tree(self) -> io.ResultsDirectoryTree:
        return self._directory_tree

    def setup(self, reinitialize: bool = True):

        if self.formulation.dmesh.is_grid_stretching:
            self.formulation.dmesh.set_grid_stretching()

        if reinitialize:
            self.formulation.initialize()

        self.residual = self.formulation.gfu.vec.CreateVector()
        self.temporary = self.formulation.gfu.vec.CreateVector()

        self._set_linearform()
        self._set_bilinearform()

        if reinitialize and self.domain_conditions.initial_conditions:
            self._solve_initial()

    def solve_stationary(self,
                         increment_at_iteration: int = 10,
                         increment_time_step_factor: int = 10,
                         state_name: str = "stationary",
                         stop_at_iteration: bool = False) -> bool:

        self.solver_configuration.time.simulation = 'stationary'
        max_iterations = self.solver_configuration.max_iterations

        self.status.reset()
        saver = self.get_saver()

        for it in range(max_iterations):

            self.formulation.update_gridfunctions()
            self._update_pseudo_time_step(it, increment_at_iteration, increment_time_step_factor)

            self._solve_update_step()
            self.status.check_convergence(self.temporary, self.residual, iteration_number=it)

            self.drawer.redraw()

            if stop_at_iteration:
                input("Iteration stopped. Hit any key to continue.")

            if self.status.is_nan:
                break

            self._update_solution()

            if self.status.is_converged:
                break

        for sensor in self.sensors:
            sensor.take_single_sample()

        if self.solver_configuration.save_state:
            saver.save_state(name=state_name)

        self.drawer.redraw()

        self.formulation.update_gridfunctions()

    def solve_transient(self, state_name: str = "transient",  save_state_every_num_step: int = 1):

        self.solver_configuration.time.simulation = 'transient'

        self.status.reset()
        saver = self.get_saver()

        for idx, t in enumerate(self.solver_configuration.time):
            self._solve_timestep(t)

            for sensor in self.sensors:
                sensor.take_single_sample(t)

            if self.solver_configuration.save_state:
                if idx % save_state_every_num_step == 0:
                    saver.save_state(name=f"{state_name}_{t}")

            if self.status.is_nan:
                break

    def solve_perturbation(self) -> GridFunction:

        lf = LinearForm(self.formulation.fes)
        self.formulation.add_perturbation_linearform(lf)
        perturbation_gfu = self._solve_mass(lf)
        for gfu in self.formulation.gridfunctions.values():
            gfu.vec.data += perturbation_gfu.vec

        self.drawer.redraw()
        logger.info("Perturbation applied!")
        return perturbation_gfu

    def add_sensor(self, sensor: Sensor):
        sensor.assign_solver(self)
        self.sensors.append(sensor)

    def get_saver(self, directory_tree: Optional[io.ResultsDirectoryTree] = None) -> io.SolverSaver:
        saver = io.SolverSaver(self)
        if directory_tree is not None:
            saver.tree = directory_tree
        return saver

    def get_loader(self, directory_tree: Optional[io.ResultsDirectoryTree] = None) -> io.SolverLoader:
        loader = io.SolverLoader(self)
        if directory_tree is not None:
            loader.tree = directory_tree
        return loader

    def _set_linearform(self):

        fes = self.formulation.fes
        self.f = LinearForm(fes)
        self.formulation.add_forcing_linearform(self.f)
        self.f.Assemble()

    def _set_bilinearform(self):

        fes = self.formulation.fes

        condense = self.solver_configuration.static_condensation
        mu = self.solver_configuration.dynamic_viscosity

        self.blf = BilinearForm(fes, condense=condense)

        self.formulation.add_time_bilinearform(self.blf)
        self.formulation.add_convective_bilinearform(self.blf)

        if not mu.is_inviscid:
            self.formulation.add_diffusive_bilinearform(self.blf)

        self.formulation.add_boundary_conditions_bilinearform(self.blf)
        self.formulation.add_domain_conditions_bilinearform(self.blf)

    def _solve_mass(self, linearform: LinearForm) -> GridFunction:
        formulation = self.formulation
        gfu = GridFunction(formulation.fes)

        blf = BilinearForm(formulation.fes)
        formulation.add_mass_bilinearform(blf)

        blf.Assemble()
        linearform.Assemble()

        blf_inverse = blf.mat.Inverse(formulation.fes.FreeDofs(), inverse="sparsecholesky")

        gfu.vec.data = blf_inverse * linearform.vec
        return gfu

    def _solve_initial(self):
        lf = LinearForm(self.formulation.fes)
        self.formulation.add_initial_linearform(lf)
        self.formulation.gfu.vec.data = self._solve_mass(lf).vec
        self.formulation.update_gridfunctions(initial_value=True)

    def _solve_update_step(self):

        linear_solver = self.solver_configuration.linear_solver
        fes = self.formulation.fes
        gfu = self.formulation.gfu

        self.blf.Apply(gfu.vec, self.residual)
        self.residual.data -= self.f.vec
        self.blf.AssembleLinearization(gfu.vec)

        inv = self.blf.mat.Inverse(fes.FreeDofs(self.blf.condense), inverse=linear_solver)
        if self.blf.condense:
            self.residual.data += self.blf.harmonic_extension_trans * self.residual
            self.temporary.data = inv * self.residual
            self.temporary.data += self.blf.harmonic_extension * self.temporary
            self.temporary.data += self.blf.inner_solve * self.residual
        else:
            self.temporary.data = inv * self.residual

    def _update_solution(self):
        damping_factor = self.solver_configuration.damping_factor
        gfu = self.formulation.gfu

        gfu.vec.data -= damping_factor * self.temporary

    def _update_pseudo_time_step(self,
                                 iteration_number: int,
                                 increment_at_iteration: int = 10,
                                 increment_time_step_factor: int = 10):

        cfg = self.solver_configuration.time

        max_time_step = cfg.max_step
        old_time_step = cfg.step.Get()

        if max_time_step > old_time_step:
            if (iteration_number % increment_at_iteration == 0) and (iteration_number > 0):
                new_time_step = old_time_step * increment_time_step_factor

                if new_time_step > max_time_step:
                    new_time_step = max_time_step

                cfg.step = new_time_step
                logger.info(f"Successfully updated time step at iteration {iteration_number}")
                logger.info(f"Updated time step 𝚫t = {new_time_step}. Previous time step 𝚫t = {old_time_step}")

    def _solve_timestep(self, time, stop_at_iteration: bool = False) -> bool:

        max_iterations = self.solver_configuration.max_iterations

        self.status.reset_convergence_status()

        for it in range(max_iterations):

            self._solve_update_step()
            self.status.check_convergence(self.temporary, self.residual, time, it)

            if self.status.is_nan:
                break

            self._update_solution()

            if stop_at_iteration:
                input("Iteration stopped. Hit any key to continue.")

            if self.status.is_converged:
                break

        self.formulation.update_gridfunctions()

        self.drawer.redraw()
