from __future__ import annotations
import abc
from dream import *
from ngsolve import *


class Benchmark(abc.ABC):

    def __init__(self, cfg: SolverConfiguration, tree: ResultsDirectoryTree, draw: bool = False) -> None:

        self.cfg = SolverConfiguration()
        self.cfg.update(cfg)

        self.tree = ResultsDirectoryTree()
        self.tree.update(tree)

        self.draw = draw

    def start(self, save_state: bool = False):

        self.cfg.save_state = save_state

        mesh = self.get_mesh()

        solver = CompressibleHDGSolver(mesh, self.cfg, self.tree)

        saver = solver.get_saver()
        saver.save_mesh()

        self.set_boundary_conditions(solver)
        self.set_domain_conditions(solver)
        self.set_sensors(solver)

        self.add_meta_data(self.cfg)
        saver.save_configuration()
        saver.save_dream_mesh()

        with TaskManager():
            self.start_solution_routine(solver)

        saver.save_state_time_scheme()
        saver.save_sensor_data()

    def start_solution_routine(self, solver: CompressibleHDGSolver):
        solver.setup()

        if self.draw:
            self.draw_scenes(solver)

        solver.solve_transient()

    @abc.abstractmethod
    def get_mesh(self): ...

    @abc.abstractmethod
    def set_boundary_conditions(self, solver: CompressibleHDGSolver): ...

    @abc.abstractmethod
    def set_domain_conditions(self, solver: CompressibleHDGSolver): ...

    def set_sensors(self, solver: CompressibleHDGSolver): ...

    def add_meta_data(self, cfg: SolverConfiguration): ...

    def draw_scenes(self, solver: CompressibleHDGSolver): ...
