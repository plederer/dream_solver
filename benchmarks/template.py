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
        self.comment = None

    def start(self, save_state: bool = False):

        self.cfg.save_state = save_state
        self.preprocessing()

        with TaskManager():
            self.solution_routine()

        self.postprocessing()

    def preprocessing(self):
        mesh = self.get_mesh()

        self.solver = CompressibleHDGSolver(mesh, self.cfg, self.tree)

        saver = self.solver.get_saver()
        saver.save_mesh()

        self.set_boundary_conditions()
        self.set_domain_conditions()
        self.set_sensors()

    def solution_routine(self):
        self.solver.setup()

        if self.draw:
            self.draw_scenes()

        self.solver.solve_transient()

    def postprocessing(self):
        self.add_meta_data()
        saver = self.solver.get_saver()

        saver.save_dream_mesh()
        saver.save_state_time_scheme()
        saver.save_sensor_data()
        saver.save_configuration(comment=self.comment)

    @abc.abstractmethod
    def get_mesh(self): ...

    @abc.abstractmethod
    def set_boundary_conditions(self): ...

    @abc.abstractmethod
    def set_domain_conditions(self): ...

    def set_sensors(self): ...

    def add_meta_data(self): ...

    def draw_scenes(self): ...
