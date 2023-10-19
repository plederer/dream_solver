from __future__ import annotations
import time
import pickle

from .tree import ResultsDirectoryTree
from ..configuration import SolverConfiguration
from ..utils import DreAmLogger


from typing import TYPE_CHECKING, Optional, Generator
if TYPE_CHECKING:
    from pandas import DataFrame
    from ngsolve import *
    from pathlib import Path
    from ..mesh import DreamMesh
    from ..solver import CompressibleHDGSolver


logger = DreAmLogger.get_logger("Loader")


class Loader:

    def __init__(self, tree: Optional[ResultsDirectoryTree] = None) -> None:
        if tree is None:
            tree = ResultsDirectoryTree()
        self.tree = tree

    @property
    def main_path(self) -> Path:
        main_path = self.tree.main_path
        if not main_path.exists():
            raise Exception(f"Can not load from {main_path}, as the path does not exist.")
        return main_path

    @property
    def state_path(self) -> Path:
        state_path = self.tree.state_path
        if not state_path.exists():
            raise Exception(f"Can not load from {state_path}, as the path does not exist.")
        return state_path

    @property
    def sensor_path(self) -> Path:
        sensor_path = self.tree.sensor_path
        if not sensor_path.exists():
            raise Exception(f"Can not load from {sensor_path}, as the path does not exist.")
        return sensor_path

    @property
    def vtk_path(self) -> Path:
        vtk_path = self.tree.vtk_path
        if not vtk_path.exists():
            raise Exception(f"Can not load from {vtk_path}, as the path does not exist.")
        return vtk_path

    def load_dream_mesh(self, name: str = "dmesh") -> DreamMesh:
        file = self.main_path.joinpath(name + '.pickle')
        with file.open("rb") as openfile:
            dmesh = pickle.load(openfile)

        return dmesh

    def load_mesh(self, name: str = "mesh") -> Mesh:
        file = self.main_path.joinpath(name + '.pickle')
        with file.open("rb") as openfile:
            mesh = pickle.load(openfile)

        return mesh

    def load_configuration(self, name: str = "config") -> SolverConfiguration:
        solver_configuration = SolverConfiguration()
        file = self.main_path.joinpath(name + '.pickle')

        with file.open("rb") as openfile:
            dictionary = pickle.load(openfile)

        solver_configuration.update(dictionary)
        return solver_configuration

    def load_state(self, gfu: GridFunction, name: str = "state") -> None:
        file = self.state_path.joinpath(name)
        gfu.Load(str(file))

    def load_state_time_sequence(self,
                                 gfu: GridFunction,
                                 solver_configuration: SolverConfiguration,
                                 name: str,
                                 sleep_time: float = 0,
                                 load_step: int = 1) -> Generator[float, None, None]:

        for t in solver_configuration.time.range(load_step):
            self.load_state(gfu, f"{name}_{t}")
            time.sleep(sleep_time)
            yield t

    def load_sensor_data(self, name: str) -> DataFrame:

        file = self.sensor_path.joinpath(name + ".pickle")
        if not file.exists():
            raise Exception(f"Can not load {file}, as it does not exists!")
        with file.open("rb") as openfile:
            df = pickle.load(openfile)

        return df

    def __repr__(self) -> str:
        return repr(self.tree)


class SolverLoader(Loader):

    def __init__(self, solver: CompressibleHDGSolver):
        self.solver = solver
        super().__init__(solver.directory_tree)

    def load_configuration(self, name: str = "config") -> SolverConfiguration:
        config = super().load_configuration(name)
        self.solver.solver_configuration.update(config)
        return self.solver.solver_configuration

    def load_state(self, gfu: GridFunction = None, name: str = "state") -> None:
        formulation = self.solver.formulation
        if gfu is None:
            gfu = formulation.gfu
        super().load_state(gfu, name)

    def load_state_time_scheme(self, name: str = "time_level"):
        gfus = self.solver.formulation._gfus
        for time_level, gfu in gfus.items():
            super().load_state(gfu, f"{name}_{time_level}")

    def load_state_time_sequence(self,
                                 name: str = "transient",
                                 sleep_time: float = 0,
                                 load_step: int = 1, blocking: bool = False) -> Generator[float, None, None]:
        cfg = self.solver.solver_configuration
        for t in cfg.time.range(load_step):
            self.load_state(name=f"{name}_{t}")
            self.solver.drawer.redraw(blocking)
            time.sleep(sleep_time)
            yield t
