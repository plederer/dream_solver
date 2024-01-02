from __future__ import annotations
import time
import pickle

from .tree import ResultsDirectoryTree
from .logger import dlogger

from typing import TYPE_CHECKING, Optional, Generator
if TYPE_CHECKING:
    from pandas import DataFrame
    from pathlib import Path
    from ngsolve import GridFunction, Mesh
    from dream.mesh import DreamMesh
    from dream.configuration import SolverConfiguration
    from dream.solver import CompressibleHDGSolver


logger = dlogger.getChild("Loader")


class LoadFolderPath:

    __slots__ = ("name")

    def __set_name__(self, owner: Loader, name: str):
        self.name = name

    def __get__(self, loader: Loader, objtype: Loader) -> Path:
        path = getattr(loader.tree, self.name)
        if not path.exists():
            raise ValueError(f"{self.name} '{str(path)}' does not exist.")

        return path


class Loader:

    main_folder: Path = LoadFolderPath()
    state_folder: Path = LoadFolderPath()
    sensor_folder: Path = LoadFolderPath()
    vtk_folder: Path = LoadFolderPath()

    def __init__(self, tree: Optional[ResultsDirectoryTree] = None) -> None:
        if tree is None:
            tree = ResultsDirectoryTree()
        self.tree = tree

    def load_dream_mesh(self, name: str = "dmesh") -> DreamMesh:
        file = self.main_folder.joinpath(name + '.pickle')
        with file.open("rb") as openfile:
            dmesh = pickle.load(openfile)

        return dmesh

    def load_mesh(self, name: str = "mesh") -> Mesh:
        file = self.main_folder.joinpath(name + '.pickle')
        with file.open("rb") as openfile:
            mesh = pickle.load(openfile)

        return mesh

    def load_configuration(self, solver_configuration: SolverConfiguration = None, name: str = "config") -> dict:
        file = self.main_folder.joinpath(name + '.pickle')

        with file.open("rb") as openfile:
            dictionary = pickle.load(openfile)

        if solver_configuration is not None:
            solver_configuration.update(dictionary)

        return dictionary

    def load_gridfunction(self, gfu: GridFunction, name: str = "state") -> None:
        file = self.state_folder.joinpath(name)
        gfu.Load(str(file))

    def load_gridfunction_sequence(self,
                                   gfu: GridFunction,
                                   solver_configuration: SolverConfiguration,
                                   name: str,
                                   sleep_time: float = 0,
                                   load_step: int = 1) -> Generator[float, None, None]:

        for t in solver_configuration.time.range(load_step):
            self.load_gridfunction(gfu, f"{name}_{t}")
            time.sleep(sleep_time)
            yield t

    def load_sensor_data(self, name: str) -> DataFrame:

        file = self.sensor_folder.joinpath(name + ".pickle")
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
        super().load_configuration(self.solver.solver_configuration, name)
        return self.solver.solver_configuration

    def load_gridfunction(self, name: str = "state") -> None:
        super().load_gridfunction(self.solver.formulation.gfu, name)

    def load_time_scheme_gridfunctions(self, name: str = "time_level"):
        gfus = self.solver.formulation._gfus
        for time_level, gfu in gfus.items():
            super().load_gridfunction(gfu, f"{name}_{time_level}")

    def load_gridfunction_sequence(self,
                                   name: str = "transient",
                                   sleep_time: float = 0,
                                   load_step: int = 1, blocking: bool = False) -> Generator[float, None, None]:
        cfg = self.solver.solver_configuration
        for t in cfg.time.range(load_step):
            self.load_gridfunction(name=f"{name}_{t}")
            self.solver.drawer.redraw(blocking)
            time.sleep(sleep_time)
            yield t
