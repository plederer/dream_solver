from __future__ import annotations
import pickle
from datetime import datetime
from ngsolve import *

from .tree import ResultsDirectoryTree
from ..utils import DreAmLogger, Formatter

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from pathlib import Path
    from ..configuration import SolverConfiguration
    from ..mesh import DreamMesh
    from ..sensor import Sensor
    from ..solver import CompressibleHDGSolver

logger = DreAmLogger.get_logger("Saver")


class Saver:

    def __init__(self, tree: Optional[ResultsDirectoryTree] = None) -> None:
        if tree is None:
            tree = ResultsDirectoryTree()
        self.tree = tree
        self._vtk = None

    @property
    def main_path(self) -> Path:
        main_path = self.tree.main_path
        if not main_path.exists():
            main_path.mkdir(parents=True)
        return main_path

    @property
    def state_path(self) -> Path:
        state_path = self.tree.state_path
        if not state_path.exists():
            state_path.mkdir(parents=True)
        return state_path

    @property
    def sensor_path(self):
        sensor_path = self.tree.sensor_path
        if not sensor_path.exists():
            sensor_path.mkdir(parents=True)
        return sensor_path

    @property
    def vtk_path(self):
        vtk_path = self.tree.vtk_path
        if not vtk_path.exists():
            vtk_path.mkdir(parents=True)
        return vtk_path

    def initialize_vtk_handler(self,
                               fields: dict[str, GridFunction],
                               mesh: Mesh,
                               filename: str = "vtk",
                               subdivision: int = 2,
                               **kwargs) -> None:

        self._vtk = VTKOutput(ma=mesh,
                              coefs=list(fields.values()),
                              names=list(fields.keys()),
                              filename=str(self.vtk_path.joinpath(filename)),
                              subdivision=subdivision, **kwargs)

    def save_dream_mesh(self, dmesh: DreamMesh,
                        name: str = "dmesh",
                        suffix: str = ".pickle") -> None:

        file = self.main_path.joinpath(name + suffix)
        with file.open("wb") as openfile:
            pickle.dump(dmesh, openfile)

    def save_mesh(self, mesh: Mesh,
                  name: str = "mesh",
                  suffix: str = ".pickle") -> None:

        file = self.main_path.joinpath(name + suffix)
        with file.open("wb") as openfile:
            pickle.dump(mesh, openfile)

    def save_configuration(self,
                           configuration: SolverConfiguration,
                           name: str = "config",
                           comment: Optional[str] = None,
                           save_pickle: bool = True,
                           save_txt: bool = True):

        if save_pickle:
            self._save_pickle_configuration(configuration, name)
        if save_txt:
            self._save_txt_configuration(configuration, name, comment)

    def save_state(self, gfu: GridFunction, name: str = "state") -> None:
        file = self.state_path.joinpath(name)
        gfu.Save(str(file))

    def save_sensor_data(self, sensor: Sensor, save_dataframe: bool = True):

        file = self.sensor_path.joinpath(f"{sensor.name}.csv")

        df = sensor.to_dataframe_all()
        df.to_csv(file)

        if save_dataframe:
            file = self.sensor_path.joinpath(f"{sensor.name}.pickle")
            with file.open("wb") as openfile:
                pickle.dump(df, openfile)

    def save_vtk(self, time: float = -1, region=None, **kwargs) -> Path:
        if self._vtk is None:
            raise ValueError("Call 'saver.initialize_vtk_handler()' before saving vtk")
        return self._vtk.Do(time, drawelems=region, **kwargs)

    def _save_txt_configuration(self, configuration: SolverConfiguration, name: str, comment: Optional[str] = None):

        formatter = Formatter()
        formatter.COLUMN_RATIO = (0.3, 0.7)

        formatter.header("Compressible HDG Solver").newline()
        formatter.entry("Authors", "Philip Lederer, Jan Ellmenreich")
        formatter.entry("Institute", "Analysis and Scientific Computing (2022 - )")
        formatter.entry("University", "TU Wien")
        formatter.entry("Funding", "FWF (Austrian Science Fund) - P35391N")
        formatter.entry("Github", "https://github.com/plederer/dream_solver")
        formatter.entry("Date", datetime.now().strftime('%Hh:%Mm:%Ss %d/%m/%Y')).newline()

        formatter.add(configuration)

        if comment is not None:
            formatter.header("Comment").newline()
            formatter.text(comment).newline()

        formatter.add(self.tree)

        file = self.main_path.joinpath(name + ".txt")

        with file.open("w") as openfile:
            openfile.write(formatter.output)

    def _save_pickle_configuration(self, configuration: SolverConfiguration, name: str, suffix: str = ".pickle"):
        file = self.main_path.joinpath(name + suffix)
        dictionary = configuration.to_dict()

        with file.open("wb") as openfile:
            pickle.dump(dictionary, openfile)

    def __repr__(self) -> str:
        return repr(self.tree)


class SolverSaver(Saver):

    def __init__(self, solver: CompressibleHDGSolver):
        self.solver = solver
        super().__init__(solver.directory_tree)

    def initialize_vtk_handler(self,
                               fields: dict[str, GridFunction],
                               filename: str = "vtk",
                               subdivision: int = 2,
                               **kwargs) -> None:
        super().initialize_vtk_handler(fields, self.solver.mesh, filename, subdivision, **kwargs)

    def save_dream_mesh(self, name: str = "dmesh", suffix: str = ".pickle") -> None:
        dmesh = self.solver.dmesh
        super().save_dream_mesh(dmesh, name, suffix)

    def save_mesh(self, name: str = "mesh", suffix: str = ".pickle") -> None:
        mesh = self.solver.formulation.mesh
        super().save_mesh(mesh, name, suffix)

    def save_configuration(self,
                           configuration: SolverConfiguration = None,
                           name: str = "config",
                           comment: Optional[str] = None,
                           save_pickle: bool = True,
                           save_txt: bool = True):
        if configuration is None:
            configuration = self.solver.solver_configuration
        super().save_configuration(configuration, name, comment, save_pickle, save_txt)

    def save_state(self, gfu: GridFunction = None, name: str = "state"):
        if gfu is None:
            gfu = self.solver.formulation.gfu
        super().save_state(gfu, name)

    def save_state_time_scheme(self, name: str = "time_level"):
        gfus = self.solver.formulation._gfus
        t = self.solver.solver_configuration.time.t.Get()
        for time_level, gfu in gfus.items():
            super().save_state(gfu, f"{name}_{t}_{time_level}")

    def save_sensor_data(self, sensor: Optional[Sensor] = None, save_dataframe: bool = True):

        if sensor is None:
            sensors = self.solver.sensors
        elif isinstance(sensor, Sensor):
            sensors = [sensor]
        else:
            sensors = list(sensor)

        for sensor in sensors:
            super().save_sensor_data(sensor, save_dataframe)
