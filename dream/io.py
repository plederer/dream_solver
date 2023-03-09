from __future__ import annotations

import pickle
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from .utils.formatter import Formatter

if TYPE_CHECKING:
    from ngsolve import Mesh
    from pandas import DataFrame
    from .sensor import Sensor
    from .hdg_solver import CompressibleHDGSolver
    from .configuration import SolverConfiguration


class ResultsDirectoryTree:

    def __init__(self,
                 directory_name: str = "results",
                 state_directory_name: str = "states",
                 sensor_directory_name: str = "sensor",
                 parent_path: Optional[Path] = None) -> None:

        self.parent_path = parent_path
        self.directory_name = directory_name
        self.state_directory_name = state_directory_name
        self.sensor_directory_name = sensor_directory_name

    @property
    def main_path(self) -> Path:
        return self.parent_path.joinpath(self.directory_name)

    @property
    def state_path(self) -> Path:
        return self.main_path.joinpath(self.state_directory_name)

    @property
    def sensor_path(self) -> Path:
        return self.main_path.joinpath(self.sensor_directory_name)

    @property
    def parent_path(self) -> Path:
        return self._parent_path

    @parent_path.setter
    def parent_path(self, base_path: Path):

        if base_path is None:
            self._parent_path = Path.cwd()

        elif isinstance(base_path, (str, Path)):
            base_path = Path(base_path)

            if not base_path.exists():
                raise ValueError(f"Path {base_path} does not exist!")

            self._parent_path = base_path

        else:
            raise ValueError(f"Type {type(base_path)} not supported!")

    def __repr__(self) -> str:
        formatter = Formatter()
        formatter.COLUMN_RATIO = (0.2, 0.8)
        formatter.header("Results Directory Tree").newline()
        formatter.entry("Path", str(self.parent_path))
        formatter.entry("Main", f"{self.parent_path.stem}/{self.directory_name}")
        formatter.entry("State", f"{self.parent_path.stem}/{self.directory_name}/{self.state_directory_name}")
        formatter.entry("Sensor", f"{self.parent_path.stem}/{self.directory_name}/{self.sensor_directory_name}")

        return formatter.output


class SolverLoader:

    def __init__(self, solver: Optional[CompressibleHDGSolver] = None, tree: Optional[ResultsDirectoryTree] = None):
        if tree is None:
            tree = ResultsDirectoryTree()
        self.tree = tree
        self.solver = solver

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

    def load_mesh(self, name: str = "mesh") -> Mesh:
        file = self.main_path.joinpath(name + '.pickle')
        with file.open("rb") as openfile:
            mesh = pickle.load(openfile)

        if self.solver is not None:
            self.solver.reset_mesh(mesh)

        return mesh

    def load_configuration(self, name: str = "config") -> SolverConfiguration:
        file = self.main_path.joinpath(name + '.pickle')
        with file.open("rb") as openfile:
            config = pickle.load(openfile)

        if self.solver is not None:
            self.solver.reset_configuration(config)

        return config

    def load_state(self, name: str):

        if self.solver is None:
            raise Exception("Assign solver before loading state")

        file = self.state_path.joinpath(name)
        self.solver.gfu.Load(str(file))

    def load_sensor_data(self, name: str) -> DataFrame:

        file = self.sensor_path.joinpath(name + ".pickle")
        if not file.exists():
            raise Exception(f"Can not load {file}, as it does not exists!")
        with file.open("rb") as openfile:
            df = pickle.load(openfile)

        return df

    def __repr__(self) -> str:
        return repr(self.tree)


class SolverSaver:

    def __init__(self, solver: CompressibleHDGSolver, tree: Optional[ResultsDirectoryTree] = None):
        if tree is None:
            tree = ResultsDirectoryTree()
        self.tree = tree
        self.solver = solver

    @property
    def main_path(self) -> Path:
        main_path = self.tree.main_path
        if not main_path.exists():
            main_path.mkdir()
        return main_path

    @property
    def solutions_path(self) -> Path:
        state_path = self.tree.state_path
        if not state_path.exists():
            state_path.mkdir()
        return state_path

    def save_mesh(self, mesh: Optional[Mesh] = None,
                  name: str = "mesh",
                  suffix: str = ".pickle"):

        if mesh is None:
            mesh = self.solver.mesh

        file = self.main_path.joinpath(name + suffix)
        with file.open("wb") as openfile:
            pickle.dump(mesh, openfile)

    def save_configuration(self,
                           configuration: Optional[SolverConfiguration] = None,
                           name: str = "config",
                           comment: Optional[str] = None,
                           save_pickle: bool = True,
                           save_txt: bool = True):

        if configuration is None:
            configuration = self.solver.solver_configuration

        if save_pickle:
            self._save_pickle_configuration(configuration, name)
        if save_txt:
            self._save_txt_configuration(configuration, name, comment)

    def save_state(self, name: str = "state"):
        file = self.solutions_path.joinpath(name)
        self.solver.gfu.Save(str(file))

    def save_sensors_data(self,
                          sensors: Optional[list[Sensor]] = None,
                          time_index=None,
                          save_dataframe: bool = True):

        if sensors is None:
            sensors = self.solver.sensors

        for sensor in sensors:
            filename = sensor.name
            file = sensor.sensor_path.joinpath(filename + '.csv')

            df = sensor.convert_samples_to_dataframe(time_index)
            df.to_csv(file)

            if save_dataframe:
                file = sensor.sensor_path.joinpath(filename + '.pickle')
                with file.open("wb") as openfile:
                    pickle.dump(df, openfile)

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
        with file.open("wb") as openfile:
            pickle.dump(configuration, openfile)

    def __repr__(self) -> str:
        return repr(self.tree)
