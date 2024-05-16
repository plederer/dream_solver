from __future__ import annotations
import time
import pickle
from pathlib import Path
from datetime import datetime
from ngsolve import CF, Redraw, VTKOutput, Mesh, GridFunction
from typing import TYPE_CHECKING, Optional, Generator

from .utils import is_notebook, Formatter
from .configuration import SolverConfiguration, ResultsDirectoryTree
from .sensor import Sensor
from .region import DreamMesh


if is_notebook():
    from ngsolve.webgui import Draw, WebGLScene
else:
    from ngsolve import Draw

if TYPE_CHECKING:
    from pandas import DataFrame
    from .hdg_solver import CompressibleHDGSolver
    from .formulations import Formulation

import logging
logger = logging.getLogger("DreAm.IO")


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
            main_path.mkdir(parents=True, exist_ok=True)
        return main_path

    @property
    def state_path(self) -> Path:
        state_path = self.tree.state_path
        if not state_path.exists():
            state_path.mkdir(parents=True, exist_ok=True)
        return state_path

    @property
    def sensor_path(self):
        sensor_path = self.tree.sensor_path
        if not sensor_path.exists():
            sensor_path.mkdir(parents=True, exist_ok=True)
        return sensor_path

    @property
    def vtk_path(self):
        vtk_path = self.tree.vtk_path
        if not vtk_path.exists():
            vtk_path.mkdir(parents=True, exist_ok=True)
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


class Drawer:

    def __init__(self, formulation: Formulation):
        self._formulation = formulation
        self._scenes: list[WebGLScene] = []

    @property
    def formulation(self):
        return self._formulation

    def draw(self,
             density: bool = True,
             velocity: bool = True,
             pressure: bool = True,
             energy: bool = False,
             temperature: bool = False,
             momentum: bool = False):

        if density:
            self.draw_density()

        if velocity:
            self.draw_velocity()

        if energy:
            self.draw_energy()

        if pressure:
            self.draw_pressure()

        if temperature:
            self.draw_temperature()

        if momentum:
            self.draw_momentum()

    def redraw(self, blocking: bool = False):
        for scene in self._scenes:
            scene.Redraw()
        Redraw(blocking)

    def draw_density(self, label: str = "rho", **kwargs):
        scene = Draw(self.formulation.density(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_momentum(self, label: str = "rho_u", **kwargs):
        scene = Draw(self.formulation.momentum(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_energy(self, label: str = "rho_E", **kwargs):
        scene = Draw(self.formulation.energy(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_pressure(self, label: str = "p", **kwargs):
        scene = Draw(self.formulation.pressure(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_temperature(self, label: str = "T", **kwargs):
        scene = Draw(self.formulation.temperature(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_velocity(self, label: str = "u", **kwargs):
        scene = Draw(self.formulation.velocity(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_vorticity(self, label: str = "omega", **kwargs):
        scene = Draw(self.formulation.vorticity(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_mach_number(self, label: str = "Mach", **kwargs):
        scene = Draw(self.formulation.mach_number(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_speed_of_sound(self, label: str = "c", **kwargs):
        scene = Draw(self.formulation.speed_of_sound(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_deviatoric_strain_tensor(self, label: str = "epsilon", **kwargs):
        scene = Draw(self.formulation.deviatoric_strain_tensor(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_deviatoric_stress_tensor(self, label: str = "tau", **kwargs):
        scene = Draw(self.formulation.deviatoric_stress_tensor(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_heat_flux(self, label: str = "q", **kwargs):
        scene = Draw(self.formulation.heat_flux(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_acoustic_density(self, mean_density: float, label: str = "rho'", **kwargs):
        scene = Draw(self.formulation.density() - mean_density, self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_acoustic_pressure(self, mean_pressure: float, label: str = "p'", **kwargs):
        acc_pressure = self.formulation.pressure() - mean_pressure
        scene = Draw(acc_pressure, self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)
        return acc_pressure

    def draw_particle_velocity(self, mean_velocity: tuple[float, ...], label: str = "u'", **kwargs):
        scene = Draw(self.formulation.velocity() - CF(mean_velocity), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def _append_scene(self, scene):
        if scene is not None:
            self._scenes.append(scene)
