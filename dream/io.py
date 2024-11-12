from __future__ import annotations
import typing
import logging
import pickle
from pathlib import Path

import ngsolve as ngs
from dream.config import InterfaceConfiguration, UniqueConfiguration, configuration, unique, CONFIG, interface, configuration, ngsdict
from dream._version import acknowledgements, header

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)


class path(configuration):

    def __init__(self, default, fset=None, fget=None, doc: str | None = None):
        super().__init__(default, fset, fget, doc)

    def __set__(self, cfg: CONFIG, path: str | None | Path) -> None:

        if self.fset is not None:
            path = self.fset(cfg, path)

        if path is None:
            path = Path.cwd()
        elif isinstance(path, str):
            path = Path(path)

        old = cfg.data.get(self.__name__, None)

        # Set new path
        if path.is_absolute():
            cfg.data[self.__name__] = path
            return
        elif hasattr(cfg, 'root'):
            cfg.data[self.__name__] = cfg.root.joinpath(path)
        else:
            raise ValueError(f"Path '{path}' is not absolute or {cfg} has no 'root' path!")

        # Update all paths containing the old path
        if old is not None:

            for key, path in cfg.data.items():
                if path == cfg.data[self.__name__]:
                    continue
                elif isinstance(path, Path) and path.is_relative_to(old):
                    name = path.name
                    parent = path.parent
                    while parent != old:
                        name = Path(parent.name).joinpath(name)
                        parent = parent.parent
                    cfg.data[key] = cfg.data[self.__name__].joinpath(name)


class stream(configuration):

    def __set__(self, cfg: CONFIG, stream: bool) -> None:

        if isinstance(stream, str) and stream == self.__name__:
            stream = True

        if not isinstance(stream, bool):
            raise ValueError(f"Pass boolean to activate/deactivate handler")

        if stream:
            cfg.data[self.__name__] = self.fset(cfg, stream)
        elif not stream and self.__name__ in cfg.data:
            del cfg.data[self.__name__]

    def __get__(self, cfg: CONFIG | None, owner: type[CONFIG]):
        if self.__name__ not in cfg.data:
            logger.warning(f"Handler '{self.__name__}' is not activated!")
            return Stream()

        return super().__get__(cfg, owner)


class IOFolders(InterfaceConfiguration, is_interface=True):

    @path(default=None)
    def root(self, path):
        return path

    @path(default='states')
    def states(self, path):
        return path

    @path(default='vtk')
    def vtk(self, path):
        return path

    @path(default='cfg')
    def configuration(self, path):
        return path

    def get_directory_paths(self, pattern: str = "") -> tuple[Path, ...]:
        return tuple(dir for dir in self.root.glob(pattern + "*") if dir.is_dir())

    def get_directory_names(self, pattern: str = "") -> tuple[str, ...]:
        return tuple(dir.name for dir in self.root.glob(pattern + "*") if dir.is_dir())

    root: Path
    states: Path
    vtk: Path
    configuration: Path


class SingleFolders(IOFolders):

    name: str = 'single'


class BenchmarkFolders(IOFolders):

    name: str = 'benchmark'


class Stream(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @configuration(default=None)
    def filename(self, filename: str):
        if filename is None:
            filename = self.name
        return filename

    def initialize(self) -> Stream:
        return self

    def save(self, **kwargs) -> None:
        pass


class MeshStream(Stream):

    name: str = "mesh"

    @property
    def path(self) -> Path:
        path = self.cfg.io.folders.root
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self,  **kwargs) -> None:
        path = self.path.joinpath(self.filename + ".pickle")

        with path.open("wb") as open:
            pickle.dump(self.mesh, open, protocol=pickle.DEFAULT_PROTOCOL)

    def load(self) -> ngs.Mesh:
        file = self.path.joinpath(self.filename + '.pickle')
        with file.open("rb") as openfile:
            return pickle.load(openfile)


class SettingsStream(Stream):

    name: str = "settings"

    @configuration(default=True)
    def pickle(self, pickle: bool):
        return bool(pickle)

    @configuration(default=True)
    def txt(self, txt: bool):
        return bool(txt)

    @configuration(default=False)
    def yaml(self, yaml: bool):
        return bool(yaml)

    @property
    def path(self) -> Path:
        path = self.cfg.io.folders.configuration
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, **kwargs):
        if self.pickle:
            self.save_to_pickle()
        if self.txt:
            self.save_to_txt()
        if self.yaml:
            self.save_to_yaml()

    def save_to_pickle(self, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        path = self.path.joinpath(filename + ".pickle")
        with path.open("wb") as open:
            pickle.dump(self.cfg.to_tree(), open, protocol=pickle.DEFAULT_PROTOCOL)

    def save_to_txt(self, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        path = self.path.joinpath(filename + ".txt")

        with path.open("w") as open:
            txt = [acknowledgements(), ""]
            if hasattr(self.cfg, 'doc'):
                txt += [header("Documentation"), self.cfg.doc, ""]
            txt += [header("Configuration"), repr(self.cfg), ""]

            open.write("\n".join(txt))

    def save_to_yaml(self, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        try:
            import yaml
        except ImportError:
            raise ImportError("Install pyyaml to save configuration as yaml")

        path = self.path.joinpath(filename + ".yaml")
        with path.open("w") as open:
            yaml.dump(self.cfg.to_tree(), open, sort_keys=False)

    def load_from_pickle(self, filename: str | None = None) -> dict:

        if filename is None:
            filename = self.filename

        path = self.path.joinpath(filename + ".pickle")

        with path.open("rb") as open:
            return pickle.load(open)

    pickle: bool
    txt: bool
    yaml: bool


class VTKStream(Stream):

    name: str = "vtk"

    @configuration(default=1)
    def rate(self, i: int):
        i = int(i)
        if i <= 0:
            raise ValueError("Saving frequency must be greater than 0, otherwise consider deactivating the vtk writer!")
        return i

    @configuration(default=2)
    def subdivision(self, subdivision: int):
        return subdivision

    @configuration(default=None)
    def region(self, region):
        return region

    @configuration(default=None)
    def fields(self, fields: dict[str, ngs.CF]):
        if fields is None:
            fields = {}
        return fields

    @property
    def path(self) -> Path:
        path = self.cfg.io.folders.vtk
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def initialize(self) -> VTKStream:

        if not self.fields:
            raise ValueError("No fields to save!")

        self.drawelems = None
        if self.region is not None:
            self.drawelems = ngs.BitArray(self.mesh.ne)
            self.drawelems.Clear()
            for el in self.cfg.mesh.Materials(self.region).Elements():
                self.drawelems[el.nr] = True

        path = self.path.joinpath(self.filename)
        self.writer = ngs.VTKOutput(ma=self.cfg.mesh, coefs=list(self.fields.values()), names=list(
            self.fields.keys()), filename=str(path), subdivision=self.subdivision)

        return self

    def save(self,  t: float | None = None, it: int = 0, **kwargs) -> None:
        if it % self.rate == 0:
            self.writer.Do(t, drawelems=self.drawelems)

    rate: int
    subdivision: int
    region: str
    fields: ngsdict


class StateStream(Stream):

    name: str = "state"

    @configuration(default=1)
    def rate(self, i: int):
        i = int(i)
        if i <= 0:
            raise ValueError(
                "Saving/loading rate must be greater than 0, otherwise consider deactivating the statestream!")
        return i

    @property
    def path(self) -> Path:
        path = self.cfg.io.folders.states
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, t: float | None = None, it: int = 0, **kwargs):

        if it % self.rate == 0:

            filename = self.filename
            if t is not None:
                filename = f"{filename}_{t}"

            self.save_gridfunction(self.cfg.pde.gfu, filename)

    def save_gridfunction(self, gfu: ngs.GridFunction, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        file = self.path.joinpath(filename + ".ngs")

        gfu.Save(str(file))

    def load_gridfunction(self, gfu: ngs.GridFunction, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        file = self.path.joinpath(filename + ".ngs")

        gfu.Load(str(file))

    def load_gridfunction_sequence(self, t: float, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        self.load_gridfunction(self.cfg.pde.gfu, f"{filename}_{t}")

    rate: int


class TimeStateStream(StateStream):

    name: str = "time_state"

    def save(self, t: float, it: int = 0, **kwargs):

        if it % self.rate == 0:
            self.save_gridfunction(t)

    def save_gridfunction(self, t: float, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        for fes, gfus in self.cfg.pde.transient_gfus.items():

            for level, gfu in gfus.items():

                super().save_gridfunction(gfu, f"{filename}_{t}_{fes}_{level}")

    def load_gridfunction(self, t: float, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        for fes, gfu in self.cfg.pde.transient_gfus.items():

            for level, gfu in gfu.items():

                super().load_gridfunction(gfu, f"{filename}_{t}_{fes}_{level}")


class IOConfiguration(UniqueConfiguration):

    cfg: SolverConfiguration
    name: str = "io"

    @interface(default=SingleFolders)
    def folders(self, folders: IOFolders):
        return folders

    @stream(default=False)
    def ngsmesh(self, activate: bool):
        return MeshStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def state(self, activate: bool):
        return StateStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def time_state(self, activate: bool):
        if self.cfg.time.is_stationary:
            raise ValueError("TimeStateStream is not available in stationary mode!")
        return TimeStateStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def settings(self, activate: bool):
        return SettingsStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def vtk(self, activate: bool):
        return VTKStream(cfg=self.cfg, mesh=self.mesh)

    def initialize_streams(self):
        self.pre_routine_streams = [writer.initialize() for writer in self.values()
                                    if isinstance(writer, (MeshStream, SettingsStream))]
        self.routine_streams = [writer.initialize() for writer in self.values()
                                if isinstance(writer, (VTKStream, StateStream, TimeStateStream))]

    def save_pre_routine_streams(self):
        for stream in self.pre_routine_streams:
            stream.save()

    def save_routine_streams(self, t: float | None = None, it: int = 0):
        for stream in self.routine_streams:
            stream.save(t, it)

    def load_gridfunction_routine(
            self, filename: str, rate: int = 1, sleep_time: float = 0) -> typing.Generator[
            float, None, None]:
        import time

        handler = StateStream(cfg=self.cfg, mesh=self.mesh, rate=rate)

        self.cfg.pde.draw()

        for t in self.cfg.time.timer.start(stride=handler.rate):
            handler.load_gridfunction_sequence(t, filename)
            self.cfg.pde.redraw()
            time.sleep(sleep_time)

            yield t

    folders: SingleFolders | BenchmarkFolders
    ngsmesh: MeshStream
    state: StateStream
    time_state: TimeStateStream
    settings: SettingsStream
    vtk: VTKStream
