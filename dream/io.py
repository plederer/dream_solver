from __future__ import annotations
import typing
import pickle
from pathlib import Path

import ngsolve as ngs
from dream.config import InterfaceConfiguration, UniqueConfiguration, configuration, unique, CONFIG, interface, configuration, ngsdict
from dream._version import acknowledgements, header

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


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


class save(configuration):

    def __set__(self, cfg: CONFIG, save: bool) -> None:

        if isinstance(save, str) and save == self.__name__:
            save = True

        if not isinstance(save, bool):
            raise ValueError(f"Pass boolean to activate/deactivate handler")

        if save:
            cfg.data[self.__name__] = self.fset(cfg, save)
        elif not save and self.__name__ in cfg.data:
            del cfg.data[self.__name__]


class DirectoryTree(InterfaceConfiguration, is_interface=True):

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


class SingleTree(DirectoryTree):

    name: str = 'single'


class BenchmarkTree(DirectoryTree):

    name: str = 'benchmark'


class Handler(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @configuration(default=None)
    def filename(self, filename: str):
        if filename is None:
            filename = self.name
        return filename

    def save(self, t: float | None = None):
        raise NotImplementedError()


class VTKHandler(Handler):

    name: str = "vtk"

    @configuration(default=2)
    def subdivision(self, subdivision: int):
        return subdivision

    @configuration(default=None)
    def region(self, region):

        if region is None:
            return region

        dofs = ngs.BitArray(self.cfg.mesh.ne)
        dofs.Clear()
        for el in self.cfg.mesh.Materials(region).Elements():
            dofs[el.nr] = True

        return dofs

    @configuration(default=None)
    def fields(self, fields: dict[str, ngs.CF]):
        if fields is None:
            return {}

        path = self.path.joinpath(self.filename)

        self.handler = ngs.VTKOutput(ma=self.cfg.mesh, coefs=list(fields.values()), names=list(
            fields.keys()), filename=str(path), subdivision=self.subdivision)

        return fields

    @property
    def path(self) -> Path:
        path = self.cfg.io.tree.vtk
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self,  t: float | None = -1, it: int | None = None) -> Path:
        return self.handler.Do(t, drawelems=self.region)


class MeshHandler(Handler):

    name: str = "mesh"

    @property
    def path(self) -> Path:
        path = self.cfg.io.tree.root
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self,  t: float | None = None, it: int | None = None) -> None:
        path = self.path.joinpath(self.filename + ".pickle")

        with path.open("wb") as open:
            pickle.dump(self.cfg.mesh, open, protocol=pickle.DEFAULT_PROTOCOL)


class ConfigurationHandler(Handler):

    name: str = "configuration"

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
        path = self.cfg.io.tree.configuration
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, t: float | None = None, it: int | None = None):
        if self.pickle:
            self.save_pickle()
        if self.txt:
            self.save_txt()
        if self.yaml:
            self.save_yaml()

    def save_pickle(self):
        path = self.cfg.io.tree.configuration.joinpath(self.filename + ".pickle")
        with path.open("wb") as open:
            pickle.dump(self.cfg.to_tree(), open, protocol=pickle.DEFAULT_PROTOCOL)

    def save_txt(self):
        path = self.path.joinpath(self.filename + ".txt")

        with path.open("w") as open:
            txt = [acknowledgements(), ""]
            if hasattr(self.cfg, 'doc'):
                txt += [header("Documentation"), self.cfg.doc, ""]
            txt += [header("Configuration"), repr(self.cfg), ""]

            open.write("\n".join(txt))

    def save_yaml(self):

        try:
            import yaml
        except ImportError:
            raise ImportError("Install pyyaml to save configuration as yaml")

        path = self.cfg.io.tree.configuration.joinpath(self.filename + ".yaml")
        with path.open("w") as open:
            yaml.dump(self.cfg.to_tree(), open, sort_keys=False)


class StateHandler(Handler):

    name: str = "state"

    @configuration(default=1)
    def save_at_ith_step(self, i: int):
        return int(i)

    @configuration(default=0)
    def save_time_level_at_ith_step(self, i: int):
        return int(i)

    @property
    def path(self) -> Path:
        path = self.cfg.io.tree.states
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, t: float | None = None, it: int | None = None):

        filename = self.filename
        if t is not None:
            filename = f"{filename}_{t}"

            if it & self.save_time_level_at_ith_step == 0 and bool(self.save_time_level_at_ith_step):
                self.save_transient_gridfunctions(filename)

        if it % self.save_at_ith_step == 0 and bool(self.save_at_ith_step):
            self.save_gridfunction(self.cfg.pde.gfu, filename)

    def save_gridfunction(self, gfu: ngs.GridFunction, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        file = self.path.joinpath(filename + ".ngs")

        gfu.Save(str(file))

    def save_transient_gridfunctions(self, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        for fes, gfus in self.cfg.pde.transient_gfus.items():

            for level, gfu in gfus.items():

                self.save_gridfunction(gfu, f"{filename}_{fes}_{level}")

    save_at_ith_step: int
    save_time_level_at_ith_step: int


class Saver(UniqueConfiguration):

    cfg: SolverConfiguration

    @save(default=False)
    def mesh(self, save: bool):
        return MeshHandler(cfg=self.cfg)

    @save(default=False)
    def state(self, save: bool):
        return StateHandler(cfg=self.cfg)

    @save(default=False)
    def configuration(self, save: bool):
        return ConfigurationHandler(cfg=self.cfg)

    @save(default=False)
    def vtk(self, save: bool):
        return VTKHandler(cfg=self.cfg)

    def pre_solution_routine_saving(self):
        for handler in self.values():
            if isinstance(handler, (MeshHandler, ConfigurationHandler)):
                handler.save()

    def solution_routine_saving(self, t: float | None = None, it: int | None = None):
        for handler in self.values():
            if isinstance(handler, (VTKHandler, StateHandler)):
                handler.save(t, it)

    mesh: MeshHandler
    state: StateHandler
    configuration: ConfigurationHandler
    vtk: VTKHandler


class InputOutputConfiguration(UniqueConfiguration):

    name: str = "io"

    @interface(default=SingleTree)
    def tree(self, tree: DirectoryTree):
        return tree

    @unique(default=Saver)
    def saver(self, saver: Saver):
        return saver

    tree: SingleTree | BenchmarkTree
    saver: Saver
