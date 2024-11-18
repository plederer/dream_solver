from __future__ import annotations
import typing
import logging
import pickle
import numpy as np
from pathlib import Path

import ngsolve as ngs
import dream.bla as bla
from dream.config import InterfaceConfiguration, UniqueConfiguration, configuration, unique, CONFIG, interface, configuration, ngsdict
from dream.mesh import get_pattern_from_sequence, get_regions_from_pattern
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


class sensor(configuration):
    def __set__(self, cfg: SensorStream, name: str) -> None:

        if self.__name__ not in cfg.data:
            cfg.data[self.__name__] = {}

        sensor = self.fset(cfg, name)
        if isinstance(sensor, Sensor):
            cfg.data[self.__name__][sensor.sensor_name] = sensor


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
    def settings(self, path):
        return path

    @path(default='sensors')
    def sensors(self, path):
        return path

    def get_directory_paths(self, pattern: str = "") -> tuple[Path, ...]:
        return tuple(dir for dir in self.root.glob(pattern + "*") if dir.is_dir())

    def get_directory_names(self, pattern: str = "") -> tuple[str, ...]:
        return tuple(dir.name for dir in self.root.glob(pattern + "*") if dir.is_dir())

    root: Path
    states: Path
    vtk: Path
    settings: Path
    sensors: Path


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

    def open(self) -> Stream:
        return self

    def save_pre_time_routine(self) -> None:
        raise NotImplementedError("Method 'save_pre_routine' is not implemented!")

    def save_in_time_routine(self, t: float, it: int) -> None:
        raise NotImplementedError("Method 'save_routine' is not implemented!")

    def save_post_time_routine(self, t: float | None = None, it: int = 0) -> None:
        raise NotImplementedError("Method 'save_post_routine' is not implemented!")

    def close(self):
        return None


class MeshStream(Stream):

    name: str = "mesh"

    @property
    def path(self) -> Path:
        path = self.cfg.io.folders.root
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save_pre_time_routine(self) -> None:
        path = self.path.joinpath(self.filename + ".pickle")

        with path.open("wb") as open:
            pickle.dump(self.mesh, open, protocol=pickle.DEFAULT_PROTOCOL)

    def load_routine(self) -> ngs.Mesh:
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
        path = self.cfg.io.folders.settings
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save_pre_time_routine(self):
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
        return ngsdict(**fields)

    @property
    def path(self) -> Path:
        path = self.cfg.io.folders.vtk
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def open(self) -> VTKStream:

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

    def save_pre_time_routine(self) -> None:
        if not self.cfg.time.is_stationary:
            self.writer.Do(self.cfg.time.timer.t.Get(), drawelems=self.drawelems)

    def save_in_time_routine(self,  t: float, it: int) -> None:
        if it % self.rate == 0:
            self.writer.Do(t, drawelems=self.drawelems)

    def save_post_time_routine(self, t: float | None = None, it: int = 0) -> None:
        if t is None:
            self.writer.Do(-1, drawelems=self.drawelems)
        elif not it % self.rate == 0:
            self.writer.Do(t, drawelems=self.drawelems)

    def close(self) -> None:
        del self.writer

    rate: int
    subdivision: int
    region: str
    fields: ngsdict


class FieldStream(Stream, is_interface=True):

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

    def save_pre_time_routine(self) -> None:
        if not self.cfg.time.is_stationary:
            self.save_in_time_routine(self.cfg.time.timer.t.Get(), it=0)

    def save_post_time_routine(self, t: float | None = None, it: int = 0) -> None:
        if t is None or not it % self.rate == 0:
            self.save_in_time_routine(t, it=0)

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

    rate: int


class GridfunctionStream(FieldStream):

    name: str = "gfu"

    def save_in_time_routine(self, t: float | None = None, it: int = 0):

        if it % self.rate == 0:

            filename = self.filename
            if t is not None:
                filename = f"{filename}_{t}"

            self.save_gridfunction(self.cfg.pde.gfu, filename)

    def load_routine(self, t: float | None = None):

        filename = self.filename
        if t is not None:
            filename = f"{filename}_{t}"

        self.load_gridfunction(self.cfg.pde.gfu, filename)

    def load_transient_routine(self, sleep: float = 0) -> typing.Generator[float, None, None]:

        if self.cfg.time.is_stationary:
            raise ValueError("Transient routine is not available in stationary mode!")

        import time

        self.cfg.pde.draw()

        logger.info(f"Loading gridfunction from '{self.filename}'")
        for t in self.cfg.time.timer.start(stride=self.rate):
            self.load_routine(t)

            self.cfg.pde.redraw()
            logger.info(f"file: {self.filename} | t: {t}")

            yield t

            time.sleep(sleep)


class TransientGridfunctionStream(FieldStream):

    name: str = "gfu_dt"

    def save_in_time_routine(self, t: float, it: int):

        if it % self.rate == 0:

            for fes, gfus in self.cfg.pde.transient_gfus.items():

                for level, gfu in gfus.items():

                    self.save_gridfunction(gfu, f"{self.filename}_{t}_{fes}_{level}")

    def load_time_levels(self, t: float) -> None:

        for fes, gfu in self.cfg.pde.transient_gfus.items():

            for level, gfu in gfu.items():

                self.load_gridfunction(gfu, f"{self.filename}_{t}_{fes}_{level}")


class LogStream(Stream):

    name: str = "dream"

    def __init__(self, cfg=None, mesh=None, **kwargs):
        self.logger = logging.getLogger('dream')
        super().__init__(cfg=cfg, mesh=mesh, **kwargs)

    @configuration(default=logging.INFO)
    def level(self, level: int):

        if isinstance(level, int):
            level = logging.getLevelName(level)

        self.logger.setLevel(level)

        if level == 'DEBUG':
            self.formatter = logging.Formatter(
                "%(name)-15s (%(levelname)8s) | %(message)s (%(filename)s:%(lineno)s)", "%Y-%m-%d %H:%M:%S")
        else:
            self.formatter = logging.Formatter("%(name)-15s (%(levelname)s) | %(message)s")

        return level

    @configuration(default=True)
    def to_terminal(self, to_terminal: bool):

        if hasattr(self, 'terminal_handler'):
            self.logger.removeHandler(self.terminal_handler)

        if to_terminal:
            self.terminal_handler = logging.StreamHandler()
            self.terminal_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.terminal_handler)

        return to_terminal

    @configuration(default=False)
    def to_file(self, to_file: bool):

        if hasattr(self, 'file_handler'):
            self.logger.removeHandler(self.file_handler)

        if to_file:
            self.file_handler = logging.FileHandler(self.path.joinpath(self.filename + ".log"))
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

        return to_file

    @property
    def path(self) -> Path:
        path = self.cfg.io.folders.root
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def __del__(self):
        if hasattr(self, 'terminal_handler'):
            self.logger.removeHandler(self.terminal_handler)
        if hasattr(self, 'file_handler'):
            self.logger.removeHandler(self.file_handler)


class SensorStream(Stream):

    name: str = "sensor"

    @sensor(default=None)
    def point(self, name: str):
        if isinstance(name, str):
            return PointSensor(sensor_name=name)
        return None

    @sensor(default=None)
    def domain(self, name: str):
        if isinstance(name, str):
            return DomainSensor(sensor_name=name)
        return None

    @sensor(default=None)
    def domain_L2(self, name: str):
        if isinstance(name, str):
            return DomainL2Sensor(sensor_name=name)
        return None

    @sensor(default=None)
    def boundary(self, name: str):
        if isinstance(name, str):
            return BoundarySensor(sensor_name=name)
        return None

    @sensor(default=None)
    def boundary_L2(self, name: str):
        if isinstance(name, str):
            return BoundaryL2Sensor(sensor_name=name)
        return None
    
    @configuration(default='w')
    def mode(self, mode: str):
        return mode

    @configuration(default=True)
    def to_csv(self, activate: bool):
        return activate

    @property
    def path(self) -> Path:
        path = self.cfg.io.folders.sensors
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def open(self) -> SensorStream:
        self.sensors: tuple[Sensor] = tuple(sensor_ for config in self.get_configurations(sensor)
                                            for sensor_ in self[config.__name__].values())

        for instance in self.sensors:
            instance.cfg = self.cfg
            instance.mesh = self.mesh
            instance.open()

        if self.to_csv:
            self.open_csv_writers()

        return self

    def open_csv_writers(self):
        import csv

        self.csv = {}
        for sensor in self.sensors:
            file = self.path.joinpath(sensor.sensor_name + ".csv").open(self.mode)
            writer = csv.writer(file)

            if self.mode == "w":
                for name, value in self.get_header(sensor).items():
                    writer.writerow([name, *value])

            self.csv[sensor.sensor_name] = (file, writer)

    def get_header(self, sensor: Sensor) -> dict[str, list[str]]:
        header = sensor.get_header()

        for value in header.values():
            if len(value) != len(header['field']):
                raise ValueError("Header fields are not consistent!")

        if not self.cfg.time.is_stationary:
            header['t'] = [' ']*len(header['field'])

        return header

    def save_pre_time_routine(self) -> None:
        if not self.cfg.time.is_stationary:

            for sensor, data in self.measure():
                if self.to_csv:
                    self.csv[sensor.sensor_name][1].writerow([self.cfg.time.timer.t.Get(), *data])

    def save_in_time_routine(self,  t: float, it: int) -> None:
        for sensor, data in self.measure():
            if it % sensor.rate == 0:
                if self.to_csv:
                    self.csv[sensor.sensor_name][1].writerow([t, *data])

    def save_post_time_routine(self, t: float | None = None, it: int = 0) -> None:
        for sensor, data in self.measure():
            if t is None:
                if self.to_csv:
                    self.csv[sensor.sensor_name][1].writerow(data)
            elif not it % sensor.rate == 0:
                if self.to_csv:
                    self.csv[sensor.sensor_name][1].writerow([t, *data])

    def load_as_dataframe(self, sensor: Sensor | str, header: tuple = [0, 1, 2], index_col: int = [0], **pd_kwargs):
        import pandas as pd

        if isinstance(sensor, Sensor):
            sensor = sensor.sensor_name

        return pd.read_csv(self.path.joinpath(sensor + ".csv"), header=header, index_col=index_col, **pd_kwargs)

    def measure(self) -> typing.Generator[tuple[Sensor, np.ndarray], None, None]:
        for sensor in self.sensors:
            yield sensor, np.concatenate([measurement for measurement in sensor.measure()])

    def close(self):
        if hasattr(self, 'csv'):
            for file, _ in self.csv.values():
                file.close()
            del self.csv

    point: dict[str, PointSensor]
    domain: dict[str, DomainSensor]
    domain_L2: dict[str, DomainL2Sensor]
    boundary: dict[str, BoundarySensor]
    boundary_L2: dict[str, BoundaryL2Sensor]
    to_csv: bool


class Sensor(UniqueConfiguration):

    @configuration(default=None)
    def sensor_name(self, name: str):
        return name

    @configuration(default=1)
    def rate(self, i: int):
        i = int(i)
        if i <= 0:
            raise ValueError("Saving frequency must be greater than 0, otherwise consider deactivating the vtk writer!")
        return i

    @configuration(default=None)
    def fields(self, fields: dict[str, ngs.CF]):
        if fields is None:
            fields = {}
        return ngsdict(**fields)

    def open(self, fields: dict[str, ngs.CF] | None = None):

        if fields is None:
            fields = self.fields

        if not fields:
            raise ValueError("No fields to evaluate!")

        # Create single compound field for concurrent evaluation
        self._fields = fields
        self._field = ngs.CF(tuple(fields.values()))

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        raise NotImplementedError("Method 'evaluate' is not implemented!")

    def get_header(self):
        header = {'field': [], 'component': []}

        for field, cf in self._fields.items():
            component = self.get_components_name(cf)
            header['field'].extend([field]*cf.dim)
            header['component'].extend(component)

        return header

    def get_components_name(self, cf: bla.SCALAR | bla.VECTOR | bla.MATRIX):
        if bla.is_scalar(cf):
            return bla.SCALAR_COMPONENT
        elif bla.is_vector(cf) and cf.dim <= 3:
            return bla.VECTOR_COMPONENT[:cf.dim]
        elif bla.is_matrix(cf) and cf.dim <= 9:
            if tuple(cf.dims) == (2, 2):
                return (bla.MATRIX_COMPONENT[0], bla.MATRIX_COMPONENT[1], bla.MATRIX_COMPONENT[3], bla.MATRIX_COMPONENT[4])
            return bla.MATRIX_COMPONENT
        else:
            return tuple(range(cf.dim))

    def close(self):
        del self._field

    sensor_name: str
    rate: int
    fields: ngsdict


class RegionSensor(Sensor):

    @configuration(default=5)
    def integration_order(self, order: int):
        return order

    @configuration(default=None)
    def regions(self, regions: tuple[str, ...]):
        return regions

    def get_header(self):
        header = {'region': [region for region in self._regions for _ in range(self._field.dim)]}
        for key, value in super().get_header().items():
            header[key] = value*len(self._regions)
        return header

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        for region in self._regions.values():
            yield np.array(ngs.Integrate(self._field, self.mesh, order=self.integration_order, definedon=region))

    def set_regions(self, expected: tuple[str], fregion: typing.Callable[[str], ngs.Region]) -> None:

        if self.regions is None:
            self._regions = {get_pattern_from_sequence(expected): None}

        elif isinstance(self.regions, str):
            regions = get_regions_from_pattern(expected, self.regions)

            for miss in set(self.regions.split('|')).difference(regions):
                logger.warning(f"Region '{miss}' does not exist! Region {miss} omitted in sensor {self.sensor_name}!")

            self.regions = get_pattern_from_sequence(regions)
            self._regions = {self.regions: fregion(self.regions)}

        elif isinstance(self.regions, typing.Sequence):
            regions = tuple(region for region in self.regions if region in expected)

            for miss in set(self.regions).difference(regions):
                logger.warning(f"Domain '{miss}' does not exist! Domain {miss} omitted in sensor {self.sensor_name}!")

            self.regions = regions
            self._regions = {domain: fregion(domain) for domain in self.regions}
        else:
            raise ValueError("Domains must be None, a string or a sequence of strings!")

    l2_norm: bool
    integration_order: int


class DomainSensor(RegionSensor):

    name: str = "domain"

    def open(self, fields: dict[str, ngs.CF] | None = None):
        super().open(fields)
        self.set_regions(self.mesh.GetMaterials(), self.mesh.Materials)


class DomainL2Sensor(DomainSensor):

    name: str = "domain_L2"

    def open(self, fields: dict[str, ngs.CF] | None = None):
        fields = {field: ngs.InnerProduct(cf, cf) for field, cf in self.fields.items()}
        super().open(fields)

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        for array in super().measure():
            yield np.sqrt(array)

    def get_components_name(self, cf: bla.SCALAR | bla.VECTOR | bla.MATRIX):
        return ('L2',)


class BoundarySensor(RegionSensor):

    name: str = "boundary"

    def open(self, fields: dict[str, ngs.CF] | None = None):
        super().open(fields)

        self._field = ngs.BoundaryFromVolumeCF(self._field)
        self.set_regions(self.mesh.GetBoundaries(), self.mesh.Boundaries)


class BoundaryL2Sensor(BoundarySensor):

    name: str = "boundary_L2"

    def open(self, fields: dict[str, ngs.CF] | None = None):
        fields = {field: ngs.InnerProduct(cf, cf) for field, cf in self.fields.items()}
        self.field_header = tuple((field, 'L2') for field in fields)
        super().open(fields)

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        for array in super().measure():
            yield np.sqrt(array)

    def get_components_name(self, cf: bla.SCALAR | bla.VECTOR | bla.MATRIX):
        return ('L2',)


class PointSensor(Sensor):

    @configuration(default=())
    def points(self, points: tuple[tuple[float, float, float], ...]):
        return points

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        for point in self.points:
            yield np.array(self._field(self.mesh(*point)))

    def get_header(self):
        header = {'point': [point for point in self.points for _ in range(self._field.dim)]}
        for key, value in super().get_header().items():
            header[key] = value*len(self.points)
        return header


class IOConfiguration(UniqueConfiguration):

    cfg: SolverConfiguration

    @stream(default=True)
    def log(self, activate: bool):
        return LogStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def ngsmesh(self, activate: bool):
        return MeshStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def gfu(self, activate: bool):
        return GridfunctionStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def transient_gfu(self, activate: bool):
        if self.cfg.time.is_stationary:
            raise ValueError("TimeStateStream is not available in stationary mode!")
        return TransientGridfunctionStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def settings(self, activate: bool):
        return SettingsStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def vtk(self, activate: bool):
        return VTKStream(cfg=self.cfg, mesh=self.mesh)

    @stream(default=False)
    def sensor(self, activate: bool):
        return SensorStream(cfg=self.cfg, mesh=self.mesh)

    @interface(default=SingleFolders)
    def folders(self, folders: IOFolders):
        return folders

    def open(self):

        self.single_streams = []
        self.file_streams = []

        for stream in self.values():
            if isinstance(stream, (MeshStream, SettingsStream)):
                self.single_streams.append(stream.open())
            elif isinstance(stream, (VTKStream, GridfunctionStream, TransientGridfunctionStream, SensorStream)):
                self.file_streams.append(stream.open())

    def save_pre_time_routine(self):
        for stream in self.single_streams:
            stream.save_pre_time_routine()

        for stream in self.file_streams:
            stream.save_pre_time_routine()

    def save_in_time_routine(self, t: float, it: int):
        for stream in self.file_streams:
            stream.save_in_time_routine(t, it)

    def save_post_time_routine(self, t: float | None = None, it: int = 0):
        for stream in self.file_streams:
            stream.save_post_time_routine(t, it)

    def close(self):
        for stream in self.file_streams:
            stream.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    log: LogStream
    ngsmesh: MeshStream
    gfu: GridfunctionStream
    transient_gfu: TransientGridfunctionStream
    settings: SettingsStream
    vtk: VTKStream
    sensor: SensorStream
    folders: SingleFolders | BenchmarkFolders
