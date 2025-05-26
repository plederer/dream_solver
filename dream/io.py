from __future__ import annotations
import typing
import logging
import pickle
import numpy as np
from pathlib import Path

import ngsolve as ngs
import dream.bla as bla
from dream.config import dream_configuration, Configuration, ngsdict, is_notebook
from dream.mesh import get_pattern_from_sequence, get_regions_from_pattern
from dream._version import acknowledgements, header

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)


def get_directory_paths(path: Path, pattern: str = "") -> tuple[Path, ...]:
    return tuple(dir for dir in path.glob(pattern + "*") if dir.is_dir())


def get_directory_names(path: Path, pattern: str = "") -> tuple[str, ...]:
    return tuple(dir.name for dir in path.glob(pattern + "*") if dir.is_dir())


class Stream(Configuration, is_interface=True):

    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {"filename": None, "enable": False, "path": Path.cwd()}
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, filename: str | None):
        if filename is None:
            filename = self.name
        self._filename = filename

    @dream_configuration
    def enable(self) -> bool:
        return self._enable

    @enable.setter
    def enable(self, enable: bool):
        self._enable = bool(enable)

    @dream_configuration
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        if not path.is_absolute():
            path = self.root.io.path.joinpath(path)
        self._path = path

    def open(self) -> Stream:

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        return self

    def save_pre_time_routine(self) -> None:
        raise NotImplementedError("Method 'save_pre_routine' is not implemented!")

    def save_in_time_routine(self, t: float, it: int) -> None:
        raise NotImplementedError("Method 'save_routine' is not implemented!")

    def save_post_time_routine(self, t: float | None = None, it: int = 0) -> None:
        raise NotImplementedError("Method 'save_post_routine' is not implemented!")

    def close(self) -> None:
        return None


class MeshStream(Stream):

    name: str = "mesh"

    def save_pre_time_routine(self, t: float | None = None) -> None:

        path = self.path.joinpath(self.filename + ".pickle")

        with path.open("wb") as open:
            pickle.dump(self.mesh, open, protocol=pickle.DEFAULT_PROTOCOL)

    def load_routine(self) -> ngs.Mesh:
        file = self.path.joinpath(self.filename + '.pickle')
        with file.open("rb") as openfile:
            return pickle.load(openfile)


class SettingsStream(Stream):

    name: str = "settings"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "to_pickle": True,
            "to_txt": True,
            "to_yaml": False,
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def to_pickle(self) -> bool:
        return self._to_pickle

    @to_pickle.setter
    def to_pickle(self, pickle: bool):
        self._to_pickle = bool(pickle)

    @dream_configuration
    def to_txt(self) -> bool:
        return self._to_txt

    @to_txt.setter
    def to_txt(self, txt: bool):
        self._to_txt = bool(txt)

    @dream_configuration
    def to_yaml(self) -> bool:
        return self._to_yaml

    @to_yaml.setter
    def to_yaml(self, yaml: bool):
        self._to_yaml = bool(yaml)

    def save_pre_time_routine(self, t: float | None = None):

        if self.to_pickle:
            self.save_to_pickle()
        if self.to_txt:
            self.save_to_txt()
        if self.to_yaml:
            self.save_to_yaml()

    def save_to_pickle(self, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        path = self.path.joinpath(filename + ".pickle")

        with path.open("wb") as open:
            pickle.dump(self.root.to_dict(), open, protocol=pickle.DEFAULT_PROTOCOL)

    def save_to_txt(self, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        path = self.path.joinpath(filename + ".txt")

        with path.open("w") as open:
            txt = [acknowledgements(), ""]
            if hasattr(self.root, 'doc'):
                txt += [header("Documentation"), self.root.doc, ""]
            txt += [header("Configuration"), repr(self.root), ""]

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
            yaml.dump(self.root.to_dict(), open, sort_keys=False)

    def load_from_pickle(self, filename: str | None = None) -> dict:

        if filename is None:
            filename = self.filename

        path = self.path.joinpath(filename + ".pickle")

        with path.open("rb") as open:
            return pickle.load(open)


class VTKStream(Stream):

    name: str = "vtk"

    def __init__(self, mesh, root=None, **default):
        DEFAULT = {
            "rate": 1,
            "subdivision": 2,
            "region": None,
            "fields": {},
        }
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def rate(self) -> int:
        return self._rate

    @rate.setter
    def rate(self, rate: int):
        rate = int(rate)
        if rate <= 0:
            raise ValueError("Saving rate must be greater than 0, otherwise consider deactivating the vtk writer!")
        self._rate = rate

    @dream_configuration
    def subdivision(self) -> int:
        return self._subdivision

    @subdivision.setter
    def subdivision(self, subdivision: int):
        subdivision = int(subdivision)
        if subdivision <= 0:
            raise ValueError("Subdivision must be greater than 0!")
        self._subdivision = subdivision

    @dream_configuration
    def region(self) -> str:
        return self._region

    @region.setter
    def region(self, region: str):
        if region is not None and not isinstance(region, str):
            raise ValueError("Region must be a string!")
        self._region = region

    @dream_configuration
    def fields(self) -> dict[str, ngs.CF]:
        return self._fields

    @fields.setter
    def fields(self, fields: dict[str, ngs.CF]):
        self._fields = ngsdict(**fields)

    def open(self) -> VTKStream:

        if not self.fields:
            logger.warning("No fields to save! VTK stream is deactivated!")
            return None

        super().open()

        self.drawelems = None
        if self.region is not None:
            self.drawelems = ngs.BitArray(self.mesh.ne)
            self.drawelems.Clear()
            for el in self.root.mesh.Materials(self.region).Elements():
                self.drawelems[el.nr] = True

        path = self.path.joinpath(self.filename)
        self.writer = ngs.VTKOutput(ma=self.root.mesh, coefs=list(self.fields.values()), names=list(
            self.fields.keys()), filename=str(path), subdivision=self.subdivision)

        return self

    def save_pre_time_routine(self, t: float | None = None) -> None:
        if t is None:
            t = -1
        self.writer.Do(t, drawelems=self.drawelems)

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


class GridfunctionStream(Stream):

    name = "gfu"

    def __init__(self, mesh, root=None, **default):
        DEFAULT = {
            "rate": 1,
            "time_level_rate": 100,
        }
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def rate(self) -> int:
        return self._rate

    @rate.setter
    def rate(self, rate: int):
        rate = int(rate)
        if rate <= 0:
            raise ValueError("Saving rate must be greater than 0, otherwise consider deactivating the gfu writer!")
        self._rate = rate

    @dream_configuration
    def time_level_rate(self) -> int:
        return self._time_level_rate

    @time_level_rate.setter
    def time_level_rate(self, time_level_rate: int):
        time_level_rate = int(time_level_rate)
        if time_level_rate <= 0:
            raise ValueError("Saving rate must be greater than 0, otherwise consider deactivating the gfu writer!")
        self._time_level_rate = time_level_rate

    def load_gridfunction(self, gfu: ngs.GridFunction, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        file = self.path.joinpath(filename + ".ngs")

        gfu.Load(str(file))

    def load_routine(self, t: float | None = None):

        filename = self.filename
        if t is not None:
            filename = f"{filename}_{t}"

        self.load_gridfunction(self.root.fem.gfu, filename)

    def load_transient_routine(self, sleep: float = 0) -> typing.Generator[float, None, None]:

        if self.root.time.is_stationary:
            raise ValueError("Transient routine is not available in stationary mode!")

        import time

        logger.info(f"Loading gridfunction from '{self.filename}'")
        for t in self.root.time.timer.start(stride=self.rate):
            self.load_routine(t)

            self.root.io.redraw()
            logger.info(f"file: {self.filename} | t: {t}")

            yield t

            time.sleep(sleep)

    def load_time_levels(self, t: float) -> None:

        if self.root.time.is_stationary:
            raise ValueError("Load time levels is not available in stationary mode!")

        for fes, gfu in self.root.fem.scheme.gfus.items():

            for level, gfu in gfu.items():

                self.load_gridfunction(gfu, f"{self.filename}_{t}_{fes}_{level}")

    def save_in_time_routine(self, t: float | None = None, it: int = 0):

        filename = self.filename
        if t is not None:
            filename = f"{filename}_{t}"

            if it % self.time_level_rate == 0:

                for fes, gfus in self.root.fem.scheme.gfus.items():
                    for level, gfu in gfus.items():
                        self.save_gridfunction(gfu, f"{filename}_{fes}_{level}")

        if it % self.rate == 0:
            self.save_gridfunction(self.root.fem.gfu, filename)

    def save_gridfunction(self, gfu: ngs.GridFunction, filename: str | None = None) -> None:

        if filename is None:
            filename = self.filename

        file = self.path.joinpath(filename + ".ngs")

        gfu.Save(str(file))

    def save_pre_time_routine(self, t: float | None = None) -> None:
        self.save_in_time_routine(t, it=0)

    def save_post_time_routine(self, t: float | None = None, it: int = 0) -> None:
        if t is None or not it % self.rate == 0:
            self.save_in_time_routine(t, it=0)


class LogStream(Stream):

    name: str = "dream"

    def __init__(self, mesh, root=None, **default):

        self.logger = logging.getLogger('dream')

        DEFAULT = {
            "level": logging.INFO
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def level(self) -> int:
        return self.logger.level

    @level.setter
    def level(self, level: int | str):
        self.logger.setLevel(level)

    @Stream.enable.setter
    def enable(self, enable: bool):
        enable = bool(enable)

        if enable:
            self.to_terminal = True
            self.to_file = False
        else:
            self.to_terminal = False
            self.to_file = False

        self._enable = bool(enable)

    @dream_configuration
    def to_terminal(self) -> bool:
        return self._to_terminal

    @to_terminal.setter
    def to_terminal(self, to_terminal: bool):

        handler = self._get_handler("terminal")

        if to_terminal and not handler:
            handler = logging.StreamHandler()
            handler.set_name("terminal")
            handler.setFormatter(self._get_formatter())
            self.logger.addHandler(handler)

        elif to_terminal and handler:
            handler.setFormatter(self._get_formatter())

        elif not to_terminal and handler:
            self.logger.removeHandler(handler)

        self._to_terminal = to_terminal

    @dream_configuration
    def to_file(self) -> bool:
        return self._to_file

    @to_file.setter
    def to_file(self, to_file: bool):

        handler = self._get_handler("file")

        if to_file and not handler:

            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)

            handler = logging.FileHandler(filename=self.path.joinpath(self.filename + ".log"))
            handler.set_name("file")
            handler.setFormatter(self._get_formatter())
            self.logger.addHandler(handler)

        elif to_file and handler:

            if not self.path.exists():
                self.path.mkdir(parents=True, exist_ok=True)

            handler.baseFilename = self.path.joinpath(self.filename + ".log")
            handler.setFormatter(self._get_formatter())

        elif not to_file and handler:
            self.logger.removeHandler(handler)

        self._to_file = to_file

    def open(self):

        # Set logger to carriage return
        stream_handler = self._get_handler("terminal")
        if stream_handler:
            stream_handler.terminator = "\r"

        return self

    def close(self):

        # Unset logger to carriage return
        stream_handler = self._get_handler("terminal")
        if stream_handler:
            stream_handler.terminator = "\n"

    def _get_handler(self, name: str) -> logging.Handler:
        for handler in self.logger.handlers:
            if handler.get_name() == name:
                return handler
        return False

    def _get_formatter(self) -> logging.Formatter:

        if self.level == logging.DEBUG:
            return logging.Formatter(
                "%(name)-15s (%(levelname)8s) | %(message)s (%(filename)s:%(lineno)s)", "%Y-%m-%d %H:%M:%S")
        else:
            return logging.Formatter("%(name)-15s (%(levelname)s) | %(message)s")


class SensorStream(Stream):

    name: str = "sensor"

    def __init__(self, mesh, root=None, **default):

        self._sensors: list[Sensor] = []

        DEFAULT = {
            "mode": "w",
            "to_csv": True,
        }
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        if mode not in ('w', 'a'):
            raise ValueError("Mode must be 'w' or 'a'!")
        self._mode = mode

    @dream_configuration
    def to_csv(self) -> bool:
        return self._to_csv

    @to_csv.setter
    def to_csv(self, to_csv: bool):
        self._to_csv = bool(to_csv)

    def add(self, sensor: Sensor) -> None:
        if not isinstance(sensor, Sensor):
            raise ValueError("Sensor must be of type 'Sensor'!")

        sensor.mesh = self.mesh

        self._sensors.append(sensor)

    def open(self) -> SensorStream:

        if not self._sensors:
            logger.warning("No sensors to save! Sensor stream is deactivated!")
            return None

        super().open()

        if self.to_csv:
            self.open_csv_writers()

        return self

    def open_csv_writers(self):
        import csv

        self.csv = {}
        for sensor in self._sensors:
            file = self.path.joinpath(sensor.name + ".csv").open(self.mode)
            writer = csv.writer(file)

            if self.mode == "w":
                names, header = self.get_header(sensor)

                for i in range(len(names)):
                    writer.writerow([names[i]] + [header_[i] for header_ in header])

            self.csv[sensor.name] = (file, writer)

    def get_header(self, sensor: Sensor) -> dict[str, list[str]]:
        names, header = sensor.get_header()

        if not self.root.time.is_stationary:
            names.append('t')
            header = [(*header_, '') for header_ in header]

        return names, header

    def save_pre_time_routine(self, t: float | None = None) -> None:

        if t is None:
            t = ''

        for sensor, data in self.measure():

            if self.to_csv:
                self.csv[sensor.name][1].writerow([t, *data])

    def save_in_time_routine(self,  t: float, it: int) -> None:

        for sensor, data in self.measure():
            if it % sensor.rate == 0:
                if self.to_csv:
                    self.csv[sensor.name][1].writerow([t, *data])

    def save_post_time_routine(self, t: float | None = None, it: int = 0) -> None:

        if t is None:
            t = ''

        for sensor, data in self.measure():
            if self.to_csv:
                if t == '' or not it % sensor.rate == 0:
                    self.csv[sensor.name][1].writerow([t, *data])

    def load_csv_as_dict(self, sensor: Sensor | str) -> tuple[np.ndarray, dict[tuple[str, str, str], np.ndarray]]:
        import csv

        if isinstance(sensor, Sensor):
            sensor = sensor.name

        names = []
        data = []
        with self.path.joinpath(sensor + ".csv").open('r') as file:
            reader = csv.reader(file)

            for row in reader:
                names.append(row[0])
                data.append(row[1:])

        header = {(loc, field, component): None for
                  loc, field, component
                  in zip(data.pop(0), data.pop(0), data.pop(0))}
        names = names[3:]

        if names[0] == "t":
            data.pop(0)
            names.pop(0)

        data = np.array(data, dtype=float)
        index = np.array(names)

        for idx, key in enumerate(header):
            header[key] = data[:, idx]

        return index, header

    def load_csv_as_dataframe(
            self, sensor: Sensor | str, header: tuple = [0, 1, 2, 3],
            index_col: int = [0],
            **pd_kwargs):
        import pandas as pd

        if isinstance(sensor, Sensor):
            sensor = sensor.name

        return pd.read_csv(self.path.joinpath(sensor + ".csv"), header=header, index_col=index_col, **pd_kwargs)

    def measure(self) -> typing.Generator[tuple[Sensor, np.ndarray], None, None]:
        for sensor in self._sensors:
            yield sensor, np.concatenate([data for data in sensor.measure()])

    def close(self):
        if hasattr(self, 'csv'):
            for file, _ in self.csv.values():
                file.close()
            del self.csv


class Sensor:

    def __init__(self, fields: ngsdict,
                 mesh: ngs.Mesh,
                 name: str,
                 rate: int = 1):

        self.fields = ngsdict(**fields)
        self.mesh = mesh
        self.name = name
        self.rate = rate

        # Create single compound field for concurrent evaluation
        self._field = ngs.CF(tuple(self.fields.values()))

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        raise NotImplementedError("Method 'evaluate' is not implemented!")

    def get_header(self):
        names = ['field', 'component']
        header = [(field, component) for field, cf in self.fields.items()
                  for component in self.get_field_components(cf)]
        return names, header

    def get_field_components(self, cf: ngs.CF):
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


class RegionSensor(Sensor):

    def __init__(self, fields, mesh, regions: dict[str, ngs.Region | None], name="region", rate=1, integration_order=5):
        self.regions = regions
        self.integration_order = integration_order
        super().__init__(fields, mesh, name, rate)

    def get_header(self):
        names = ['region']
        names_, header_ = super().get_header()

        names.extend(names_)
        header = [(region, *header) for region in self.regions for header in header_]
        return names, header

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        for region in self.regions.values():
            yield np.array(ngs.Integrate(self._field, self.mesh, order=self.integration_order, definedon=region), ndmin=1)

    def measure_as_dict(self) -> dict[tuple[float, float, float], np.ndarray]:
        return {region: value for region, value in zip(self.regions, self.measure())}

    def measure_as_dataframe(self):
        import pandas as pd

        names, columns = self.get_header()
        columns = pd.MultiIndex.from_tuples(columns, names=names)
        values = np.array([value for value in self.measure()])

        return pd.DataFrame(values, columns=columns)

    def parse_region(self, region: str | None, expected: tuple[str],
                     REGION: typing.Callable[[str],
                                             ngs.Region]) -> dict[str, ngs.Region | None]:

        if region is None:
            return {get_pattern_from_sequence(expected): None}

        elif isinstance(region, str):
            regions = get_regions_from_pattern(expected, region)

            for miss in set(region.split('|')).difference(regions):
                logger.warning(f"Region '{miss}' does not exist! Region {miss} omitted!")

            regions = get_pattern_from_sequence(regions)
            return {regions: REGION(regions)}

        elif isinstance(region, typing.Sequence):
            regions = tuple(region_ for region_ in region if region_ in expected)

            for miss in set(region).difference(regions):
                logger.warning(f"Domain '{miss}' does not exist! Domain {miss} omitted!")

            return {region_: REGION(region_) for region_ in regions}

        else:
            raise ValueError("Domains must be None, a string or a sequence of strings!")


class DomainSensor(RegionSensor):

    def __init__(
            self, fields: dict[str, ngs.CF],
            mesh, domain: str = None, name="domain", rate=1, integration_order=5):
        domain = self.parse_region(domain, mesh.GetMaterials(), mesh.Materials)
        super().__init__(fields, mesh, domain, name, rate, integration_order)


class DomainL2Sensor(DomainSensor):

    def __init__(
            self, fields: dict[str, ngs.CF],
            mesh, domain: str = None, name="domainL2", rate=1, integration_order=5):
        if not isinstance(fields, ngsdict):
            fields = ngsdict(**fields)
        fields = {field: ngs.InnerProduct(cf, cf) for field, cf in fields.items()}
        super().__init__(fields, mesh, domain, name, rate, integration_order)

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        for array in super().measure():
            yield np.sqrt(array)

    def get_field_components(self, cf: ngs.CF):
        return ('L2',)


class BoundarySensor(RegionSensor):

    def __init__(self, fields: dict[str, ngs.CF], mesh, boundary: str, name="boundary", rate=1, integration_order=5):
        boundary = self.parse_region(boundary, mesh.GetBoundaries(), mesh.Boundaries)
        super().__init__(fields, mesh, boundary, name, rate, integration_order)
        self._field = ngs.BoundaryFromVolumeCF(self._field)


class BoundaryL2Sensor(BoundarySensor):

    def __init__(self, fields: dict[str, ngs.CF], mesh, boundary: str, name="boundaryL2", rate=1, integration_order=5):
        if not isinstance(fields, ngsdict):
            fields = ngsdict(**fields)
        fields = {field: ngs.InnerProduct(cf, cf) for field, cf in fields.items()}
        super().__init__(fields, mesh, boundary, name, rate, integration_order)

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        for array in super().measure():
            yield np.sqrt(array)

    def get_field_components(self, cf: ngs.CF):
        return ('L2',)


class PointSensor(Sensor):

    @classmethod
    def from_boundary(cls, fields: ngsdict, mesh: ngs.Mesh, boundary: str, **init):
        return cls(
            fields, mesh, tuple(
                set(mesh[v].point for el in mesh.Boundaries(boundary).Elements() for v in el.vertices)),
            **init)

    def __init__(self, fields, mesh, points, name: str = "point", rate=1):
        self.points = points
        super().__init__(fields, mesh, name, rate)

    def measure(self) -> typing.Generator[np.ndarray, None, None]:
        for point in self.points:
            yield np.array(self._field(self.mesh(*point)), ndmin=1)

    def measure_as_dict(self) -> dict[tuple[float, float, float], np.ndarray]:
        return {point: value for point, value in zip(self.points, self.measure())}

    def measure_as_dataframe(self):
        import pandas as pd

        names, columns = super().get_header()
        columns = pd.MultiIndex.from_tuples(columns, names=names)
        values = np.array([value for value in self.measure()])
        index = pd.Index(list(self.points), name=('x', 'y', 'z')[:self.mesh.dim])

        return pd.DataFrame(values, index=index, columns=columns)

    def get_header(self):
        names = ['point']
        names_, header_ = super().get_header()

        names.extend(names_)
        header = [(point, *header) for point in self.points for header in header_]
        return names, header


class IOConfiguration(Configuration):

    name = "io"

    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):

        path = Path.cwd().joinpath("output")
        DEFAULT = {
            "path": path,
            "log": LogStream(mesh, root=root, path=path, enable=True),
            "ngsmesh": MeshStream(mesh, root=root, path=path),
            "gfu": GridfunctionStream(mesh, root=root, path=path.joinpath("states")),
            "settings": SettingsStream(mesh, root=root, path=path.joinpath("settings")),
            "vtk": VTKStream(mesh, root=root, path=path.joinpath("vtk")),
            "sensor": SensorStream(mesh, root=root, path=path.joinpath("sensors")),
        }
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)

        if not path.is_absolute():
            path = Path.cwd().joinpath(path)

        # Change paths of all streams if they are relative to the current path
        streams = [stream for stream in vars(self).values() if isinstance(stream, Stream)]
        for stream in streams:
            if stream.path.is_relative_to(self._path):
                stream.path = path.joinpath(stream.path.relative_to(self._path))

        self._path = path

    @dream_configuration
    def log(self) -> LogStream:
        return self._log

    @log.setter
    def log(self, log: LogStream | bool):
        self._log = self._parse_stream(log, LogStream)

    @dream_configuration
    def ngsmesh(self) -> MeshStream:
        return self._ngsmesh

    @ngsmesh.setter
    def ngsmesh(self, ngsmesh: MeshStream | bool):
        self._ngsmesh = self._parse_stream(ngsmesh, MeshStream)

    @dream_configuration
    def gfu(self) -> GridfunctionStream:
        return self._gfu

    @gfu.setter
    def gfu(self, gfu: GridfunctionStream | bool):
        self._gfu = self._parse_stream(gfu, GridfunctionStream)

    @dream_configuration
    def settings(self) -> SettingsStream:
        return self._settings

    @settings.setter
    def settings(self, settings: SettingsStream | bool):
        self._settings = self._parse_stream(settings, SettingsStream)

    @dream_configuration
    def vtk(self) -> VTKStream:
        return self._vtk

    @vtk.setter
    def vtk(self, vtk: VTKStream | bool):
        self._vtk = self._parse_stream(vtk, VTKStream)

    @dream_configuration
    def sensor(self) -> SensorStream:
        return self._sensor

    @sensor.setter
    def sensor(self, sensor: SensorStream | bool):
        self._sensor = self._parse_stream(sensor, SensorStream)

    def draw(self, fields: ngsdict, **kwargs):
        if is_notebook():
            from ngsolve.webgui import Draw
        else:
            from ngsolve import Draw

        self.scenes = [Draw(draw, self.mesh, name, **kwargs) for name, draw in fields.items()]

    def redraw(self, blocking: bool = False):

        if hasattr(self, "scenes"):

            for scene in self.scenes:
                if scene is not None:
                    scene.Redraw()
            ngs.Redraw(blocking)

    def open(self):

        SINGLE = (MeshStream, SettingsStream)
        FILE = (VTKStream, GridfunctionStream, SensorStream)

        streams = [stream.open() for stream in vars(self).values() if isinstance(stream, Stream) and stream.enable]

        self.single_streams = [stream for stream in streams if isinstance(stream, SINGLE)]
        self.file_streams = [stream for stream in streams if isinstance(stream, FILE)]

    def save_pre_time_routine(self, t: float | None = None):
        for stream in self.single_streams:
            stream.save_pre_time_routine(t)

        for stream in self.file_streams:
            stream.save_pre_time_routine(t)

    def save_in_time_routine(self, t: float, it: int):
        for stream in self.file_streams:
            stream.save_in_time_routine(t, it)

    def save_post_time_routine(self, t: float | None = None, it: int = 0):
        for stream in self.file_streams:
            stream.save_post_time_routine(t, it)

    def close(self):

        self.log.close()

        for stream in self.file_streams:
            stream.close()

    def _parse_stream(self, stream: Stream, type: Stream):
        if not isinstance(stream, type):
            raise ValueError(f"Stream must be of type '{type}'!")
        stream.mesh = self.mesh
        stream.root = self.root
        return stream

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
