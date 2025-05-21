import pytest
import ngsolve as ngs
import numpy as np
import csv
from dream.io import (
    MeshStream,
    VTKStream,
    GridfunctionStream,
    SettingsStream,
    SensorStream,
    PointSensor,
    DomainSensor,
    DomainL2Sensor,
    BoundarySensor,
    BoundaryL2Sensor
)
from dream.config import dream_configuration
from dream.solver import SolverConfiguration, FiniteElementMethod
from dream.mesh import BoundaryConditions, DomainConditions
from dream.time import StationaryRoutine, TransientRoutine, TimeRoutine, TimeSchemes


class DummyTimeScheme(TimeSchemes):
    time_levels = ("n", "n+1")

    def add_symbolic_temporal_forms(self, blf, lf):
        ...


class DummyFiniteElementMethod(FiniteElementMethod):

    name: str = 'dummy'
    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "scheme": DummyTimeScheme(mesh, root)
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self):
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes):
        OPTIONS = [DummyTimeScheme]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def initialize_time_scheme_gridfunctions(self):
        return super().initialize_time_scheme_gridfunctions('U', 'Uhat')

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):
        spaces['U'] = ngs.L2(self.mesh, order=0)
        spaces['Uhat'] = ngs.L2(self.mesh, order=0)

    def get_temporal_integrators(self):
        return {'U': ngs.dx, 'Uhat': ngs.dx}

    def add_symbolic_spatial_forms(self, blf, lf):
        u, v = self.TnT['U']

        blf['U']['test'] = u * v * ngs.dx
        lf['U']['test'] = v * ngs.dx

    def add_symbolic_temporal_forms(self, blf, lf):
        pass

    def get_solution_fields(self):
        pass

    def set_initial_conditions(self) -> None:
        pass

    def set_boundary_conditions(self) -> None:
        pass


class DummySolverConfiguration(SolverConfiguration):

    name: str = 'dummy'

    def __init__(self, mesh, **default):
        bcs = BoundaryConditions(mesh, [])
        dcs = DomainConditions(mesh, [])

        DEFAULT = {
            "fem": DummyFiniteElementMethod(mesh, self),
            "time": TransientRoutine(mesh, self),
        }
        DEFAULT.update(default)
        super().__init__(mesh, bcs, dcs, **DEFAULT)

    @dream_configuration
    def fem(self):
        return self._fem

    @fem.setter
    def fem(self, fem: FiniteElementMethod):
        self._fem = fem

    @dream_configuration
    def time(self) -> TransientRoutine:
        return self._time

    @time.setter
    def time(self, time: FiniteElementMethod):
        OPTIONS = [TransientRoutine, StationaryRoutine]
        self._time = self._get_configuration_option(time, OPTIONS, TimeRoutine)


mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))


@pytest.fixture
def cfg():
    return DummySolverConfiguration(mesh)


def cleanup_path(path):
    if path.exists():
        for p in path.iterdir():
            p.unlink()
        path.rmdir()


def test_meshstream_initialize(cfg):

    handler = MeshStream(cfg.mesh, cfg, path="mesh_test")
    assert handler.open() is handler

    cleanup_path(handler.path)


def test_meshstream_default_save_and_load(cfg):

    handler = MeshStream(cfg.mesh, cfg, path="mesh_test").open()
    handler.save_pre_time_routine()

    path = handler.path.joinpath("mesh.pickle")
    assert path.exists()

    mesh = handler.load_routine()
    assert isinstance(mesh, ngs.Mesh)

    cleanup_path(handler.path)


def test_meshstream_save_and_load(cfg):

    handler = MeshStream(cfg.mesh, cfg, path="mesh_test").open()
    handler.filename = "test"
    handler.save_pre_time_routine()

    path = handler.path.joinpath("test.pickle")
    assert path.exists()

    mesh = handler.load_routine()
    assert isinstance(mesh, ngs.Mesh)

    cleanup_path(handler.path)


def test_vtkstream_initialize(cfg):

    handler = VTKStream(cfg.mesh, cfg.root, path="vtk_test")
    handler.fields = {'u': ngs.CoefficientFunction(1)}

    assert handler.open() is handler

    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_vtkstream_save_pre_time_routine(cfg):

    handler = VTKStream(cfg.mesh, cfg.root, path="vtk_test")
    handler.fields = {'u': ngs.CoefficientFunction(1)}
    handler = handler.open()

    cfg.time = "stationary"
    handler.save_pre_time_routine()
    path = handler.path.joinpath("vtk.vtu")
    assert path.exists()

    cfg.time = "transient"
    handler.save_pre_time_routine(0.0)
    handler.save_pre_time_routine(0.0)
    path = handler.path.joinpath("vtk.pvd")
    assert path.exists()

    with path.open('r') as openf:
        data = openf.readlines()
    assert data[3:5] == [
        """<DataSet timestep="0" file="vtk.vtu"/>\n""",
        """<DataSet timestep="0" file="vtk_step00001.vtu"/>\n"""
    ]

    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_vtkstream_default_save_in_time_routine(cfg):

    handler = VTKStream(cfg.mesh, cfg.root, path="vtk_test")
    handler.fields = {'u': ngs.CoefficientFunction(1)}
    handler = handler.open()

    for t in range(5):
        handler.save_in_time_routine(t, 0)
    path = handler.path.joinpath("vtk.pvd")
    assert path.exists()

    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_vtkstream_save_in_time_routine(cfg):

    handler = VTKStream(cfg.mesh, cfg.root, path="vtk_test")
    handler.filename = "test"
    handler.fields = {'u': ngs.CoefficientFunction(1)}
    handler = handler.open()

    for t in range(5):
        handler.save_in_time_routine(t, 0)
    path = handler.path.joinpath("test.pvd")
    assert path.exists()

    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_vtkstream_saving_rate_in_time_routine(cfg):

    handler = VTKStream(cfg.mesh, cfg.root, path="vtk_test")
    handler.rate = 2
    handler.fields = {'u': ngs.CoefficientFunction(1)}

    handler = handler.open()
    for it, t in enumerate(range(10)):
        handler.save_in_time_routine(t, it)
    path = handler.path.joinpath("vtk.pvd")
    assert path.exists()

    with path.open('r') as openf:
        data = openf.readlines()
    assert data[3:8] == [
        """<DataSet timestep="0" file="vtk.vtu"/>\n""",
        """<DataSet timestep="2" file="vtk_step00001.vtu"/>\n""",
        """<DataSet timestep="4" file="vtk_step00002.vtu"/>\n""",
        """<DataSet timestep="6" file="vtk_step00003.vtu"/>\n""",
        """<DataSet timestep="8" file="vtk_step00004.vtu"/>\n"""
    ]

    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_vtkstream_save_post_time_routine_stationary(cfg):

    handler = VTKStream(cfg.mesh, cfg.root, path="vtk_test")
    handler.fields = {'u': ngs.CoefficientFunction(1)}
    handler = handler.open()

    handler.save_post_time_routine(t=None)
    path = handler.path.joinpath('vtk.vtu')
    assert path.exists()

    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_vtkstream_save_post_time_routine_transient(cfg):

    handler = VTKStream(cfg.mesh, cfg.root, path="vtk_test")
    handler.fields = {'u': ngs.CoefficientFunction(1)}
    handler = handler.open()

    handler.rate = 3
    handler.save_post_time_routine(t=0.1, it=10)
    path = handler.path.joinpath('vtk.vtu')
    assert path.exists()

    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


@pytest.fixture
def gfu_handler(cfg):
    cfg.time = "transient"
    cfg.time.timer.step = 1
    cfg.initialize()
    cfg.fem.gfu.vec[:] = 1
    handler = GridfunctionStream(cfg.mesh, cfg.root, path="gfu_test")
    yield handler, cfg

    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_gridfunctionstream_initialize(gfu_handler):
    handler, _ = gfu_handler
    assert handler.open() is handler


def test_gridfunctionstream_save_pre_time_routine(gfu_handler):
    handler, cfg = gfu_handler
    handler = handler.open()

    cfg.time = "stationary"
    handler.save_pre_time_routine()
    path = handler.path.joinpath("gfu.ngs")
    assert path.exists()

    cfg.time = "transient"
    cfg.fem.initialize_time_scheme_gridfunctions()
    handler.save_pre_time_routine(0.0)
    path = handler.path.joinpath("gfu_0.0.ngs")
    assert path.exists()


def test_gridfunctionstream_save_in_time_routine(gfu_handler):

    handler, cfg = gfu_handler
    handler.filename = "test"
    handler = handler.open()

    handler.save_in_time_routine(t=0.0)
    path = handler.path.joinpath("test_0.0.ngs")
    assert path.exists()

    gfu = ngs.GridFunction(ngs.L2(cfg.mesh, order=0)**2)
    gfu.Load(str(path))
    np.testing.assert_array_equal(gfu.vec, cfg.fem.gfu.vec)


def test_gridfunctionstream_saving_rate_in_time_routine(gfu_handler):

    handler, cfg = gfu_handler
    handler.rate = 2
    handler = handler.open()
    for it, t in enumerate(range(10)):
        handler.save_in_time_routine(t, it)
    for t in range(0, 10, 2):
        path = handler.path.joinpath(f"gfu_{t}.ngs")
        assert path.exists()


def test_gridfunctionstream_save_post_time_routine_stationary(gfu_handler):
    handler, _ = gfu_handler
    handler = handler.open()
    handler.save_post_time_routine(t=None)
    path = handler.path.joinpath('gfu.ngs')
    assert path.exists()


def test_gridfunctionstream_save_post_time_routine_transient(gfu_handler):
    handler, _ = gfu_handler
    handler.rate = 3
    handler = handler.open()
    handler.save_post_time_routine(t=0.1, it=10)
    path = handler.path.joinpath('gfu_0.1.ngs')
    assert path.exists()


def test_gridfunctionstream_save_and_load_gridfunction(gfu_handler):
    handler, cfg = gfu_handler
    path = handler.path.joinpath("test.ngs")
    gfu = ngs.GridFunction(ngs.L2(cfg.mesh, order=0)**2)
    handler = handler.open()

    handler.save_gridfunction(cfg.fem.gfu, 'test')
    assert path.exists()

    handler.load_gridfunction(gfu, 'test')
    np.testing.assert_array_equal(gfu.vec, cfg.fem.gfu.vec)


def test_gridfunctionstream_load_transient_routine(gfu_handler):
    handler, cfg = gfu_handler
    handler.filename = "test"
    handler = handler.open()
    for t in cfg.time.timer.start(True):
        cfg.fem.gfu.vec[:] = t
        handler.save_in_time_routine(t)
    gfu = ngs.GridFunction(ngs.L2(cfg.mesh, order=0)**2)
    for t in handler.load_transient_routine():
        gfu.vec[:] = t
        np.testing.assert_array_equal(gfu.vec, cfg.fem.gfu.vec)


def test_gridfunctionstream_save_and_load_gridfunction_levels(gfu_handler):
    handler, cfg = gfu_handler
    handler.filename = 'test'
    handler = handler.open()
    for fes in cfg.fem.scheme.gfus:
        for level in cfg.fem.scheme.gfus[fes]:
            gfu = cfg.fem.scheme.gfus[fes][level]
            gfu.vec[:] = 1
    handler.save_in_time_routine(0.0, it=0)
    for fes in cfg.fem.scheme.gfus:
        for level in cfg.fem.scheme.gfus[fes]:
            assert handler.path.joinpath(f"test_0.0_{fes}_{level}.ngs").exists()
    handler.load_time_levels(0.0)
    for fes in cfg.fem.scheme.gfus:
        for level in cfg.fem.scheme.gfus[fes]:
            np.testing.assert_array_equal(
                cfg.fem.scheme.gfus[fes][level].vec, np.ones_like(cfg.fem.scheme.gfus[fes][level].vec)
            )


def test_settingsstream_initialize(cfg):
    handler = SettingsStream(cfg.mesh, cfg, path="settings_test")
    assert handler.open() is handler
    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_settingsstream_save_and_load_to_pickle(cfg):
    handler = SettingsStream(cfg.mesh, cfg, path="settings_test").open()
    path = handler.path.joinpath("test.pickle")
    handler.save_to_pickle('test')
    assert path.exists()
    cfg2 = handler.load_from_pickle('test')
    assert cfg2 == cfg.to_dict()
    path.unlink()
    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


@pytest.fixture
def sensor_handler(cfg):
    cfg.io.sensor.path = "sensor_test"
    handler = SensorStream(cfg.mesh, cfg, path="sensor_test")
    handler.to_csv = True
    yield handler, cfg
    cleanup_path(handler.path)
    cleanup_path(cfg.io.path)


def test_sensorstream_empty_initialize(sensor_handler):
    handler, _ = sensor_handler
    assert handler.open() is None


def test_sensorstream_initialize(sensor_handler):
    handler, _ = sensor_handler
    point = PointSensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, handler.mesh,
                        ((0, 0), (1, 0), (0, 1)), name="asdasa")
    handler.add(point)
    assert handler.open() is handler
    handler.close()


def test_sensorstream_save_transient_point_sensor(sensor_handler):
    handler, cfg = sensor_handler
    point = PointSensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, handler.mesh,
                        ((0, 0), (1, 0), (0, 1)), name="point", rate=5)
    cfg.io.sensor.add(point)
    cfg.time.timer.step = 0.1
    with cfg.io as io:
        io.save_pre_time_routine(0.0)
        for it, t in enumerate(cfg.time.timer()):
            io.save_in_time_routine(t, it)
        io.save_post_time_routine(t, it)
    path = cfg.io.sensor.path.joinpath("point.csv")
    assert path.exists()
    expected = [
        ['point', '(0, 0)', '(0, 0)', '(0, 0)', '(1, 0)', '(1, 0)', '(1, 0)', '(0, 1)', '(0, 1)', '(0, 1)'],
        ['field', 'rho', 'u', 'u', 'rho', 'u', 'u', 'rho', 'u', 'u'],
        ['component', ' ', 'x', 'y', ' ', 'x', 'y', ' ', 'x', 'y'],
        ['t', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ['0.0', '0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '1.0'],
        ['0.1', '0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '1.0'],
        ['0.6', '0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '1.0'],
        ['1.0', '0.0', '0.0', '0.0', '1.0', '1.0', '0.0', '0.0', '0.0', '1.0']
    ]
    with path.open('r') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            assert line == expected[i]


def test_sensorstream_save_transient_domain_sensor(sensor_handler):
    handler, cfg = sensor_handler
    cfg.time = "transient"
    cfg.time.timer.step = 0.1
    domain = DomainSensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, handler.mesh, 'default', name="domain", rate=5)
    cfg.io.sensor.add(domain)
    with cfg.io as io:
        io.save_pre_time_routine(0.0)
        for it, t in enumerate(cfg.time.timer()):
            io.save_in_time_routine(t, it)
        io.save_post_time_routine(t, it)
    path = handler.path.joinpath("domain.csv")
    assert path.exists()
    expected = [
        ['region', 'default', 'default', 'default'],
        ['field', 'rho', 'u', 'u'],
        ['component', ' ', 'x', 'y'],
        ['t', ' ', ' ', ' '],
        np.array([0.0, 1/2, 1/2, 1/2]),
        np.array([0.1, 1/2, 1/2, 1/2]),
        np.array([0.6, 1/2, 1/2, 1/2]),
        np.array([1.0, 1/2, 1/2, 1/2])
    ]
    with path.open('r') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i < 4:
                assert line == expected[i]
            else:
                np.testing.assert_array_almost_equal(np.array(line, dtype=float), expected[i])


def test_sensorstream_save_transient_domain_l2_sensor(sensor_handler):
    handler, cfg = sensor_handler
    cfg.time = "transient"
    cfg.time.timer.step = 0.1
    domain = DomainL2Sensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, handler.mesh, 'default', name="domainl2", rate=5)
    cfg.io.sensor.add(domain)
    with cfg.io as io:
        io.save_pre_time_routine(0.0)
        for it, t in enumerate(cfg.time.timer()):
            io.save_in_time_routine(t, it)
        io.save_post_time_routine(t, it)
    path = handler.path.joinpath("domainl2.csv")
    assert path.exists()
    expected = [
        ['region', 'default', 'default'],
        ['field', 'rho', 'u'],
        ['component', 'L2', 'L2'],
        ['t', ' ', ' '],
        np.array([0.0, np.sqrt(1/3), np.sqrt(2/3)]),
        np.array([0.1, np.sqrt(1/3), np.sqrt(2/3)]),
        np.array([0.6, np.sqrt(1/3), np.sqrt(2/3)]),
        np.array([1.0, np.sqrt(1/3), np.sqrt(2/3)])
    ]
    with path.open('r') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i < 4:
                assert line == expected[i]
            else:
                np.testing.assert_array_almost_equal(np.array(line, dtype=float), expected[i])


def test_sensorstream_save_transient_boundary_sensor(sensor_handler):
    handler, cfg = sensor_handler
    cfg.time = "transient"
    cfg.time.timer.step = 0.1
    boundary = BoundarySensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, handler.mesh, 'left|bottom', name="boundary", rate=5)
    cfg.io.sensor.add(boundary)
    with cfg.io as io:
        io.save_pre_time_routine(0.0)
        for it, t in enumerate(cfg.time.timer()):
            io.save_in_time_routine(t, it)
        io.save_post_time_routine(t, it)
    path = handler.path.joinpath("boundary.csv")
    assert path.exists()
    expected = [
        ['region', 'left|bottom', 'left|bottom', 'left|bottom'],
        ['field', 'rho', 'u', 'u'],
        ['component', ' ', 'x', 'y'],
        ['t', ' ', ' ', ' '],
        np.array([0.0, 0.5, 0.5, 0.5]),
        np.array([0.1, 0.5, 0.5, 0.5]),
        np.array([0.6, 0.5, 0.5, 0.5]),
        np.array([1.0, 0.5, 0.5, 0.5])
    ]
    with path.open('r') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i < 4:
                assert line == expected[i]
            else:
                np.testing.assert_array_almost_equal(np.array(line, dtype=float), expected[i])


def test_sensorstream_save_transient_boundary_l2_sensor(sensor_handler):
    handler, cfg = sensor_handler
    cfg.time = "transient"
    cfg.time.timer.step = 0.1
    boundary = BoundaryL2Sensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, handler.mesh, 'left|bottom', name="test", rate=5)
    cfg.io.sensor.add(boundary)
    with cfg.io as io:
        io.save_pre_time_routine(0.0)
        for it, t in enumerate(cfg.time.timer()):
            io.save_in_time_routine(t, it)
        io.save_post_time_routine(t, it)
    path = handler.path.joinpath("test.csv")
    assert path.exists()
    expected = [
        ['region', 'left|bottom', 'left|bottom'],
        ['field', 'rho', 'u'],
        ['component', 'L2', 'L2'],
        ['t', ' ', ' '],
        np.array([0.0, np.sqrt(1/3), np.sqrt(2/3)]),
        np.array([0.1, np.sqrt(1/3), np.sqrt(2/3)]),
        np.array([0.6, np.sqrt(1/3), np.sqrt(2/3)]),
        np.array([1.0, np.sqrt(1/3), np.sqrt(2/3)])
    ]
    with path.open('r') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i < 4:
                assert line == expected[i]
            else:
                np.testing.assert_array_almost_equal(np.array(line, dtype=float), expected[i])
