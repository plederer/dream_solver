
from __future__ import annotations
import unittest
import ngsolve as ngs
import numpy as np

from tests import simplex, DummySolverConfiguration
from dream.io import MeshStream, VTKStream, GridfunctionStream, SettingsStream, SensorStream, PointSensor, DomainSensor, DomainL2Sensor, BoundarySensor, BoundaryL2Sensor


class TestMeshStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = DummySolverConfiguration(mesh=simplex())
        self.handler = MeshStream(self.cfg.mesh, self.cfg, path="mesh_test")

    def test_initialize(self):
        handler = self.handler.open()
        self.assertIs(handler, self.handler)

    def test_default_save_and_load(self):
        handler = self.handler.open()

        handler.save_pre_time_routine()
        path = handler.path.joinpath("mesh.pickle")
        self.assertTrue(path.exists())

        mesh = handler.load_routine()
        self.assertIsInstance(mesh, ngs.Mesh)
        path.unlink()

    def test_save_and_load(self):

        handler = self.handler.open()

        handler.filename = "test"

        handler.save_pre_time_routine()
        path = handler.path.joinpath("test.pickle")
        self.assertTrue(path.exists())

        mesh = handler.load_routine()
        self.assertIsInstance(mesh, ngs.Mesh)
        path.unlink()

    def tearDown(self):
        self.handler.path.rmdir()


class TestVTKStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = DummySolverConfiguration(mesh=simplex())
        self.handler = VTKStream(self.cfg.mesh, self.cfg.root, path="vtk_test")
        self.handler.fields = {'u': ngs.CoefficientFunction(1)}

    def test_initialize(self):
        self.assertIs(self.handler.open(), self.handler)

    def test_save_pre_time_routine(self):
        handler = self.handler.open()

        self.cfg.time = "stationary"
        handler.save_pre_time_routine()

        path = handler.path.joinpath(f"vtk.vtu")
        self.assertTrue(path.exists())

        self.cfg.time = "transient"
        handler.save_pre_time_routine(0.0)
        handler.save_pre_time_routine(0.0)

        path = handler.path.joinpath(f"vtk.pvd")
        self.assertTrue(path.exists())

        with path.open('r') as open:
            data = open.readlines()

        self.assertListEqual(data[3:5], [
            """<DataSet timestep="0" file="vtk.vtu"/>\n""",
            """<DataSet timestep="0" file="vtk_step00001.vtu"/>\n"""])

    def test_default_save_in_time_routine(self):

        handler = self.handler.open()
        for t in range(5):
            handler.save_in_time_routine(t, 0)
        path = handler.path.joinpath(f"vtk.pvd")
        self.assertTrue(path.exists())

    def test_save_in_time_routine(self):
        self.handler.filename = "test"
        handler = self.handler.open()
        for t in range(5):
            handler.save_in_time_routine(t, 0)
        path = handler.path.joinpath(f"test.pvd")
        self.assertTrue(path.exists())

    def test_saving_rate_in_time_routine(self):
        self.handler.rate = 2
        handler = self.handler.open()
        for it, t in enumerate(range(10)):
            handler.save_in_time_routine(t, it)

        path = handler.path.joinpath(f"vtk.pvd")
        self.assertTrue(path.exists())

        with path.open('r') as open:
            data = open.readlines()

        self.assertListEqual(data[3:8], [
            """<DataSet timestep="0" file="vtk.vtu"/>\n""",
            """<DataSet timestep="2" file="vtk_step00001.vtu"/>\n""",
            """<DataSet timestep="4" file="vtk_step00002.vtu"/>\n""",
            """<DataSet timestep="6" file="vtk_step00003.vtu"/>\n""",
            """<DataSet timestep="8" file="vtk_step00004.vtu"/>\n"""])

    def test_save_post_time_routine_stationary(self):

        handler = self.handler.open()
        handler.save_post_time_routine(t=None)
        path = handler.path.joinpath('vtk.vtu')
        self.assertTrue(path.exists())

    def test_save_post_time_routine_transient(self):

        handler = self.handler.open()
        handler.rate = 3

        handler.save_post_time_routine(t=0.1, it=10)
        path = handler.path.joinpath('vtk.vtu')
        self.assertTrue(path.exists())

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()
        self.cfg.io.path.rmdir()


class TestGridfunctionStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = DummySolverConfiguration(mesh=simplex())
        self.cfg.time = "transient"
        self.cfg.time.timer.step = 1
        self.cfg.initialize()
        self.cfg.gfu.vec[:] = 1

        self.handler = GridfunctionStream(self.cfg.mesh, self.cfg.root, path="gfu_test")

    def test_initialize(self):
        self.assertIs(self.handler.open(), self.handler)

    def test_save_pre_time_routine(self):
        handler = self.handler.open()

        self.cfg.time = "stationary"
        handler.save_pre_time_routine()

        path = handler.path.joinpath(f"gfu.ngs")
        self.assertTrue(path.exists())

        self.cfg.time = "transient"
        self.cfg.time.initialize()
        handler.save_pre_time_routine(0.0)

        path = handler.path.joinpath(f"gfu_0.0.ngs")
        self.assertTrue(path.exists())

    def test_save_in_time_routine(self):

        self.handler.filename = "test"
        handler = self.handler.open()
        handler.save_in_time_routine(t=0.0)
        path = handler.path.joinpath(f"test_0.0.ngs")
        self.assertTrue(path.exists())

        gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)
        gfu.Load(str(path))
        np.testing.assert_array_equal(gfu.vec, self.cfg.gfu.vec)

    def test_saving_rate_in_time_routine(self):
        self.handler.rate = 2
        handler = self.handler.open()
        for it, t in enumerate(range(10)):
            handler.save_in_time_routine(t, it)

        for t in range(0, 10, 2):
            path = handler.path.joinpath(f"gfu_{t}.ngs")
            self.assertTrue(path.exists())

    def test_save_post_time_routine_stationary(self):

        handler = self.handler.open()
        handler.save_post_time_routine(t=None)
        path = handler.path.joinpath('gfu.ngs')
        self.assertTrue(path.exists())

    def test_save_post_time_routine_transient(self):

        self.handler.rate = 3
        handler = self.handler.open()

        handler.save_post_time_routine(t=0.1, it=10)
        path = handler.path.joinpath('gfu_0.1.ngs')
        self.assertTrue(path.exists())

    def test_save_and_load_gridfunction(self):

        path = self.handler.path.joinpath(f"test.ngs")
        gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)

        handler = self.handler.open()

        handler.save_gridfunction(self.cfg.gfu, 'test')
        self.assertTrue(path.exists())

        handler.load_gridfunction(gfu, 'test')
        np.testing.assert_array_equal(gfu.vec, self.cfg.gfu.vec)

    def test_load_transient_routine(self):

        self.handler.filename = "test"
        handler = self.handler.open()

        for t in self.cfg.time.timer.start(True):
            self.cfg.gfu.vec[:] = t
            handler.save_in_time_routine(t)

        gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)
        for t in handler.load_transient_routine():
            gfu.vec[:] = t
            np.testing.assert_array_equal(gfu.vec, self.cfg.gfu.vec)

    def test_save_and_load_gridfunction(self):

        self.handler.filename = 'test'
        handler = self.handler.open()

        for fes in self.cfg.time.scheme.gfus:
            for level in self.cfg.time.scheme.gfus[fes]:
                gfu = self.cfg.time.scheme.gfus[fes][level]
                gfu.vec[:] = 1

        handler.save_in_time_routine(0.0, it=0)
        for fes in self.cfg.time.scheme.gfus:
            for level in self.cfg.time.scheme.gfus[fes]:
                self.assertTrue(handler.path.joinpath(f"test_0.0_{fes}_{level}.ngs").exists())

        handler.load_time_levels(0.0)
        for fes in self.cfg.time.scheme.gfus:
            for level in self.cfg.time.scheme.gfus[fes]:
                np.testing.assert_array_equal(
                    self.cfg.time.scheme.gfus[fes][level].vec, np.ones_like(
                        self.cfg.time.scheme.gfus[fes][level].vec))

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()
        self.cfg.io.path.rmdir()


class TestSettingsStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = DummySolverConfiguration(mesh=simplex())
        self.handler = SettingsStream(self.cfg.mesh, self.cfg, path="settings_test")

    def test_initialize(self):
        self.assertIs(self.handler.open(), self.handler)

    def test_save_and_load_to_pickle(self):
        handler = self.handler.open()

        path = handler.path.joinpath(f"test.pickle")
        self.handler.save_to_pickle('test')
        self.assertTrue(path.exists())

        cfg = self.handler.load_from_pickle('test')
        self.assertDictEqual(cfg, self.cfg.to_dict())
        path.unlink()

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()
        self.cfg.io.path.rmdir()


class TestSensorStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = DummySolverConfiguration(mesh=simplex())
        self.cfg.io.sensor.path = "sensor_test"
        self.handler = SensorStream(self.cfg.mesh, self.cfg, path="sensor_test")
        self.handler.to_csv = True

    def test_empty_initialize(self):
        self.assertIs(self.handler.open(), None)

    def test_initialize(self):
        point = PointSensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, self.handler.mesh,
                            ((0, 0), (1, 0), (0, 1)), name="asdasa")

        self.handler.add(point)
        self.assertIs(self.handler.open(), self.handler)
        self.handler.close()

    def test_save_transient_point_sensor(self):
        import csv
        point = PointSensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, self.handler.mesh,
                            ((0, 0), (1, 0), (0, 1)), name="point", rate=5)
        self.cfg.io.sensor.add(point)

        self.cfg.time.timer.step = 0.1

        with self.cfg.io as io:
            io.save_pre_time_routine(0.0)
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.cfg.io.sensor.path.joinpath(f"point.csv")
        self.assertTrue(path.exists())

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
                self.assertListEqual(line, expected[i])

    def test_save_transient_domain_sensor(self):
        import csv

        self.cfg.time = "transient"
        self.cfg.time.timer.step = 0.1

        domain = DomainSensor({'rho': ngs.x, 'u': (ngs.x, ngs.y)}, self.handler.mesh, 'default', name="domain", rate=5)
        self.cfg.io.sensor.add(domain)

        with self.cfg.io as io:
            io.save_pre_time_routine(0.0)
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.handler.path.joinpath(f"domain.csv")
        self.assertTrue(path.exists())

        expected = [
            ['region', 'default', 'default', 'default'],
            ['field', 'rho', 'u', 'u'],
            ['component', ' ', 'x', 'y'],
            ['t', ' ', ' ', ' '],
            np.array([0.0, 1/6, 1/6, 1/6]),
            np.array([0.1, 1/6, 1/6, 1/6]),
            np.array([0.6, 1/6, 1/6, 1/6]),
            np.array([1.0, 1/6, 1/6, 1/6])
        ]

        with path.open('r') as file:
            reader = csv.reader(file)
            for i, line in enumerate(reader):
                if i < 4:
                    self.assertListEqual(line, expected[i])
                else:
                    np.testing.assert_array_almost_equal(np.array(line, dtype=float), expected[i])

    def test_save_transient_domain_l2_sensor(self):
        import csv

        self.cfg.time = "transient"
        self.cfg.time.timer.step = 0.1

        domain = DomainL2Sensor(
            {'rho': ngs.x, 'u': (ngs.x, ngs.y)},
            self.handler.mesh, 'default', name="domainl2", rate=5)
        self.cfg.io.sensor.add(domain)

        with self.cfg.io as io:
            io.save_pre_time_routine(0.0)
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.handler.path.joinpath(f"domainl2.csv")
        self.assertTrue(path.exists())

        expected = [
            ['region', 'default', 'default'],
            ['field', 'rho', 'u'],
            ['component', 'L2', 'L2'],
            ['t', ' ', ' '],
            np.array([0.0, np.sqrt(1/12), np.sqrt(2/12)]),
            np.array([0.1, np.sqrt(1/12), np.sqrt(2/12)]),
            np.array([0.6, np.sqrt(1/12), np.sqrt(2/12)]),
            np.array([1.0, np.sqrt(1/12), np.sqrt(2/12)])
        ]

        with path.open('r') as file:
            reader = csv.reader(file)
            for i, line in enumerate(reader):
                if i < 4:
                    self.assertListEqual(line, expected[i])
                else:
                    np.testing.assert_array_almost_equal(np.array(line, dtype=float), expected[i])

    def test_save_transient_boundary_sensor(self):
        import csv

        self.cfg.time = "transient"
        self.cfg.time.timer.step = 0.1

        boundary = BoundarySensor(
            {'rho': ngs.x, 'u': (ngs.x, ngs.y)},
            self.handler.mesh, 'left|bottom', name="boundary", rate=5)
        self.cfg.io.sensor.add(boundary)

        with self.cfg.io as io:
            io.save_pre_time_routine(0.0)
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.handler.path.joinpath(f"boundary.csv")
        self.assertTrue(path.exists())

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
                    self.assertListEqual(line, expected[i])
                else:
                    np.testing.assert_array_almost_equal(np.array(line, dtype=float), expected[i])

    def test_save_transient_boundary_l2_sensor(self):
        import csv

        self.cfg.time = "transient"
        self.cfg.time.timer.step = 0.1

        boundary = BoundaryL2Sensor(
            {'rho': ngs.x, 'u': (ngs.x, ngs.y)},
            self.handler.mesh, 'left|bottom', name="test", rate=5)
        self.cfg.io.sensor.add(boundary)

        with self.cfg.io as io:
            io.save_pre_time_routine(0.0)
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.handler.path.joinpath(f"test.csv")
        self.assertTrue(path.exists())

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
                    self.assertListEqual(line, expected[i])
                else:
                    np.testing.assert_array_almost_equal(np.array(line, dtype=float), expected[i])

    def tearDown(self):
        if self.handler.path.exists():
            for path in self.handler.path.iterdir():
                path.unlink()

            self.handler.path.rmdir()

        if self.cfg.io.path.exists():
            self.cfg.io.path.rmdir()
