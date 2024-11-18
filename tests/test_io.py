
from __future__ import annotations
import unittest
import ngsolve as ngs
import numpy as np
from pathlib import Path

from tests import simplex
from dream.solver import SolverConfiguration
from dream.io import MeshStream, VTKStream, GridfunctionStream, TransientGridfunctionStream, SettingsStream, SensorStream


class TestMeshStream(unittest.TestCase):

    def setUp(self) -> None:
        cfg = SolverConfiguration(mesh=simplex())
        self.handler = MeshStream(cfg=cfg, mesh=cfg.mesh)

    def test_initialize(self):
        handler = self.handler.open()
        self.assertIs(handler, self.handler)

    def test_default_save_and_load(self):

        self.handler.save_pre_time_routine()
        path = Path().cwd().joinpath("mesh.pickle")
        self.assertTrue(path.exists())

        mesh = self.handler.load_routine()
        self.assertIsInstance(mesh, ngs.Mesh)
        path.unlink()

    def test_save_and_load(self):

        self.handler.filename = "test"

        self.handler.save_pre_time_routine()
        path = Path().cwd().joinpath("test.pickle")
        self.assertTrue(path.exists())

        mesh = self.handler.load_routine()
        self.assertIsInstance(mesh, ngs.Mesh)
        path.unlink()


class TestVTKStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = SolverConfiguration(mesh=simplex())
        self.handler = VTKStream(cfg=self.cfg, mesh=self.cfg.mesh)
        self.handler.fields = {'u': ngs.CoefficientFunction(1)}

    def test_initialize(self):
        self.assertIs(self.handler.open(), self.handler)

    def test_save_pre_time_routine(self):
        self.handler.open()

        self.cfg.time = "stationary"
        self.handler.save_pre_time_routine()

        files = [file for file in self.handler.path.iterdir()]
        self.assertListEqual(files, [])

        self.cfg.time = "transient"
        self.handler.save_pre_time_routine()
        self.handler.save_pre_time_routine()

        path = self.handler.path.joinpath(f"vtk.pvd")
        self.assertTrue(path.exists())

        with path.open('r') as open:
            data = open.readlines()

        self.assertListEqual(data[3:5], [
            """<DataSet timestep="0" file="vtk.vtu"/>\n""",
            """<DataSet timestep="0" file="vtk_step00001.vtu"/>\n"""])

    def test_default_save_in_time_routine(self):

        self.handler.open()
        for t in range(5):
            self.handler.save_in_time_routine(t, 0)
        path = self.handler.path.joinpath(f"vtk.pvd")
        self.assertTrue(path.exists())

    def test_save_in_time_routine(self):
        self.handler.filename = "test"
        self.handler.open()
        for t in range(5):
            self.handler.save_in_time_routine(t, 0)
        path = self.handler.path.joinpath(f"test.pvd")
        self.assertTrue(path.exists())

    def test_saving_rate_in_time_routine(self):
        self.handler.rate = 2
        self.handler.open()
        for it, t in enumerate(range(10)):
            self.handler.save_in_time_routine(t, it)

        path = self.handler.path.joinpath(f"vtk.pvd")
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

        self.handler.open()
        self.handler.save_post_time_routine(t=None)
        path = self.handler.path.joinpath('vtk.vtu')
        self.assertTrue(path.exists())

    def test_save_post_time_routine_transient(self):

        self.handler.open()
        self.handler.rate = 3

        self.handler.save_post_time_routine(t=0.1, it=10)
        path = self.handler.path.joinpath('vtk.vtu')
        self.assertTrue(path.exists())

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()


class TestGridfunctionStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = SolverConfiguration(mesh=simplex(), pde="dummy")
        self.cfg.time = "transient"
        self.cfg.time.timer.step = 1
        self.cfg.pde.initialize_system()
        self.cfg.pde.gfu.vec[:] = 1

        self.handler = GridfunctionStream(cfg=self.cfg, mesh=self.cfg.mesh)

    def test_initialize(self):
        self.assertIs(self.handler.open(), self.handler)

    def test_save_pre_time_routine(self):
        self.handler.open()

        self.cfg.time = "stationary"
        self.handler.save_pre_time_routine()

        files = [file for file in self.handler.path.iterdir()]
        self.assertListEqual(files, [])

        self.cfg.time = "transient"
        self.handler.save_pre_time_routine()

        path = self.handler.path.joinpath(f"gfu_0.0.ngs")
        self.assertTrue(path.exists())

    def test_save_in_time_routine(self):

        self.handler.filename = "test"
        self.handler.open()
        self.handler.save_in_time_routine(t=0.0)
        path = self.handler.path.joinpath(f"test_0.0.ngs")
        self.assertTrue(path.exists())

        gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)
        gfu.Load(str(path))
        np.testing.assert_array_equal(gfu.vec, self.cfg.pde.gfu.vec)

    def test_saving_rate_in_time_routine(self):
        self.handler.open()
        self.handler.rate = 2
        for it, t in enumerate(range(10)):
            self.handler.save_in_time_routine(t, it)

        for t in range(0, 10, 2):
            path = self.handler.path.joinpath(f"gfu_{t}.ngs")
            self.assertTrue(path.exists())

    def test_save_post_time_routine_stationary(self):

        self.handler.open()
        self.handler.save_post_time_routine(t=None)
        path = self.handler.path.joinpath('gfu.ngs')
        self.assertTrue(path.exists())

    def test_save_post_time_routine_transient(self):

        self.handler.open()
        self.handler.rate = 3

        self.handler.save_post_time_routine(t=0.1, it=10)
        path = self.handler.path.joinpath('gfu_0.1.ngs')
        self.assertTrue(path.exists())

    def test_save_and_load_gridfunction(self):
        path = self.handler.path.joinpath(f"test.ngs")
        gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)

        self.handler.open()

        self.handler.save_gridfunction(self.cfg.pde.gfu, 'test')
        self.assertTrue(path.exists())

        self.handler.load_gridfunction(gfu, 'test')
        np.testing.assert_array_equal(gfu.vec, self.cfg.pde.gfu.vec)

    def test_load_transient_routine(self):

        self.handler.filename = "test"
        self.handler.open()

        for t in self.cfg.time.timer.start(True):
            self.cfg.pde.gfu.vec[:] = t
            self.handler.save_in_time_routine(t)

        gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)
        for t in self.handler.load_transient_routine():
            gfu.vec[:] = t
            np.testing.assert_array_equal(gfu.vec, self.cfg.pde.gfu.vec)

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()


class TestTransientGridfunctionStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = SolverConfiguration(mesh=simplex(), pde="dummy")
        self.cfg.time = "transient"
        self.cfg.pde.initialize_system()

        self.handler = TransientGridfunctionStream(cfg=self.cfg, mesh=self.cfg.mesh)

    def test_save_and_load_gridfunction(self):

        self.handler.open()
        self.handler.filename = 'test'

        for fes in self.cfg.pde.transient_gfus:
            for level in self.cfg.pde.transient_gfus[fes]:
                gfu = self.cfg.pde.transient_gfus[fes][level]
                gfu.vec[:] = 1

        self.handler.save_in_time_routine(0.0, it=0)
        for fes in self.cfg.pde.transient_gfus:
            for level in self.cfg.pde.transient_gfus[fes]:
                self.assertTrue(self.handler.path.joinpath(f"test_0.0_{fes}_{level}.ngs").exists())

        self.handler.load_time_levels(0.0)
        for fes in self.cfg.pde.transient_gfus:
            for level in self.cfg.pde.transient_gfus[fes]:
                np.testing.assert_array_equal(
                    self.cfg.pde.transient_gfus[fes][level].vec, np.ones_like(
                        self.cfg.pde.transient_gfus[fes][level].vec))

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()


class TestSettingsStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = SolverConfiguration(mesh=simplex(), pde="dummy")
        self.handler = SettingsStream(cfg=self.cfg, mesh=self.cfg.mesh)

    def test_initialize(self):
        self.assertIs(self.handler.open(), self.handler)

    def test_save_and_load_to_pickle(self):
        self.handler.open()

        path = self.handler.path.joinpath(f"test.pickle")
        self.handler.save_to_pickle('test')
        self.assertTrue(path.exists())

        cfg = self.handler.load_from_pickle('test')
        self.assertDictEqual(cfg, self.cfg.to_tree())
        path.unlink()


class TestSensorStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = SolverConfiguration(mesh=simplex(), pde="dummy")
        self.cfg.io.sensor = True
        self.handler = self.cfg.io.sensor
        self.handler.to_csv = True

    def test_initialize(self):
        self.assertIs(self.handler.open(), self.handler)

    def test_save_transient_point_sensor(self):
        import csv

        self.cfg.time = "transient"
        self.cfg.time.timer.step = 0.1

        self.handler.point = "test"
        self.handler.point['test'].points = ((0, 0), (1, 0), (0, 1))
        self.handler.point['test'].fields = {'rho': ngs.x, 'u': (ngs.x, ngs.y)}
        self.handler.point['test'].rate = 5

        with self.cfg.io as io:
            io.save_pre_time_routine()
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.handler.path.joinpath(f"test.csv")
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

        self.handler.domain = "test"
        self.handler.domain['test'].fields = {'rho': ngs.x, 'u': (ngs.x, ngs.y)}
        self.handler.domain['test'].rate = 5

        with self.cfg.io as io:
            io.save_pre_time_routine()
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.handler.path.joinpath(f"test.csv")
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

        self.handler.domain_L2 = "test"
        self.handler.domain_L2['test'].fields = {'rho': ngs.x, 'u': (ngs.x, ngs.y)}
        self.handler.domain_L2['test'].rate = 5

        with self.cfg.io as io:
            io.save_pre_time_routine()
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.handler.path.joinpath(f"test.csv")
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

        self.handler.boundary = "test"
        self.handler.boundary['test'].regions = 'left|bottom'
        self.handler.boundary['test'].fields = {'rho': ngs.x, 'u': (ngs.x, ngs.y)}
        self.handler.boundary['test'].rate = 5

        with self.cfg.io as io:
            io.save_pre_time_routine()
            for it, t in enumerate(self.cfg.time.timer()):
                io.save_in_time_routine(t, it)
            io.save_post_time_routine(t, it)

        path = self.handler.path.joinpath(f"test.csv")
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

        self.handler.boundary_L2 = "test"
        self.handler.boundary_L2['test'].regions = 'left|bottom'
        self.handler.boundary_L2['test'].fields = {'rho': ngs.x, 'u': (ngs.x, ngs.y)}
        self.handler.boundary_L2['test'].rate = 5

        with self.cfg.io as io:
            io.save_pre_time_routine()
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
        # self.handler.open()

        # self.handler.save_pre_time_routine()
        # for it, t in enumerate(self.cfg.time.timer()):
        #     self.handler.save_in_time_routine(t, it)
        # self.handler.save_post_time_routine(t, it)
        # self.handler.close()

        # df = io.sensor.load_as_dataframe('test')

        # path = self.handler.path.joinpath(f"test.pickle")
        # self.handler.save_to_pickle('test')
        # self.assertTrue(path.exists())

        # cfg = self.handler.load_from_pickle('test')
        # self.assertDictEqual(cfg, self.cfg.to_tree())
        # path.unlink()

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()
