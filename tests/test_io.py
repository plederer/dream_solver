
from __future__ import annotations
import unittest
import ngsolve as ngs
import numpy as np
from pathlib import Path

from tests import simplex
from dream.solver import SolverConfiguration
from dream.io import MeshStream, VTKStream, StateStream, TimeStateStream, SettingsStream


class TestMeshStream(unittest.TestCase):

    def setUp(self) -> None:
        cfg = SolverConfiguration(mesh=simplex())
        self.handler = MeshStream(cfg=cfg, mesh=cfg.mesh)

    def test_initialize(self):
        handler = self.handler.initialize()
        self.assertIs(handler, self.handler)

    def test_default_save_and_load(self):

        self.handler.save()
        path = Path().cwd().joinpath("mesh.pickle")
        self.assertTrue(path.exists())

        mesh = self.handler.load()
        self.assertIsInstance(mesh, ngs.Mesh)
        path.unlink()

    def test_save_and_load(self):

        self.handler.filename = "test"

        self.handler.save()
        path = Path().cwd().joinpath("test.pickle")
        self.assertTrue(path.exists())

        mesh = self.handler.load()
        self.assertIsInstance(mesh, ngs.Mesh)
        path.unlink()


class TestVTKStream(unittest.TestCase):

    def setUp(self) -> None:
        cfg = SolverConfiguration(mesh=simplex())
        self.handler = VTKStream(cfg=cfg, mesh=cfg.mesh)
        self.handler.fields = {'u': ngs.CoefficientFunction(1)}

    def test_initialize(self):
        self.assertIs(self.handler.initialize(), self.handler)

    def test_default_save(self):

        self.handler.initialize()
        for t in range(5):
            self.handler.save(t)
        path = self.handler.path.joinpath(f"vtk.pvd")
        self.assertTrue(path.exists())

    def test_save(self):
        self.handler.filename = "test"
        self.handler.initialize()
        for t in range(5):
            self.handler.save(t)
        path = self.handler.path.joinpath(f"test.pvd")
        self.assertTrue(path.exists())

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()


class TestStateStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = SolverConfiguration(mesh=simplex(), pde="dummy")
        self.cfg.pde.initialize_system()
        self.cfg.pde.gfu.vec[:] = 1

        self.handler = StateStream(cfg=self.cfg, mesh=self.cfg.mesh)

    def test_initialize(self):
        self.assertIs(self.handler.initialize(), self.handler)

    def test_save(self):

        self.handler.filename = "test"
        self.handler.initialize()
        self.handler.save(t=0.0)
        path = self.handler.path.joinpath(f"test_0.0.ngs")
        self.assertTrue(path.exists())

        gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)
        gfu.Load(str(path))
        np.testing.assert_array_equal(gfu.vec, self.cfg.pde.gfu.vec)

    def test_save_and_load_gridfunction(self):
        path = self.handler.path.joinpath(f"test.ngs")
        gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)

        self.handler.initialize()

        self.handler.save_gridfunction(self.cfg.pde.gfu, 'test')
        self.assertTrue(path.exists())

        self.handler.load_gridfunction(gfu, 'test')
        np.testing.assert_array_equal(gfu.vec, self.cfg.pde.gfu.vec)

    def test_load_gridfunction_sequence(self):

        self.handler.filename = "test"
        self.handler.initialize()

        for t in range(5):
            self.cfg.pde.gfu.vec[:] = t
            self.handler.save(t)

        for t in range(5):
            gfu = ngs.GridFunction(ngs.L2(self.cfg.mesh, order=0)**2)
            gfu.vec[:] = t
            self.handler.load_gridfunction_sequence(t, f"test")
            np.testing.assert_array_equal(gfu.vec, self.cfg.pde.gfu.vec)

    def tearDown(self):
        for path in self.handler.path.iterdir():
            path.unlink()
        self.handler.path.rmdir()


class TestTimeStateStream(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = SolverConfiguration(mesh=simplex(), pde="dummy")
        self.cfg.time = "transient"
        self.cfg.pde.initialize_system()

        self.handler = TimeStateStream(cfg=self.cfg, mesh=self.cfg.mesh)

    def test_save_and_load_gridfunction(self):

        self.handler.initialize()

        for fes in self.cfg.pde.transient_gfus:
            for level in self.cfg.pde.transient_gfus[fes]:
                gfu = self.cfg.pde.transient_gfus[fes][level]
                gfu.vec[:] = 1

        self.handler.save_gridfunction(0.0, 'test')
        for fes in self.cfg.pde.transient_gfus:
            for level in self.cfg.pde.transient_gfus[fes]:
                self.assertTrue(self.handler.path.joinpath(f"test_0.0_{fes}_{level}.ngs").exists())

        self.handler.load_gridfunction(0.0, 'test')
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
        self.assertIs(self.handler.initialize(), self.handler)

    def test_save_and_load_to_pickle(self):
        self.handler.initialize()

        path = self.handler.path.joinpath(f"test.pickle")
        self.handler.save_to_pickle('test')
        self.assertTrue(path.exists())

        cfg = self.handler.load_from_pickle('test')
        self.assertDictEqual(cfg, self.cfg.to_tree())
        path.unlink()