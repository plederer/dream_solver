from __future__ import annotations
import unittest
from tests import simplex

from dream.compressible import CompressibleFlowConfiguration, CompressibleState
from dream.compressible.viscosity import Inviscid, Constant, Sutherland
from tests.compressible.setup import cfg, mip

class TestInviscid(unittest.TestCase):

    def setUp(self) -> None:
        cfg.pde.dynamic_viscosity = "inviscid"
        self.mu = cfg.pde.dynamic_viscosity

    def test_is_inviscid(self):
        self.assertTrue(self.mu.is_inviscid)

    def test_viscosity(self):
        with self.assertRaises(TypeError):
            self.mu.viscosity(CompressibleState())


class TestConstant(unittest.TestCase):

    def setUp(self) -> None:
        cfg.pde.dynamic_viscosity = "constant"
        self.mu = cfg.pde.dynamic_viscosity

    def test_is_inviscid(self):
        self.assertFalse(self.mu.is_inviscid)

    def test_viscosity(self):
        state = CompressibleState()
        self.assertAlmostEqual(self.mu.viscosity(state), 1)


class TestSutherland(unittest.TestCase):

    def setUp(self) -> None:
        cfg.pde.mach_number = 1
        cfg.pde.equation_of_state.heat_capacity_ratio = 1.4

        cfg.pde.dynamic_viscosity = "sutherland"
        cfg.pde.dynamic_viscosity.measurement_temperature = 1

        self.mu = cfg.pde.dynamic_viscosity

    def test_is_inviscid(self):
        self.assertFalse(self.mu.is_inviscid)

    def test_viscosity(self):

        state = CompressibleState()
        self.assertIs(self.mu.viscosity(state), None)

        state = CompressibleState(temperature=1)

        cfg.pde.scaling = "aerodynamic"
        cfg.pde.scaling.reference_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(state)(mip), (0.4)**(3/2) * (2/0.4)/(1+1/0.4))

        cfg.pde.scaling = "acoustic"
        cfg.pde.scaling.reference_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(state)(mip), (0.4)**(3/2) * (2/0.4)/(1+1/0.4))

        cfg.pde.scaling = "aeroacoustic"
        cfg.pde.scaling.reference_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(state)(mip), (1.6)**(3/2) * (2/1.6)/(1+1/1.6))

