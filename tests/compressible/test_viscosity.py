from __future__ import annotations
import unittest
from tests import simplex

from dream.compressible import CompressibleFlowConfiguration, CompressibleState
from dream.compressible.viscosity import Inviscid, Constant, Sutherland
class TestInviscid(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = Inviscid()

    def test_is_inviscid(self):
        self.assertTrue(self.mu.is_inviscid)

    def test_viscosity(self):
        with self.assertRaises(TypeError):
            self.mu.viscosity(CompressibleState())


class TestConstant(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = Constant()

    def test_is_inviscid(self):
        self.assertFalse(self.mu.is_inviscid)

    def test_viscosity(self):
        state = CompressibleState()
        self.assertAlmostEqual(self.mu.viscosity(state), 1)


class TestSutherland(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = Sutherland()
        self.mu.measurement_temperature = 1

        self.cfg = CompressibleFlowConfiguration()
        self.cfg.mach_number = 1
        self.cfg.equation_of_state.heat_capacity_ratio = 1.4

        self.mesh = simplex()
        self.mip = self.mesh(0.25, 0.25)

    def test_is_inviscid(self):
        self.assertFalse(self.mu.is_inviscid)

    def test_viscosity(self):

        state = CompressibleState()
        self.assertIs(self.mu.viscosity(state, self.cfg.equations), None)

        state = CompressibleState(temperature=1)

        self.cfg.scaling = "aerodynamic"
        self.cfg.scaling.reference_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(state, self.cfg.equations)(self.mip), (0.4)**(3/2) * (2/0.4)/(1+1/0.4))

        self.cfg.scaling = "acoustic"
        self.cfg.scaling.reference_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(state, self.cfg.equations)(self.mip), (0.4)**(3/2) * (2/0.4)/(1+1/0.4))

        self.cfg.scaling = "aeroacoustic"
        self.cfg.scaling.reference_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(state, self.cfg.equations)(self.mip), (1.6)**(3/2) * (2/1.6)/(1+1/1.6))

