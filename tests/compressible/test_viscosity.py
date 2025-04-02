from __future__ import annotations
import unittest

from dream.compressible import flowfields
from tests.compressible.setup import cfg, mip

class TestInviscid(unittest.TestCase):

    def setUp(self) -> None:
        cfg.dynamic_viscosity = "inviscid"
        self.mu = cfg.dynamic_viscosity

    def test_is_inviscid(self):
        self.assertTrue(self.mu.is_inviscid)

    def test_viscosity(self):
        with self.assertRaises(TypeError):
            self.mu.viscosity(flowfields())


class TestConstant(unittest.TestCase):

    def setUp(self) -> None:
        cfg.dynamic_viscosity = "constant"
        self.mu = cfg.dynamic_viscosity

    def test_is_inviscid(self):
        self.assertFalse(self.mu.is_inviscid)

    def test_viscosity(self):
        fields = flowfields()
        self.assertAlmostEqual(self.mu.viscosity(fields), 1)


class TestSutherland(unittest.TestCase):

    def setUp(self) -> None:
        cfg.mach_number = 1
        cfg.equation_of_state.heat_capacity_ratio = 1.4

        cfg.dynamic_viscosity = "sutherland"
        cfg.dynamic_viscosity.measurement_temperature = 1

        self.mu = cfg.dynamic_viscosity

    def test_is_inviscid(self):
        self.assertFalse(self.mu.is_inviscid)

    def test_viscosity(self):

        fields = flowfields()
        self.assertIs(self.mu.viscosity(fields), None)

        fields = flowfields(temperature=1)

        cfg.scaling = "aerodynamic"
        cfg.scaling.dimensionful_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(fields)(mip), (0.4)**(3/2) * (2/0.4)/(1+1/0.4))

        cfg.scaling = "acoustic"
        cfg.scaling.dimensionful_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(fields)(mip), (0.4)**(3/2) * (2/0.4)/(1+1/0.4))

        cfg.scaling = "aeroacoustic"
        cfg.scaling.dimensionful_values.T = 1
        self.assertAlmostEqual(self.mu.viscosity(fields)(mip), (1.6)**(3/2) * (2/1.6)/(1+1/1.6))

