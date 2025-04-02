from __future__ import annotations
import unittest
from tests.compressible.setup import cfg, mip


class TestAerodynamic(unittest.TestCase):

    def setUp(self) -> None:
        cfg.scaling = "aerodynamic"
        self.scaling = cfg.scaling

    def test_velocity_magnitude(self):
        self.assertAlmostEqual(self.scaling.velocity_magnitude(0.1), 1)

    def test_speed_of_sound(self):
        self.assertAlmostEqual(self.scaling.speed_of_sound(0.2), 5)

        with self.assertRaises(ValueError):
            self.scaling.speed_of_sound(0)


class TestAcoustic(unittest.TestCase):

    def setUp(self) -> None:
        cfg.scaling = "acoustic"
        self.scaling = cfg.scaling


    def test_velocity_magnitude(self):
        self.assertAlmostEqual(self.scaling.velocity_magnitude(0.1), 0.1)

    def test_speed_of_sound(self):
        self.assertAlmostEqual(self.scaling.speed_of_sound(0.2), 1)


class TestAeroacoustic(unittest.TestCase):

    def setUp(self) -> None:
        cfg.scaling = "aeroacoustic"
        self.scaling = cfg.scaling


    def test_velocity_magnitude(self):
        self.assertAlmostEqual(self.scaling.velocity_magnitude(0.1), 0.1/(1 + 0.1))

    def test_speed_of_sound(self):
        self.assertAlmostEqual(self.scaling.speed_of_sound(0.2), 1/(1 + 0.2))
