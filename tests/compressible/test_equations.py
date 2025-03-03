from __future__ import annotations
import unittest
import numpy.testing as nptest
from tests import simplex
import ngsolve as ngs

from dream.compressible.config import flowfields
from tests.compressible.setup import cfg, mip

def test_equation(throws: bool = False, is_vector: bool = False):

    def wrapper(func):

        def test(self):
            name = func.__name__.split("test_")[1]

            if throws:
                U = flowfields()
                with self.assertRaises(ValueError):
                    getattr(self.cfg, name)(U)

            value = 1
            if is_vector:
                value = (1, 0)

            U.update({name: value})
            self.assertAlmostEqual(getattr(self.cfg, name)(U)(mip), value)

            func(self)

        return test

    return wrapper

class TestCompressibleEquations(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = cfg
        self.cfg.reynolds_number = 2
        self.cfg.prandtl_number = 1
        self.cfg.dynamic_viscosity = "constant"

    @test_equation(throws=True)
    def test_density(self):
        ...

    @test_equation(throws=True, is_vector=True)
    def test_velocity(self):
        U = flowfields(density=2, momentum=(2, 2))
        nptest.assert_almost_equal(cfg.velocity(U)(mip), (1, 1))

    @test_equation(throws=True, is_vector=True)
    def test_momentum(self):
        U = flowfields(density=0.5, velocity=(1, 1))
        nptest.assert_almost_equal(cfg.momentum(U)(mip), (0.5, 0.5))

    @test_equation(throws=True)
    def test_pressure(self):
        ...

    @test_equation(throws=True)
    def test_temperature(self):
        ...

    @test_equation(throws=True)
    def test_inner_energy(self):
        U = flowfields(energy=2, kinetic_energy=1)
        self.assertAlmostEqual(cfg.inner_energy(U)(mip), 1)

    @test_equation(throws=True)
    def test_specific_inner_energy(self):
        U = flowfields(specific_energy=2, specific_kinetic_energy=1)
        self.assertAlmostEqual(cfg.specific_inner_energy(U)(mip), 1)

        U = flowfields(inner_energy=2, density=1)
        self.assertAlmostEqual(cfg.specific_inner_energy(U)(mip), 2)

    @test_equation(throws=True)
    def test_kinetic_energy(self):
        U = flowfields(density=2, velocity=(2, 2))
        self.assertAlmostEqual(cfg.kinetic_energy(U)(mip), 8)

        U = flowfields(density=2, momentum=(2, 2))
        self.assertAlmostEqual(cfg.kinetic_energy(U)(mip), 2)

        U = flowfields(energy=2, inner_energy=1)
        self.assertAlmostEqual(cfg.kinetic_energy(U)(mip), 1)

        U = flowfields(specific_kinetic_energy=2, density=2)
        self.assertAlmostEqual(cfg.kinetic_energy(U)(mip), 4)

    @test_equation(throws=True)
    def test_specific_kinetic_energy(self):
        U = flowfields(velocity=(2, 2))
        self.assertAlmostEqual(cfg.specific_kinetic_energy(U)(mip), 4)

        U = flowfields(density=2, momentum=(2, 2))
        self.assertAlmostEqual(cfg.specific_kinetic_energy(U)(mip), 1)

        U = flowfields(specific_energy=2, specific_inner_energy=1)
        self.assertAlmostEqual(cfg.specific_kinetic_energy(U)(mip), 1)

        U = flowfields(kinetic_energy=2, density=2)
        self.assertAlmostEqual(cfg.specific_kinetic_energy(U)(mip), 1)

    @test_equation(throws=True)
    def test_energy(self):
        U = flowfields(specific_energy=2, density=2)
        self.assertAlmostEqual(cfg.energy(U)(mip), 4)

        U = flowfields(kinetic_energy=2, inner_energy=2)
        self.assertAlmostEqual(cfg.energy(U)(mip), 4)

    @test_equation(throws=True)
    def test_specific_energy(self):
        U = flowfields(energy=2, density=2)
        self.assertAlmostEqual(cfg.specific_energy(U)(mip), 1)

        U = flowfields(specific_kinetic_energy=2, specific_inner_energy=2)
        self.assertAlmostEqual(cfg.specific_energy(U)(mip), 4)

    @test_equation(throws=True)
    def test_enthalpy(self):
        U = flowfields(pressure=2, energy=2)
        self.assertAlmostEqual(cfg.enthalpy(U)(mip), 4)

        U = flowfields(specific_enthalpy=2, density=2)
        self.assertAlmostEqual(cfg.enthalpy(U)(mip), 4)

    @test_equation(throws=True)
    def test_specific_enthalpy(self):
        U = flowfields(enthalpy=2, density=2)
        self.assertAlmostEqual(cfg.specific_enthalpy(U)(mip), 1)

    def test_convective_flux(self):

        U = flowfields()
        with self.assertRaises(Exception):
            cfg.get_convective_flux(U)

        U = flowfields(density=1, momentum=(1, 0), enthalpy=1, velocity=(1, 0), pressure=2)
        nptest.assert_almost_equal(cfg.get_convective_flux(U)(mip), (1, 0, 3, 0, 0, 2, 1, 0))

    def test_diffusive_flux(self):

        U = flowfields()
        dU = flowfields()
        with self.assertRaises(Exception):
            cfg.get_diffusive_flux(U, dU)

        U = flowfields(velocity=(1, 0))
        dU = flowfields(strain_rate_tensor=ngs.CF((0, 0.5, 0.5,0),dims=(2,2)), grad_T=(1, 1))
        nptest.assert_almost_equal(cfg.get_diffusive_flux(U, dU)(mip), (0, 0, 0, 0.5, 0.5, 0, 0.5, 1.0))

    def test_transformation(self):
        U = flowfields(density=2, speed_of_sound=2, velocity=(2, 2), specific_energy=11.14285714)
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        A = cfg.primitive_convective_jacobian_x(U)
        B = cfg.primitive_convective_jacobian_y(U)
        An_1 = A * unit_vector[0] + B * unit_vector[1]
        An_2 = cfg.get_primitive_convective_jacobian(U, unit_vector)

        nptest.assert_almost_equal(An_1(mip), An_2(mip))

        A = cfg.conservative_convective_jacobian_x(U)
        B = cfg.conservative_convective_jacobian_y(U)
        An_1 = A * unit_vector[0] + B * unit_vector[1]
        An_2 = cfg.get_conservative_convective_jacobian(U, unit_vector)

        nptest.assert_almost_equal(An_1(mip), An_2(mip))

    def test_characteristic_identity(self):
        U = flowfields(velocity=(1, 0), speed_of_sound=1.4)
        unit_vector = (1, 0)

        with self.assertRaises(ValueError):
            cfg.get_characteristic_identity(U, unit_vector)(mip)

        nptest.assert_almost_equal(cfg.get_characteristic_identity(U, unit_vector, "incoming")(
            mip), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        nptest.assert_almost_equal(cfg.get_characteristic_identity(U, unit_vector, "outgoing")(
            mip), (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))

    def test_farfield_state(self):

        cfg.scaling = "aerodynamic"
        INF = cfg.get_farfield_state((1, 0)).to_py()
        results = {"density": 1.0, "speed_of_sound": 10/3, "temperature": 1/(0.4 * 0.3**2),
                   "pressure": 1/(1.4 * 0.3**2), "velocity": (1.0, 0.0), "inner_energy": 1/(1.4 * 0.3**2 * 0.4),
                   "kinetic_energy": 0.5, "energy": 1/(1.4 * 0.3**2 * 0.4) + 0.5}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])

        cfg.scaling = "aeroacoustic"
        INF = cfg.get_farfield_state((1, 0)).to_py()
        results = {
            "density": 1.0, "speed_of_sound": 10 / 13, "temperature": 1 / (0.4 * (1 + 0.3) ** 2),
            "pressure": 1 / (1.4 * (1 + 0.3) ** 2),
            "velocity": (3 / 13, 0.0),
            "inner_energy": 1 / (1.4 * (1 + 0.3) ** 2 * 0.4),
            "kinetic_energy": 9 / 338, "energy": 1 / (1.4 * (1 + 0.3) ** 2 * 0.4) + 9 / 338}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])

        cfg.scaling = "acoustic"
        INF = cfg.get_farfield_state((1, 0)).to_py()
        results = {"density": 1.0, "speed_of_sound": 1, "temperature": 10/4,
                   "pressure": 1/1.4, "velocity": (0.3, 0.0), "inner_energy": 10/5.6,
                   "kinetic_energy": 0.045, "energy": 10/5.6 + 0.045}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])
