from __future__ import annotations
import unittest
import numpy.testing as nptest
from tests import simplex
import ngsolve as ngs

from dream.compressible.equations import CompressibleEquations
from dream.compressible.state import CompressibleState

def test_equation(throws: bool = False, is_vector: bool = False):

    def wrapper(func):

        def test(self):
            name = func.__name__.split("test_")[1]

            if throws:
                U = CompressibleState()
                with self.assertRaises(ValueError):
                    getattr(self.eq, name)(U)

            value = 1
            if is_vector:
                value = (1, 0)

            setattr(U, name, value)
            self.assertAlmostEqual(getattr(self.eq, name)(U)(self.mip), value)

            func(self)

        return test

    return wrapper

class TestCompressibleEquations(unittest.TestCase):

    def setUp(self) -> None:
        self.eq = CompressibleEquations()

        self.mesh = simplex()
        self.mip = self.mesh(0.25, 0.25)

        self.eq.cfg.Reynolds_number = 2
        self.eq.cfg.Prandtl_number = 1
        self.eq.cfg.dynamic_viscosity = "constant"

    @test_equation(throws=True)
    def test_density(self):
        ...

    @test_equation(throws=True, is_vector=True)
    def test_velocity(self):
        U = CompressibleState(density=2, momentum=(2, 2))
        nptest.assert_almost_equal(self.eq.velocity(U)(self.mip), (1, 1))

    @test_equation(throws=True, is_vector=True)
    def test_momentum(self):
        U = CompressibleState(density=0.5, velocity=(1, 1))
        nptest.assert_almost_equal(self.eq.momentum(U)(self.mip), (0.5, 0.5))

    @test_equation(throws=True)
    def test_pressure(self):
        ...

    @test_equation(throws=True)
    def test_temperature(self):
        ...

    @test_equation(throws=True)
    def test_inner_energy(self):
        U = CompressibleState(energy=2, kinetic_energy=1)
        self.assertAlmostEqual(self.eq.inner_energy(U)(self.mip), 1)

    @test_equation(throws=True)
    def test_specific_inner_energy(self):
        U = CompressibleState(specific_energy=2, specific_kinetic_energy=1)
        self.assertAlmostEqual(self.eq.specific_inner_energy(U)(self.mip), 1)

        U = CompressibleState(inner_energy=2, density=1)
        self.assertAlmostEqual(self.eq.specific_inner_energy(U)(self.mip), 2)

    @test_equation(throws=True)
    def test_kinetic_energy(self):
        U = CompressibleState(density=2, velocity=(2, 2))
        self.assertAlmostEqual(self.eq.kinetic_energy(U)(self.mip), 8)

        U = CompressibleState(density=2, momentum=(2, 2))
        self.assertAlmostEqual(self.eq.kinetic_energy(U)(self.mip), 2)

        U = CompressibleState(energy=2, inner_energy=1)
        self.assertAlmostEqual(self.eq.kinetic_energy(U)(self.mip), 1)

        U = CompressibleState(specific_kinetic_energy=2, density=2)
        self.assertAlmostEqual(self.eq.kinetic_energy(U)(self.mip), 4)

    @test_equation(throws=True)
    def test_specific_kinetic_energy(self):
        U = CompressibleState(velocity=(2, 2))
        self.assertAlmostEqual(self.eq.specific_kinetic_energy(U)(self.mip), 4)

        U = CompressibleState(density=2, momentum=(2, 2))
        self.assertAlmostEqual(self.eq.specific_kinetic_energy(U)(self.mip), 1)

        U = CompressibleState(specific_energy=2, specific_inner_energy=1)
        self.assertAlmostEqual(self.eq.specific_kinetic_energy(U)(self.mip), 1)

        U = CompressibleState(kinetic_energy=2, density=2)
        self.assertAlmostEqual(self.eq.specific_kinetic_energy(U)(self.mip), 1)

    @test_equation(throws=True)
    def test_energy(self):
        U = CompressibleState(specific_energy=2, density=2)
        self.assertAlmostEqual(self.eq.energy(U)(self.mip), 4)

        U = CompressibleState(kinetic_energy=2, inner_energy=2)
        self.assertAlmostEqual(self.eq.energy(U)(self.mip), 4)

    @test_equation(throws=True)
    def test_specific_energy(self):
        U = CompressibleState(energy=2, density=2)
        self.assertAlmostEqual(self.eq.specific_energy(U)(self.mip), 1)

        U = CompressibleState(specific_kinetic_energy=2, specific_inner_energy=2)
        self.assertAlmostEqual(self.eq.specific_energy(U)(self.mip), 4)

    @test_equation(throws=True)
    def test_enthalpy(self):
        U = CompressibleState(pressure=2, energy=2)
        self.assertAlmostEqual(self.eq.enthalpy(U)(self.mip), 4)

        U = CompressibleState(specific_enthalpy=2, density=2)
        self.assertAlmostEqual(self.eq.enthalpy(U)(self.mip), 4)

    @test_equation(throws=True)
    def test_specific_enthalpy(self):
        U = CompressibleState(enthalpy=2, density=2)
        self.assertAlmostEqual(self.eq.specific_enthalpy(U)(self.mip), 1)

    def test_convective_flux(self):

        U = CompressibleState()
        with self.assertRaises(Exception):
            self.eq.convective_flux(U)

        U = CompressibleState(density=1, momentum=(1, 0), enthalpy=1, velocity=(1, 0), pressure=2)
        nptest.assert_almost_equal(self.eq.convective_flux(U)(self.mip), (1, 0, 3, 0, 0, 2, 1, 0))

    def test_diffusive_flux(self):

        U = CompressibleState()
        with self.assertRaises(Exception):
            self.eq.convective_flux(U)

        U = CompressibleState(velocity=(1, 0), deviatoric_stress_tensor=(0, 1, 1, 0), heat_flux=(1, 1))
        nptest.assert_almost_equal(self.eq.diffusive_flux(U)(self.mip), (0, 0, 0, 0.5, 0.5, 0, -0.5, 0))

    def test_transformation(self):
        U = CompressibleState(density=2, speed_of_sound=2, velocity=(2, 2), specific_energy=11.14285714)
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        A = self.eq.primitive_convective_jacobian_x(U)
        B = self.eq.primitive_convective_jacobian_y(U)
        An_1 = A * unit_vector[0] + B * unit_vector[1]
        An_2 = self.eq.get_primitive_convective_jacobian(U, unit_vector)

        nptest.assert_almost_equal(An_1(self.mip), An_2(self.mip))

        A = self.eq.conservative_convective_jacobian_x(U)
        B = self.eq.conservative_convective_jacobian_y(U)
        An_1 = A * unit_vector[0] + B * unit_vector[1]
        An_2 = self.eq.get_conservative_convective_jacobian(U, unit_vector)

        nptest.assert_almost_equal(An_1(self.mip), An_2(self.mip))

    def test_characteristic_identity(self):
        U = CompressibleState(velocity=(1, 0), speed_of_sound=1.4)
        unit_vector = (1, 0)

        with self.assertRaises(ValueError):
            self.eq.get_characteristic_identity(U, unit_vector)(self.mip)

        nptest.assert_almost_equal(self.eq.get_characteristic_identity(U, unit_vector, "incoming")(
            self.mip), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        nptest.assert_almost_equal(self.eq.get_characteristic_identity(U, unit_vector, "outgoing")(
            self.mip), (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))

    def test_farfield_state(self):

        self.eq.cfg.scaling = "aerodynamic"
        INF = self.eq.get_farfield_state((1, 0)).to_python()
        results = {"density": 1.0, "speed_of_sound": 10/3, "temperature": 1/(0.4 * 0.3**2),
                   "pressure": 1/(1.4 * 0.3**2), "velocity": (1.0, 0.0), "inner_energy": 1/(1.4 * 0.3**2 * 0.4),
                   "kinetic_energy": 0.5, "energy": 1/(1.4 * 0.3**2 * 0.4) + 0.5}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])

        self.eq.cfg.scaling = "aeroacoustic"
        INF = self.eq.get_farfield_state((1, 0)).to_python()
        results = {
            "density": 1.0, "speed_of_sound": 10 / 13, "temperature": 1 / (0.4 * (1 + 0.3) ** 2),
            "pressure": 1 / (1.4 * (1 + 0.3) ** 2),
            "velocity": (3 / 13, 0.0),
            "inner_energy": 1 / (1.4 * (1 + 0.3) ** 2 * 0.4),
            "kinetic_energy": 9 / 338, "energy": 1 / (1.4 * (1 + 0.3) ** 2 * 0.4) + 9 / 338}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])

        self.eq.cfg.scaling = "acoustic"
        INF = self.eq.get_farfield_state((1, 0)).to_python()
        results = {"density": 1.0, "speed_of_sound": 1, "temperature": 10/4,
                   "pressure": 1/1.4, "velocity": (0.3, 0.0), "inner_energy": 10/5.6,
                   "kinetic_energy": 0.045, "energy": 10/5.6 + 0.045}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])
